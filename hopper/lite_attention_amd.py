"""
LiteAttention AMD (CK): Skip-list attention for AMD MI300X via Composable Kernel.

Drop-in replacement for the Hopper LiteAttention module, targeting ROCm/AMD GPUs
using the aiter CK tile kernel with qr_lite pipeline.

Usage:
    from lite_attention import LiteAttentionAMD

    attn = LiteAttentionAMD(threshold=-5.0)
    output = attn(query, key, value)            # step 0: dense + writes skip list
    output = attn(query, key, value)            # step 1+: reads skip list, skips tiles

    # Patch into a model:
    for mod in model.modules():
        if hasattr(mod, 'attn_op'):
            mod.attn_op = LiteAttentionAMD(threshold=-5.0).cuda()
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class LiteAttentionAMD(nn.Module):
    """AMD CK lite attention with double-buffered skip lists.

    Args:
        threshold: Log2-scale threshold for tile skipping.
            More negative = keep more tiles (conservative).
            Typical: -6 (conservative), -5 (balanced), -4 (aggressive).
        enable_skipping: Set False to run pure dense attention.
        reverse_skip_list: Skip list direction (matches Hopper convention).
    """

    # CK tile sizes (must match the qr_lite pipeline)
    _BLOCK_M = 128
    _BLOCK_N = 128

    def __init__(
        self,
        threshold: float = -5.0,
        enable_skipping: bool = True,
        reverse_skip_list: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.enable_skipping = enable_skipping
        self.reverse_skip_list = reverse_skip_list

        # Double-buffered skip lists
        self._skip_lists = [None, None]
        self._phase = 0
        self._scale = None

        # Cache for shape changes
        self._last_shape = None

    def _ensure_skip_lists(self, B: int, S_q: int, S_k: int, H: int, device):
        """Allocate or reallocate skip list buffers if shape changed."""
        shape_key = (B, S_q, S_k, H)
        if self._last_shape == shape_key and self._skip_lists[0] is not None:
            return

        num_qt = (S_q + self._BLOCK_M - 1) // self._BLOCK_M
        num_kt = (S_k + self._BLOCK_N - 1) // self._BLOCK_N
        stride_q = num_kt + 2
        stride_h = num_qt * stride_q
        stride_b = H * stride_h
        total = B * stride_b

        # Initialize both buffers as dense (keep all tiles) — vectorized
        for i in range(2):
            buf = torch.zeros(total, dtype=torch.int16, device=device)
            # Build one tile pattern [2, 0, num_kt, 0, 0, ...] of length stride_q
            pattern = torch.zeros(stride_q, dtype=torch.int16, device=device)
            pattern[0] = 2        # length = 2 (one range)
            pattern[1] = 0        # start
            pattern[2] = num_kt   # end
            # Tile it across all (B * H * num_qt) slots
            num_slots = B * H * num_qt
            buf[:num_slots * stride_q] = pattern.repeat(num_slots)
            self._skip_lists[i] = buf

        self._phase = 0
        self._last_shape = shape_key

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
        flatten_heads: bool = False,
    ) -> torch.Tensor:
        """Run attention with skip list optimization.

        Args:
            query: [B, S_q, H, D] bfloat16
            key:   [B, S_k, H, D] bfloat16
            value: [B, S_k, H, D] bfloat16
            scale: Softmax scale (default: 1/sqrt(D))
            flatten_heads: If True, return [B, S_q, H*D] instead of [B, S_q, H, D]

        Returns:
            Attention output tensor.
        """
        from lite_attention._C import lite_attn_fwd

        B, S_q, H, D = query.shape
        S_k = key.shape[1]

        if scale is None:
            if self._scale is None:
                self._scale = D ** -0.5
            scale = self._scale

        if not self.enable_skipping:
            out, _ = lite_attn_fwd(
                query.contiguous(), key.contiguous(), value.contiguous(),
                softmax_scale=scale, is_causal=False,
            )
            return out.view(B, S_q, H * D) if flatten_heads else out

        self._ensure_skip_lists(B, S_q, S_k, H, query.device)

        skip_read = self._skip_lists[self._phase]
        skip_write = self._skip_lists[1 - self._phase]

        # Alternate reverse direction with phase (matches Hopper convention)
        reverse = self.reverse_skip_list and (self._phase == 0)

        out, _ = lite_attn_fwd(
            query.contiguous(), key.contiguous(), value.contiguous(),
            softmax_scale=scale, is_causal=False,
            skip_read=skip_read, skip_write=skip_write,
            skip_threshold=self.threshold,
            skip_reverse_list=reverse,
        )

        # Swap buffers
        self._phase = 1 - self._phase

        return out.view(B, S_q, H * D) if flatten_heads else out

    def reset(self):
        """Reset skip lists to dense (useful between different inputs)."""
        self._skip_lists = [None, None]
        self._phase = 0
        self._last_shape = None

    @torch.no_grad()
    def sparsity(self) -> float:
        """Return the fraction of tiles being skipped (0.0 = dense, 1.0 = all skipped)."""
        read = self._skip_lists[self._phase]
        if read is None:
            return 0.0

        shape = self._last_shape
        if shape is None:
            return 0.0

        B, S_q, S_k, H = shape
        num_qt = (S_q + self._BLOCK_M - 1) // self._BLOCK_M
        num_kt = (S_k + self._BLOCK_N - 1) // self._BLOCK_N
        stride_q = num_kt + 2
        stride_h = num_qt * stride_q
        stride_b = H * stride_h

        total_tiles = 0
        kept_tiles = 0
        read_cpu = read.cpu()
        for b in range(B):
            for h in range(H):
                for qt in range(num_qt):
                    off = b * stride_b + h * stride_h + qt * stride_q
                    ll = read_cpu[off].item()
                    kept = 0
                    for r in range(ll // 2):
                        s = read_cpu[off + 1 + r * 2].item()
                        e = read_cpu[off + 2 + r * 2].item()
                        kept += abs(e - s)
                    kept_tiles += kept
                    total_tiles += num_kt

        return 1.0 - (kept_tiles / max(total_tiles, 1))


def patch_model(model: nn.Module, threshold: float = -5.0, attr: str = "attn_op") -> int:
    """Patch all self-attention modules in a model to use LiteAttentionAMD.

    Args:
        model: The model to patch.
        threshold: Skip list threshold.
        attr: Attribute name of the attention op to replace.

    Returns:
        Number of modules patched.

    Example:
        from lite_attention.lite_attention_amd import patch_model
        n = patch_model(model, threshold=-5.0)
        print(f"Patched {n} attention modules")
    """
    count = 0
    for name, mod in model.named_modules():
        if hasattr(mod, attr) and hasattr(mod, "is_selfattn") and mod.is_selfattn:
            lite = LiteAttentionAMD(threshold=threshold)
            lite = lite.to(next(mod.parameters()).device)

            def make_fn(lite_mod):
                def fn(q, k, v, flatten_heads=True, **kwargs):
                    return lite_mod(q, k, v, flatten_heads=flatten_heads)
                fn.set_context_parallel_group = lambda *a, **kw: None
                return fn

            setattr(mod, attr, make_fn(lite))
            count += 1
    return count
