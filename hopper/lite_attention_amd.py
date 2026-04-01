"""LiteAttention AMD: Skip-list sparse attention for MI300X via CK qr-lite pipeline."""

from typing import Optional
import torch
import torch.nn as nn


class LiteAttentionAMD(nn.Module):
    """Double-buffered skip-list attention with alternating phase.

    Args:
        threshold: Log2-scale skip threshold. More negative = more conservative.
        reverse_skip_list: Alternate iteration direction each step (recommended).
    """

    _BLOCK_M = 128
    _BLOCK_N = 128
    _SEED_STEPS = 2  # Dense warmup steps to seed both buffers

    def __init__(self, threshold: float = -5.0, reverse_skip_list: bool = True):
        super().__init__()
        self.threshold = threshold
        self.reverse_skip_list = reverse_skip_list
        self._bufs = [None, None]
        self._phase = 0
        self._step = 0
        self._scale = None
        self._shape_key = None

    def _alloc(self, B, S_q, S_k, H, device):
        key = (B, S_q, S_k, H)
        if self._shape_key == key and self._bufs[0] is not None:
            return
        qt = (S_q + self._BLOCK_M - 1) // self._BLOCK_M
        kt = (S_k + self._BLOCK_N - 1) // self._BLOCK_N
        stride = kt + 2
        n = B * H * qt * stride
        pat = torch.zeros(stride, dtype=torch.int16, device=device)
        pat[0], pat[1], pat[2] = 2, 0, kt
        for i in range(2):
            self._bufs[i] = pat.repeat(B * H * qt)[:n].clone()
        self._phase, self._step, self._shape_key = 0, 0, key

    def forward(self, query, key, value, scale=None, flatten_heads=False):
        from lite_attention._C import lite_attn_fwd
        B, S_q, H, D = query.shape
        if scale is None:
            self._scale = self._scale or D ** -0.5
            scale = self._scale
        self._alloc(B, S_q, key.shape[1], H, query.device)

        phase = 1 - self._phase if self.reverse_skip_list else -1
        thr = -1e30 if self._step < self._SEED_STEPS else self.threshold

        out, _ = lite_attn_fwd(
            query.contiguous(), key.contiguous(), value.contiguous(),
            softmax_scale=scale, is_causal=False,
            skip_read=self._bufs[self._phase],
            skip_write=self._bufs[1 - self._phase],
            skip_threshold=thr,
            skip_reverse_list=self.reverse_skip_list,
            skip_phase=phase,
        )
        self._phase = 1 - self._phase
        self._step += 1
        return out.view(B, S_q, H * D) if flatten_heads else out

    def reset(self):
        self._bufs = [None, None]
        self._phase = self._step = 0
        self._shape_key = None


def patch_model(model: nn.Module, threshold: float = -5.0, attr: str = "attn_op") -> int:
    """Patch self-attention modules to use LiteAttentionAMD. Returns count patched."""
    count = 0
    for _, mod in model.named_modules():
        if hasattr(mod, attr) and getattr(mod, "is_selfattn", False):
            lite = LiteAttentionAMD(threshold=threshold)
            lite = lite.to(next(mod.parameters()).device)

            def _make(m):
                def fn(q, k, v, flatten_heads=True, **kw):
                    return m(q, k, v, flatten_heads=flatten_heads)
                fn.set_context_parallel_group = lambda *a, **kw: None
                return fn

            setattr(mod, attr, _make(lite))
            count += 1
    return count
