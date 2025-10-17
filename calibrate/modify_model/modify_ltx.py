# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm

from diffusers.models.transformers.transformer_ltx import LTXAttention, apply_rotary_emb, LTXVideoRotaryPosEmbed
from functools import partial
from skip_pv_attention.core import skip_pv_attention_ref
from ..utils import PVTrialAbort, precision_metric


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TunePVLTXVideoAttnProcessor:
    r"""
    Processor for implementing attention (SDPA is used by default if you're using PyTorch 2.0). This is used in the LTX
    model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    _attention_backend = None

    def __init__(self, idx, pv_threshold, out_dir="./calibrate/skipratios/ltx", part="generate"):
        self.idx = idx
        self.pv_threshold = pv_threshold       # need to be reset
        self.out_dir = out_dir
        self.part = part
        if is_torch_version("<", "2.0"):
            raise ValueError(
                "LTX attention processors require a minimum PyTorch version of 2.0. Please upgrade your PyTorch installation."
            )

    def __call__(
        self,
        attn: "LTXAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep_index: Optional[int] = None,
        force_dense: bool = False,
        tuning_pv: Optional[float] = None,
        pv_l1: Optional[float] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        per_head_dim = query.shape[-1]
        use_kernel = (
            attention_mask is None and                   # self-attention only
            hidden_states.is_cuda and                    # CUDA only
            query.dtype == torch.bfloat16 and            # kernel expects bf16
            sequence_length >= 128 and                   # N >= 128
            per_head_dim in (64, 128)                    # D in {64, 128}
        )
        

        # -------------------------
        # 1) Dense (baseline pass)
        # -------------------------
        if force_dense:
            hidden_states = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                backend=getattr(self, "_attention_backend", None),
            )

        # -------------------------
        # 2) TRIAL (tuning step)
        # -------------------------
        elif (tuning_pv is not None) and (pv_l1 is not None):
            # Dense reference
            dense_heads = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                backend=getattr(self, "_attention_backend", None),
            )
            # Skip with candidate threshold
            assert use_kernel, "Processor: skip pv kernel is not supported"
            # to [B, H, N, D]
            q = query.permute(0, 2, 1, 3).contiguous()
            k = key.permute(0, 2, 1, 3).contiguous()
            v = value.permute(0, 2, 1, 3).contiguous()
            o, _skip_ratio_per_head = skip_pv_attention_ref(
                q, k, v, is_causal=False, pvthreshd=float(tuning_pv),
            )
            skip_heads = o.permute(0, 2, 1, 3).contiguous()  # back to [B, N, H, D]

            if precision_metric(skip_heads, dense_heads, verbose=False)["L1"] >= float(pv_l1):
                raise PVTrialAbort(f"Layer {self.idx} violated L1 bound at tuning_pv={tuning_pv}")

            # Trials do not alter the forward trajectory—return dense output
            hidden_states = dense_heads

        # -------------------------
        # 3) INFERENCE / FIXED
        # -------------------------
        else:
            thresh = float(tuning_pv) if tuning_pv is not None else float(self.pv_threshold)
            # to [B, H, N, D]
            q = query.permute(0, 2, 1, 3).contiguous()
            k = key.permute(0, 2, 1, 3).contiguous()
            v = value.permute(0, 2, 1, 3).contiguous()
            o, skip_ratio_per_head = skip_pv_attention_ref(
                q, k, v, is_causal=False, pvthreshd=thresh,
            )
            hidden_states = o.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]

            # # Convert skip ratios to numpy (keep whatever length it has: H or B*H)
            # sr = skip_ratio_per_head
            # if isinstance(sr, torch.Tensor):
            #     sr = sr.detach().to(torch.float32).cpu().numpy()
            # else:
            #     sr = np.asarray(sr, dtype=np.float32)

            # # Build path: skipratios/pv_<thr>/<part>/layer_{LL}_t_{TT}.npz
            # out_dir = os.path.join(self.out_dir, self.run_tag, self.part)
            # os.makedirs(out_dir, exist_ok=True)
            # fname = f"layer_{self.idx:02d}_t_{timestep_index:02d}.npz"
            # # Save raw vector + minimal meta (both live in the single .npz)
            # np.savez_compressed(
            #     os.path.join(out_dir, fname),
            #     skip_ratio=sr,            # raw per-head (or per-batch*head) vector
            #     layer=self.idx,
            #     timestep=timestep_index,
            # )
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


@maybe_allow_in_graph
class LTXVideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTXAttention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTXAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep_index: Optional[int] = None,
        force_dense: bool = False,
        tuning_pv: Optional[float] = None,
        pv_l1: Optional[float] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
            timestep_index=timestep_index,
            force_dense=force_dense, tuning_pv=tuning_pv, pv_l1=pv_l1
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


@maybe_allow_in_graph
class LTXVideoTransformer3DModel(
    ModelMixin, ConfigMixin, AttentionMixin, FromOriginalModelMixin, PeftAdapterMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]
    _repeated_blocks = ["LTXVideoTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
            timestep_index = attention_kwargs.pop("timestep_index", None)
            force_dense = bool(attention_kwargs.pop("force_dense", None))
            tuning_pv = attention_kwargs.pop("tuning_pv", None)
            pv_l1 = attention_kwargs.pop("pv_l1", None)
        else:
            lora_scale = 1.0
            timestep_index = None
            force_dense = False
            tuning_pv = None
            pv_l1 = None

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                wrapped = partial(block, timestep_index=timestep_index, force_dense=force_dense, tuning_pv=tuning_pv, pv_l1=pv_l1) # Warp kwargs inside the block
                hidden_states = self._gradient_checkpointing_func(
                    wrapped,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    encoder_attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep_index=timestep_index,
                    force_dense=force_dense, tuning_pv=tuning_pv, pv_l1=pv_l1
                )

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def set_pv_tune_ltx(
    model: LTXVideoTransformer3DModel,
    verbose=False,
    pv_threshold=20.0,
    out_dir="./calibrate/skipratios/ltx",
    part="generate"
):
    for idx, block in enumerate(model.transformer_blocks):
        block.attn1.verbose = verbose
        origin_processor = block.attn1.get_processor()
        processor = TunePVLTXVideoAttnProcessor(idx, pv_threshold, out_dir=out_dir, part=part)
        block.attn1.set_processor(processor)
        if not hasattr(block.attn1, "origin_processor"):
            block.attn1.origin_processor = origin_processor

# Reset pv_threshold for denoising
def reset_pv_tune_ltx(
    model: LTXVideoTransformer3DModel,
    pv_threshold: float,
    part="denoise",
):
    for block in model.transformer_blocks:
        proc = block.attn1.get_processor()
        if hasattr(proc, "pv_threshold"):
            proc.pv_threshold = pv_threshold
        if hasattr(proc, "part"):
            proc.part = part
        # run_tag and out_dir remain unchanged, so both parts write under the same run folder