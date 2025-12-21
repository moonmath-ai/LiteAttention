#!/usr/bin/env python3
"""
Example script for Flash Attention with FP8 inputs.
Designed for benchmarking with ncu (NVIDIA Compute Profiler).

Configuration:
- Head dimension: 128
- Sequence length: ~16k
- FP8 (float8_e4m3fn) inputs
"""

import torch
import math

def compute_error_metrics(output, reference, name):
    """Compute and print error metrics between output and reference."""
    # Convert both to float32 for accurate error computation
    out_f32 = output.float()
    ref_f32 = reference.float()
    
    # Absolute errors
    abs_diff = (out_f32 - ref_f32).abs()
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    
    # Relative errors (avoid division by zero)
    ref_abs = ref_f32.abs().clamp(min=1e-7)
    rel_diff = abs_diff / ref_abs
    max_rel_error = rel_diff.max().item()
    mean_rel_error = rel_diff.mean().item()
    
    # RMSE
    rmse = torch.sqrt((abs_diff ** 2).mean()).item()
    
    # Cosine similarity (flatten and compute)
    out_flat = out_f32.flatten()
    ref_flat = ref_f32.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(out_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
    
    print(f"\n  Error metrics for {name} vs BF16 FA3 reference:")
    print(f"    Max Absolute Error:  {max_abs_error:.6e}")
    print(f"    Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"    Max Relative Error:  {max_rel_error:.6e}")
    print(f"    Mean Relative Error: {mean_rel_error:.6e}")
    print(f"    RMSE:                {rmse:.6e}")
    print(f"    Cosine Similarity:   {cosine_sim:.8f}")


def quantize_to_fp8(tensor, fp8_dtype=torch.float8_e4m3fn):
    """
    Quantize a tensor to FP8 with proper per-head scaling.
    
    Args:
        tensor: Input tensor of shape (batch, seqlen, num_heads, headdim)
        fp8_dtype: FP8 dtype to use (default: torch.float8_e4m3fn)
    
    Returns:
        fp8_tensor: Quantized FP8 tensor
        descale: Per-head descale factors of shape (batch, num_heads) for dequantization
    """
    FP8_MAX = torch.finfo(fp8_dtype).max  # 448.0 for e4m3fn
    
    # Compute per-head max absolute values
    # Input shape: (batch, seqlen, num_heads, headdim)
    # Output shape: (batch, num_heads)
    amax = tensor.abs().amax(dim=(1, 3))
    
    # Compute scale factor (to map max value to FP8 range)
    scale = (amax / FP8_MAX).clamp(min=1e-12)  # [batch, num_heads]
    
    # Scale input before FP8 conversion
    # Reshape scale for broadcasting: [batch, 1, num_heads, 1]
    scale_bc = scale[:, None, :, None]
    tensor_scaled = tensor / scale_bc
    
    # Convert to FP8
    fp8_tensor = tensor_scaled.to(fp8_dtype)
    
    # Descale is the same as scale (used to multiply back after FP8 ops)
    descale = scale.to(torch.float32)
    
    return fp8_tensor, descale

# try:
#     from flash_attn_interface import flash_attn_func
# except ImportError:
#     try:
#         from flash_attn.flash_attn_interface import flash_attn_func
#     except ImportError:
#         raise ImportError("Could not import flash_attn_func. Make sure flash-attention is properly installed.")

from lite_attention import LiteAttention
from flash_attn_interface import flash_attn_func

def main():
    # Configuration
    device = 'cuda'
    batch_size = 2
    # seqlen = 16384  # ~16k as requested
    seqlen = 19 + 2**14  # ~16k as requested
    num_heads = 32  # Adjust based on your model
    headdim = 128   # As requested
    causal = False  # Set to True for autoregressive (causal) attention, False for bidirectional
    
    # Ensure we're on CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a CUDA-capable GPU.")
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seqlen}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {headdim}")
    print(f"Causal attention: {causal}")
    
    # Create input tensors
    # Shape: (batch, seqlen, num_heads, headdim)
    # Start with bfloat16, will use for both bf16 and FP8 runs
    q = torch.randn(batch_size, seqlen, num_heads, headdim, 
                    device=device, dtype=torch.bfloat16, requires_grad=False)
    k = torch.randn(batch_size, seqlen, num_heads, headdim, 
                    device=device, dtype=torch.bfloat16, requires_grad=False)
    v = torch.randn(batch_size, seqlen, num_heads, headdim, 
                    device=device, dtype=torch.bfloat16, requires_grad=False)
    
    # Compute softmax scale
    softmax_scale = 1.0 / math.sqrt(headdim)
    
    # ============================================================================
    # Warmup Phase
    # ============================================================================
    print("\n" + "="*70)
    print("Running warmup phase...")
    print("="*70)
    
    warmup_iters = 1
    
    # Initialize LiteAttention instances for warmup
    lite_attn_warmup = LiteAttention(enable_skipping=False)
    lite_attn_int8_warmup = LiteAttention(enable_skipping=False, use_int8=True)
    
    # Prepare FP8 tensors for warmup
    q_fp8_warmup, descale_q_warmup = quantize_to_fp8(q)
    k_fp8_warmup, descale_k_warmup = quantize_to_fp8(k)
    v_fp8_warmup, descale_v_warmup = quantize_to_fp8(v)
    
    for i in range(warmup_iters):
        # BF16 FA3
        _ = flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal, window_size=(-1, -1))
        # BF16 LiteAttention
        _ = lite_attn_warmup(q, k, v, scale=softmax_scale)
        # FP8 FA3
        _ = flash_attn_func(q_fp8_warmup, k_fp8_warmup, v_fp8_warmup, softmax_scale=softmax_scale, 
                           causal=causal, window_size=(-1, -1), 
                           q_descale=descale_q_warmup, k_descale=descale_k_warmup, v_descale=descale_v_warmup)
        # INT8 LiteAttention
        _ = lite_attn_int8_warmup(q, k, v, scale=softmax_scale)
    
    torch.cuda.synchronize()
    print(f"Warmup completed ({warmup_iters} iterations per kernel)")
    
    # ============================================================================
    # BF16 Forward Pass (Reference - Vanilla Flash Attention 3)
    # ============================================================================
    print("\n" + "="*70)
    print("Running BF16 forward pass (Reference - Vanilla FA3)...")
    print("="*70)
    torch.cuda.synchronize()
    out_bf16_ref = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    torch.cuda.synchronize()
    
    print(f"BF16 Reference Output shape: {out_bf16_ref.shape}")
    print(f"BF16 Reference Output dtype: {out_bf16_ref.dtype}")

    # ============================================================================
    # BF16 Forward Pass (LiteAttention)
    # ============================================================================
    print("\n" + "="*70)
    print("Running BF16 forward pass (LiteAttention)...")
    print("="*70)
    lite_attn = LiteAttention(enable_skipping=False)
    torch.cuda.synchronize()
    out_bf16_lite = lite_attn(
        q,
        k,
        v,
        scale=softmax_scale,
    )
    torch.cuda.synchronize()
    
    print(f"BF16 LiteAttention Output shape: {out_bf16_lite.shape}")
    print(f"BF16 LiteAttention Output dtype: {out_bf16_lite.dtype}")
    compute_error_metrics(out_bf16_lite, out_bf16_ref, "BF16 LiteAttention")
    
    # ============================================================================
    # FP8 Forward Pass (without descale - naive conversion)
    # ============================================================================
    print("\n" + "="*70)
    print("Running FP8 forward pass (without descale - naive conversion)...")
    print("="*70)
    
    # Naive conversion to FP8 (no scaling, values may clip)
    q_fp8_naive = q.to(torch.float8_e4m3fn)
    k_fp8_naive = k.to(torch.float8_e4m3fn)
    v_fp8_naive = v.to(torch.float8_e4m3fn)
    
    torch.cuda.synchronize()
    out_fp8_no_descale = flash_attn_func(
        q_fp8_naive,
        k_fp8_naive,
        v_fp8_naive,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
    )
    torch.cuda.synchronize()
    
    print(f"FP8 (no descale) Output shape: {out_fp8_no_descale.shape}")
    print(f"FP8 (no descale) Output dtype: {out_fp8_no_descale.dtype}")
    compute_error_metrics(out_fp8_no_descale, out_bf16_ref, "FP8 (no descale)")
    
    # ============================================================================
    # FP8 Forward Pass (with proper quantization and descale)
    # ============================================================================
    print("\n" + "="*70)
    print("Running FP8 forward pass (with proper quantization and descale)...")
    print("="*70)
    
    # Quantize Q, K, V to FP8 with proper per-head scaling
    q_fp8, descale_q = quantize_to_fp8(q)
    k_fp8, descale_k = quantize_to_fp8(k)
    v_fp8, descale_v = quantize_to_fp8(v)
    
    print(f"  Q descale range: [{descale_q.min().item():.6f}, {descale_q.max().item():.6f}]")
    print(f"  K descale range: [{descale_k.min().item():.6f}, {descale_k.max().item():.6f}]")
    print(f"  V descale range: [{descale_v.min().item():.6f}, {descale_v.max().item():.6f}]")
    
    torch.cuda.synchronize()
    out_fp8_with_descale = flash_attn_func(
        q_fp8,
        k_fp8,
        v_fp8,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        q_descale=descale_q,
        k_descale=descale_k,
        v_descale=descale_v,
    )
    torch.cuda.synchronize()
    
    print(f"FP8 (with descale) Output shape: {out_fp8_with_descale.shape}")
    print(f"FP8 (with descale) Output dtype: {out_fp8_with_descale.dtype}")
    compute_error_metrics(out_fp8_with_descale, out_bf16_ref, "FP8 (with descale)")
    
    # ============================================================================
    # INT8 Forward Pass (LiteAttention with int8 enabled)
    # ============================================================================
    print("\n" + "="*70)
    print("Running INT8 forward pass (LiteAttention with int8 enabled)...")
    print("="*70)
    
    # Initialize LiteAttention with int8 enabled
    lite_attn_int8 = LiteAttention(enable_skipping=False, use_int8=True)
    
    torch.cuda.synchronize()
    out_int8 = lite_attn_int8(
        q,  # Using original bfloat16 inputs - LiteAttention will handle quantization
        k,
        v,
        scale=softmax_scale,
    )
    torch.cuda.synchronize()
    
    print(f"INT8 Output shape: {out_int8.shape}")
    print(f"INT8 Output dtype: {out_int8.dtype}")
    compute_error_metrics(out_int8, out_bf16_ref, "INT8 LiteAttention")
    
    # ============================================================================
    # Summary of all error metrics
    # ============================================================================
    print("\n" + "="*70)
    print("ERROR SUMMARY (vs BF16 FA3 Reference)")
    print("="*70)
    
    results = [
        ("BF16 LiteAttention", out_bf16_lite),
        ("FP8 (no descale)", out_fp8_no_descale),
        ("FP8 (with descale)", out_fp8_with_descale),
        ("INT8 LiteAttention", out_int8),
    ]
    
    print(f"\n{'Method':<25} {'Max Abs Err':<14} {'Mean Abs Err':<14} {'RMSE':<14} {'Cosine Sim':<12}")
    print("-" * 80)
    
    for name, output in results:
        out_f32 = output.float()
        ref_f32 = out_bf16_ref.float()
        abs_diff = (out_f32 - ref_f32).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        rmse = torch.sqrt((abs_diff ** 2).mean()).item()
        cosine = torch.nn.functional.cosine_similarity(
            out_f32.flatten().unsqueeze(0), 
            ref_f32.flatten().unsqueeze(0)
        ).item()
        print(f"{name:<25} {max_abs:<14.6e} {mean_abs:<14.6e} {rmse:<14.6e} {cosine:<12.8f}")
    
    print("\n" + "="*70)
    print("All forward passes completed successfully!")
    print("="*70)
    print("\nTo benchmark with ncu, run:")
    print(f"  ncu --set full python {__file__}")


if __name__ == "__main__":
    main()


'''
ncu -o bf16_fp8_int8_FA3_LA_profile%i --kernel-name device_kernel --launch-skip 4 --set full python bf16_fp8_int8_FA3_LA_profile.py
'''