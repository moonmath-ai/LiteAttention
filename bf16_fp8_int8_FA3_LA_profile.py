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

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
    except ImportError:
        raise ImportError("Could not import flash_attn_func. Make sure flash-attention is properly installed.")

from lite_attention import LiteAttention

def main():
    # Configuration
    device = 'cuda'
    batch_size = 2
    seqlen = 16384  # ~16k as requested
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
    
    if True:
        # ============================================================================
        # BF16 Forward Pass
        # ============================================================================
        print("\n" + "="*70)
        print("Running BF16 forward pass...")
        print("="*70)
        torch.cuda.synchronize()
        out_bf16 = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
        )
        torch.cuda.synchronize()
        
        print(f"BF16 Output shape: {out_bf16.shape}")
        print(f"BF16 Output dtype: {out_bf16.dtype}")

        # ============================================================================
        # BF16 Forward Pass (LiteAttention)
        # ============================================================================
        print("\n" + "="*70)
        print("Running BF16 forward pass (LiteAttention)...")
        print("="*70)
        lite_attn = LiteAttention(enable_skipping=False)
        torch.cuda.synchronize()
        out_bf16 = lite_attn(
            q,
            k,
            v,
            scale=softmax_scale,
        )
        torch.cuda.synchronize()
        
        print(f"BF16 Output shape: {out_bf16.shape}")
        print(f"BF16 Output dtype: {out_bf16.dtype}")
        
        # ============================================================================
        # FP8 Forward Pass (without descale)
        # ============================================================================
        print("\n" + "="*70)
        print("Running FP8 forward pass (without descale)...")
        print("="*70)
        
        # Convert to FP8
        q_fp8 = q.to(torch.float8_e4m3fn)
        k_fp8 = k.to(torch.float8_e4m3fn)
        v_fp8 = v.to(torch.float8_e4m3fn)
        
        torch.cuda.synchronize()
        out_fp8_no_descale = flash_attn_func(
            q_fp8,
            k_fp8,
            v_fp8,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
        )
        torch.cuda.synchronize()
        
        print(f"FP8 (no descale) Output shape: {out_fp8_no_descale.shape}")
        print(f"FP8 (no descale) Output dtype: {out_fp8_no_descale.dtype}")
        
        # ============================================================================
        # FP8 Forward Pass (with descale)
        # ============================================================================
        print("\n" + "="*70)
        print("Running FP8 forward pass (with descale)...")
        print("="*70)
        
        # Create descale tensors (required for FP8)
        # These are scaling factors for dequantization
        # Shape must be (batch_size, num_heads_k)
        # For standard attention, num_heads_k = num_heads
        num_heads_k = num_heads  # For GQA/MQA, this would be different
        descale_q = torch.ones(batch_size, num_heads_k, dtype=torch.float32, device=device)
        descale_k = torch.ones(batch_size, num_heads_k, dtype=torch.float32, device=device)
        descale_v = torch.ones(batch_size, num_heads_k, dtype=torch.float32, device=device)
        
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
    
    print("\n" + "="*70)
    print("All forward passes completed successfully!")
    print("="*70)
    print("\nTo benchmark with ncu, run:")
    print(f"  ncu --set full python {__file__}")


if __name__ == "__main__":
    main()


'''
ncu -o bf16_fp8_int8_FA3_LA_profile --kernel-name device_kernel --set full python bf16_fp8_int8_FA3_LA_profile.py
'''