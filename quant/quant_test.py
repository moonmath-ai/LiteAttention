# Test the quantization kernel separately from the attention kernel
# Build the kernel: python setup.py build_ext --inplace

import torch
import numpy as np
import sys

try:
    import quant_tma
except ImportError as e:
    print(f"Extension not installed. Run: python setup.py build_ext --inplace")
    sys.exit(1)

TILE_SEQ_Q = 192 # 192  # kBlockM - Q sequence dimension tile size

def get_tile_seq_k(head_dim):
    """Get BLOCK_N size for K tiles based on head_dim"""
    tile_map = {
        64: 224,
        96: 208,
        128: 128, #176,
        192: 112,
        256: 80,
    }
    if head_dim not in tile_map:
        raise ValueError(f"Unsupported head_dim={head_dim}. Only 64, 96, 128, 192, 256 supported.")
    return tile_map[head_dim]


def quantize_qk(Q, K, dtype='f32'):
    """
    Quantize Q and K tensors with K mean centering.

    Args:
        Q: [batch, seqlen_q, num_heads, head_dim] tensor (BSHD layout)
        K: [batch, seqlen_k, num_heads, head_dim] tensor (BSHD layout)
        dtype: 'f32', 'f16', or 'bf16'

    Returns:
        Q_q: quantized Q [batch, seqlen_q, num_heads, head_dim] int8
        K_q: quantized K (mean-centered) [batch, seqlen_k, num_heads, head_dim] int8
        q_scales: [batch * num_heads, num_seq_tiles_q] per-seq-tile scales (shared across head_dim)
        k_scales: [batch * num_heads, num_seq_tiles_k] per-seq-tile scales (shared across head_dim)
        k_mean: [batch * num_heads, head_dim] per-dimension means
    """
    batch, seqlen_q, num_heads, head_dim = Q.shape
    _, seqlen_k, _, _ = K.shape
    device = Q.device
    total_bh = batch * num_heads

    # Allocate outputs in same layout [batch, seqlen, num_heads, head_dim]
    Q_q = torch.zeros(batch, seqlen_q, num_heads, head_dim, device=device, dtype=torch.int8)
    K_q = torch.zeros(batch, seqlen_k, num_heads, head_dim, device=device, dtype=torch.int8)

    # Kernel uses per-seq-tile quantization (shared across all head_dim), per-dimension for means
    TILE_SEQ_K = get_tile_seq_k(head_dim)
    num_seq_tiles_q = (seqlen_q + TILE_SEQ_Q - 1) // TILE_SEQ_Q
    num_seq_tiles_k = (seqlen_k + TILE_SEQ_K - 1) // TILE_SEQ_K

    q_scales = torch.zeros(total_bh, num_seq_tiles_q, device=device, dtype=torch.float32)
    k_scales = torch.zeros(total_bh, num_seq_tiles_k, device=device, dtype=torch.float32)
    
    # Pre-compute K mean in Python: mean across seq_dim (dim=1)
    k_mean = K.float().mean(dim=1).contiguous()

    # Select kernel based on dtype
    func = {'f16': quant_tma.quantize_qk_f16,
            'bf16': quant_tma.quantize_qk_bf16}[dtype]

    # Pass contiguous 4D tensors directly - kernel expects [batch, seqlen, num_heads, head_dim]
    block_m = TILE_SEQ_Q
    block_n = TILE_SEQ_K
    func(Q.contiguous().data_ptr(), K.contiguous().data_ptr(),
         Q_q.data_ptr(), K_q.data_ptr(),
         q_scales.data_ptr(), k_scales.data_ptr(), k_mean.data_ptr(),
         batch, seqlen_q, seqlen_k, num_heads, head_dim,
         block_m, block_n)

    return Q_q, K_q, q_scales, k_scales, k_mean


def test_correctness(batch, seqlen_q, seqlen_k, num_heads, head_dim, dtype='f32', name=""):
    """Test quantization correctness."""
    device = torch.device('cuda')

    # Map dtype string to torch dtype
    torch_dtype = {'f32': torch.float32, 'f16': torch.float16, 'bf16': torch.bfloat16}[dtype]

    # Generate random inputs with non-zero mean to properly test mean centering
    # Q: standard random with small scale
    Q = torch.randn(batch, seqlen_q, num_heads, head_dim, device=device, dtype=torch_dtype) * 0.1

    # K: add significant non-zero bias to test mean centering
    # Each dimension gets a different offset to test per-dimension mean computation
    K = torch.randn(batch, seqlen_k, num_heads, head_dim, device=device, dtype=torch_dtype) * 0.1
    # Add per-dimension bias: offset increases with dimension index
    dim_offsets = torch.linspace(0.5, 2.0, head_dim, device=device, dtype=torch_dtype)
    K = K + dim_offsets.view(1, 1, 1, head_dim)
    
    # Run quantization
    Q_q, K_q, q_scales, k_scales, k_mean = quantize_qk(Q.contiguous(), K.contiguous(), dtype)
    torch.cuda.synchronize()
    
    # Reference implementation
    inv_sqrt_d = 1.0 / np.sqrt(head_dim)
    
    # K mean: mean across seqlen dimension -> [batch, num_heads, head_dim]
    k_mean_ref = K.float().mean(dim=1)  # [batch, num_heads, head_dim]
    
    # Dequantize and compare - convert to float only when needed
    Q_f = Q.float()
    K_f = K.float()
    Q_q_f = Q_q.float()
    K_q_f = K_q.float()

    num_seq_tiles_q = (seqlen_q + TILE_SEQ_Q - 1) // TILE_SEQ_Q
    TILE_SEQ_K = get_tile_seq_k(head_dim)
    num_seq_tiles_k = (seqlen_k + TILE_SEQ_K - 1) // TILE_SEQ_K

    # Dequantize Q tile by tile using per-seq-tile scales (shared across all head_dim)
    # Dequantized value should equal original * scale
    Q_dequant = torch.zeros_like(Q_f)
    for b in range(batch):
        for h in range(num_heads):
            bh = b * num_heads + h
            for st in range(num_seq_tiles_q):
                s_start, s_end = st * TILE_SEQ_Q, min((st + 1) * TILE_SEQ_Q, seqlen_q)
                scale = q_scales[bh, st].item()
                # Scale is shared across all head_dim
                Q_dequant[b, s_start:s_end, h, :] = \
                    Q_q_f[b, s_start:s_end, h, :] * scale
    
    # Dequantize K tile by tile using per-seq-tile scales (K was mean-centered)
    K_dequant_centered = torch.zeros_like(K_f)
    for b in range(batch):
        for h in range(num_heads):
            bh = b * num_heads + h
            for st in range(num_seq_tiles_k):
                s_start, s_end = st * TILE_SEQ_K, min((st + 1) * TILE_SEQ_K, seqlen_k)
                scale = k_scales[bh, st].item()
                # Scale is shared across all head_dim
                K_dequant_centered[b, s_start:s_end, h, :] = \
                    K_q_f[b, s_start:s_end, h, :] * scale
    
    # Q expected: original Q (dequantized should equal original * scale, which with q_scale=1.0 is just original)
    Q_expected = Q_f
    # K expected: K - mean (mean-centered)
    K_centered_ref = K_f - k_mean_ref.unsqueeze(1)
    
    # Check k_mean
    k_mean_err = (k_mean - k_mean_ref).abs().max().item()

    # Print mean statistics to verify mean centering is working
    k_mean_magnitude = k_mean_ref.abs().mean().item()

    # Check Q quantization error
    Q_err = (Q_expected - Q_dequant).abs().max().item()

    # Check K quantization error (comparing mean-centered versions)
    K_err = (K_centered_ref - K_dequant_centered).abs().max().item()

    # Tolerance for int8 quantization
    tolerance = 0.05
    passed = Q_err < tolerance and K_err < tolerance and k_mean_err < tolerance

    status = "✓" if passed else "✗"
    print(f"  {status} {name} [{batch},{seqlen_q},{seqlen_k},{num_heads},{head_dim}] {dtype}: "
          f"Q_err={Q_err:.2e}, K_err={K_err:.2e}, mean_err={k_mean_err:.2e}, mean_mag={k_mean_magnitude:.3f}")
    
    # Explicitly free large tensors to help with memory management
    del Q, K, Q_f, K_f, Q_q, K_q, Q_q_f, K_q_f, Q_dequant, K_dequant_centered
    del Q_expected, K_centered_ref, k_mean_ref
    torch.cuda.empty_cache()
    
    return passed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    
    print(f"Q/K Quantization Tests - {torch.cuda.get_device_name()}\n")
    
    tests = [
        # (batch, seqlen_q, seqlen_k, num_heads, head_dim, dtype, name)
        # Only fp16/bf16 supported, head_dim limited to 64, 96, 128 (192/256 exceed shared memory)
        # (1, 128, 128, 1, 128, 'bf16', "Small BF16"),
        # (2, 128, 128, 4, 64, 'bf16', "Medium BF16"),
        # (2, 256, 256, 8, 64, 'bf16', "Large BF16"),
        # (1, 512, 256, 4, 128, 'bf16', "Q>K BF16"),
        # (1, 256, 512, 4, 128, 'bf16', "K>Q BF16"),
        # (4, 128, 128, 8, 64, 'bf16', "Multi-batch BF16"),
        # (2, 512, 512, 8, 128, 'bf16', "Medium seq BF16"),
        # (8, 7040, 7040, 8, 128, 'bf16', "Large seq BF16"),
        # (8, 10000, 10000, 8, 128, 'bf16', "Large seq BF16 with non-tile multiple"),
        (2, 16384, 16384, 32, 128, 'bf16', "Large seq BF16"),
    ]
    
    passed = sum(test_correctness(*t) for t in tests)
    print(f"\nResult: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()