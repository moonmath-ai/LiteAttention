import torch
import os
import sys
import numpy as np
import ctypes

# Preload PyTorch libraries to avoid import errors
torch_lib_path = os.path.dirname(torch.__file__) + '/lib'
libc10_path = os.path.join(torch_lib_path, 'libc10.so')
if os.path.exists(libc10_path):
    try:
        ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
    except:
        pass

try:
    import quant_tma
except ImportError as e:
    print(f"Extension not installed. Error: {e}")
    print("Please run: python setup.py build_ext --inplace")
    sys.exit(1)


def run_correctness_test(M, N, dim_K, name=""):
    """Run correctness test for TMA quantization."""

    device = torch.device('cuda')
    
    # Generate random input data on GPU
    # Use continuous tensors to ensure stride=1 for simple pointer arithmetic
    Q_gpu = torch.randn(M, dim_K, device=device, dtype=torch.float32) * 0.1
    K_gpu = torch.randn(N, dim_K, device=device, dtype=torch.float32) * 0.1

    # Allocate output arrays on GPU
    Q_q_gpu = torch.zeros(M, dim_K, device=device, dtype=torch.int8)
    K_q_gpu = torch.zeros(N, dim_K, device=device, dtype=torch.int8)

    # Calculate number of scale elements
    TileM, TileN, TileK = 64, 64, 64
    num_m_tiles = (M + TileM - 1) // TileM
    num_n_tiles = (N + TileN - 1) // TileN
    num_k_tiles = (dim_K + TileK - 1) // TileK
    num_q_scales = num_m_tiles * num_k_tiles
    num_k_scales = num_n_tiles * num_k_tiles

    q_scales_gpu = torch.zeros(num_q_scales, device=device, dtype=torch.float32)
    k_scales_gpu = torch.zeros(num_k_scales, device=device, dtype=torch.float32)

    try:
        # Use device pointer version for direct GPU access
        quant_tma.tma_load_qk_device(
            Q_gpu.data_ptr(),
            K_gpu.data_ptr(),
            Q_q_gpu.data_ptr(),
            K_q_gpu.data_ptr(),
            q_scales_gpu.data_ptr(),
            k_scales_gpu.data_ptr(),
            M, N, dim_K
        )
        torch.cuda.synchronize()
        
    except Exception as e:
        print(f"  ✗ {name} Q[{M},{dim_K}] K[{N},{dim_K}] - Error: {e}")
        return False

    Q = Q_gpu.cpu().numpy()
    K = K_gpu.cpu().numpy()
    Q_q = Q_q_gpu.cpu().numpy()
    K_q = K_q_gpu.cpu().numpy()
    q_scales = q_scales_gpu.cpu().numpy()
    k_scales = k_scales_gpu.cpu().numpy()

    inv_sqrt_d = 1.0 / np.sqrt(float(dim_K))

    # Dequantize tile by tile
    Q_dequant = np.zeros_like(Q)
    K_dequant = np.zeros_like(K)

    for m_idx in range(num_m_tiles):
        for k_idx in range(num_k_tiles):
            q_block_idx = m_idx * num_k_tiles + k_idx
            m_start = m_idx * TileM
            m_end = min(m_start + TileM, M)
            k_start = k_idx * TileK
            k_end = min(k_start + TileK, dim_K)

            Q_dequant[m_start:m_end, k_start:k_end] = Q_q[m_start:m_end, k_start:k_end].astype(np.float32) * q_scales[q_block_idx]

    for n_idx in range(num_n_tiles):
        for k_idx in range(num_k_tiles):
            k_block_idx = n_idx * num_k_tiles + k_idx
            n_start = n_idx * TileN
            n_end = min(n_start + TileN, N)
            k_start = k_idx * TileK
            k_end = min(k_start + TileK, dim_K)

            K_dequant[n_start:n_end, k_start:k_end] = K_q[n_start:n_end, k_start:k_end].astype(np.float32) * k_scales[k_block_idx]

    # Q should be scaled by inv_sqrt_d in the quantization
    Q_expected = Q * inv_sqrt_d
    K_expected = K

    Q_max_diff = np.max(np.abs(Q_expected - Q_dequant))
    K_max_diff = np.max(np.abs(K_expected - K_dequant))
    has_nan = np.isnan(Q_dequant).any() or np.isnan(K_dequant).any()
    has_inf = np.isinf(Q_dequant).any() or np.isinf(K_dequant).any()

    # Check scale sanity
    scales_ok = (q_scales > 0).all() and (k_scales > 0).all()

    # Quantization error tolerance (int8 quantization introduces error)
    # Relaxed slightly to 0.02 to account for float vs int precision effects
    tolerance = 0.02  
    if Q_max_diff < tolerance and K_max_diff < tolerance and not has_nan and not has_inf and scales_ok:
        print(f"  ✓ {name} Q[{M},{dim_K}] K[{N},{dim_K}] - Q_err={Q_max_diff:.2e}, K_err={K_max_diff:.2e}")
        return True
    else:
        print(f"  ✗ {name} Q[{M},{dim_K}] K[{N},{dim_K}] - Q_err={Q_max_diff:.2e}, K_err={K_max_diff:.2e}, scales_ok={scales_ok}")
        if not scales_ok:
            print(f"    Scale issues: q_scales_valid={np.all(q_scales > 0)}, k_scales_valid={np.all(k_scales > 0)}")
        return False


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"TMA Quantization Correctness Tests - {torch.cuda.get_device_name()}\n")

    test_cases = [
        (64, 64, 64, "Small"),
        (128, 128, 64, "Medium"),
        (512, 512, 64, "Large"),
        (512, 256, 64, "M>N"),
        (256, 512, 64, "N>M"),
        (128, 128, 128, "Large K"),
        (1024, 1024, 256, "Very large"),
    ]

    passed = sum(run_correctness_test(M, N, dim_K, name) for M, N, dim_K, name in test_cases)
    total = len(test_cases)

    print(f"\nResult: {passed}/{total} passed")
    if passed == total:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()