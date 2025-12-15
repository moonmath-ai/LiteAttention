#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cstdint>

namespace py = pybind11;

// Forward declaration from quant.cu
template <typename Element>
void launch_quantize_qk_runtime(
    const Element* Q, const Element* K,
    int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, float* k_mean,
    int batch, int seqlen_q, int seqlen_k, int num_heads,
    int head_dim, int block_m, int block_n,
    cudaStream_t stream);

// Explicit instantiation declarations
extern template void launch_quantize_qk_runtime<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*, int8_t*, int8_t*, float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);

extern template void launch_quantize_qk_runtime<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*, int8_t*, int8_t*, float*, float*, float*,
    int, int, int, int, int, int, int, cudaStream_t);

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
} while(0)

// Python wrapper - float16
void quantize_qk_f16(
    uint64_t Q_ptr, uint64_t K_ptr,
    uint64_t Q_q_ptr, uint64_t K_q_ptr,
    uint64_t q_scales_ptr, uint64_t k_scales_ptr, uint64_t k_mean_ptr,
    int batch, int seqlen_q, int seqlen_k, int num_heads, int head_dim,
    int block_m, int block_n)
{
    launch_quantize_qk_runtime<cutlass::half_t>(
        reinterpret_cast<cutlass::half_t*>(Q_ptr),
        reinterpret_cast<cutlass::half_t*>(K_ptr),
        reinterpret_cast<int8_t*>(Q_q_ptr),
        reinterpret_cast<int8_t*>(K_q_ptr),
        reinterpret_cast<float*>(q_scales_ptr),
        reinterpret_cast<float*>(k_scales_ptr),
        reinterpret_cast<float*>(k_mean_ptr),
        batch, seqlen_q, seqlen_k, num_heads, head_dim, block_m, block_n, 0);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(0));
}

// Python wrapper - bfloat16
void quantize_qk_bf16(
    uint64_t Q_ptr, uint64_t K_ptr,
    uint64_t Q_q_ptr, uint64_t K_q_ptr,
    uint64_t q_scales_ptr, uint64_t k_scales_ptr, uint64_t k_mean_ptr,
    int batch, int seqlen_q, int seqlen_k, int num_heads, int head_dim,
    int block_m, int block_n)
{
    launch_quantize_qk_runtime<cutlass::bfloat16_t>(
        reinterpret_cast<cutlass::bfloat16_t*>(Q_ptr),
        reinterpret_cast<cutlass::bfloat16_t*>(K_ptr),
        reinterpret_cast<int8_t*>(Q_q_ptr),
        reinterpret_cast<int8_t*>(K_q_ptr),
        reinterpret_cast<float*>(q_scales_ptr),
        reinterpret_cast<float*>(k_scales_ptr),
        reinterpret_cast<float*>(k_mean_ptr),
        batch, seqlen_q, seqlen_k, num_heads, head_dim, block_m, block_n, 0);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(0));
}

PYBIND11_MODULE(quant_tma, m) {
    m.doc() = "TMA-based Q/K quantization with mean centering for attention (FP16/BF16 only)";

    m.def("quantize_qk_f16", &quantize_qk_f16,
          "Quantize Q and K (float16) with K mean centering",
          py::arg("Q_ptr"), py::arg("K_ptr"),
          py::arg("Q_q_ptr"), py::arg("K_q_ptr"),
          py::arg("q_scales_ptr"), py::arg("k_scales_ptr"), py::arg("k_mean_ptr"),
          py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
          py::arg("num_heads"), py::arg("head_dim"),
          py::arg("block_m"), py::arg("block_n"));

    m.def("quantize_qk_bf16", &quantize_qk_bf16,
          "Quantize Q and K (bfloat16) with K mean centering",
          py::arg("Q_ptr"), py::arg("K_ptr"),
          py::arg("Q_q_ptr"), py::arg("K_q_ptr"),
          py::arg("q_scales_ptr"), py::arg("k_scales_ptr"), py::arg("k_mean_ptr"),
          py::arg("batch"), py::arg("seqlen_q"), py::arg("seqlen_k"),
          py::arg("num_heads"), py::arg("head_dim"),
          py::arg("block_m"), py::arg("block_n"));
}
