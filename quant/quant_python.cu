#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cstdint>

namespace py = pybind11;

// Forward declarations from quant.cu
template <typename Element>
void launch_tma_load_qk(
    Element* Q,
    Element* K,
    int8_t* Q_q,
    int8_t* K_q,
    float* q_scales,
    float* k_scales,
    int M, int N, int dim_K,
    cudaStream_t stream);

// Explicit instantiation declarations
extern template void launch_tma_load_qk<float>(
    float* Q, float* K, int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, int M, int N, int dim_K, cudaStream_t stream);

extern template void launch_tma_load_qk<cutlass::bfloat16_t>(
    cutlass::bfloat16_t* Q, cutlass::bfloat16_t* K, int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, int M, int N, int dim_K, cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                   cudaGetErrorString(err)); \
        } \
    } while(0)

// Python wrapper for device pointers - bfloat16 version
void tma_load_qk_device_bfloat16(
    uint64_t Q_ptr,
    uint64_t K_ptr,
    uint64_t Q_q_ptr,
    uint64_t K_q_ptr,
    uint64_t q_scales_ptr,
    uint64_t k_scales_ptr,
    int M, int N, int dim_K)
{
    cutlass::bfloat16_t* d_Q = reinterpret_cast<cutlass::bfloat16_t*>(Q_ptr);
    cutlass::bfloat16_t* d_K = reinterpret_cast<cutlass::bfloat16_t*>(K_ptr);
    int8_t* d_Q_q = reinterpret_cast<int8_t*>(Q_q_ptr);
    int8_t* d_K_q = reinterpret_cast<int8_t*>(K_q_ptr);
    float* d_q_scales = reinterpret_cast<float*>(q_scales_ptr);
    float* d_k_scales = reinterpret_cast<float*>(k_scales_ptr);

    // Launch kernel
    launch_tma_load_qk<cutlass::bfloat16_t>(d_Q, d_K, d_Q_q, d_K_q, d_q_scales, d_k_scales, M, N, dim_K, 0);
    CHECK_CUDA(cudaGetLastError());
}

// Pybind11 module definition
PYBIND11_MODULE(quant_tma, m) {
    m.doc() = "TMA-based quantization for Q and K matrices using TMA on H100";

    m.def("tma_load_qk_device_bf16", &tma_load_qk_device_bfloat16,
          "Load, quantize Q and K matrices using TMA on H100 (device pointers, bfloat16)",
          py::arg("Q_ptr"),
          py::arg("K_ptr"),
          py::arg("Q_q_ptr"),
          py::arg("K_q_ptr"),
          py::arg("q_scales_ptr"),
          py::arg("k_scales_ptr"),
          py::arg("M"),
          py::arg("N"),
          py::arg("dim_K"));
}
