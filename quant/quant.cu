#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_utils.h>
#include <cstdint>
#include <cmath>

using namespace cute;

template <typename Element_, int TileM_, int TileN_, int TileK_>
struct TileConfig {
    using Element = Element_;
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int NumThreads = 128;
};

template <typename Config, class TmaLoadQ, class TmaLoadK>
__global__ void tma_quantize_kernel(
    __grid_constant__ const TmaLoadQ tma_q,
    __grid_constant__ const TmaLoadK tma_k,
    int8_t* __restrict__ Q_q_ptr,
    int8_t* __restrict__ K_q_ptr,
    float* __restrict__ q_scales,
    float* __restrict__ k_scales,
    int M, int N, int dim_K,
    float inv_sqrt_d)
{
    using Element = typename Config::Element;
    constexpr int TileM = Config::TileM;
    constexpr int TileN = Config::TileN;
    constexpr int TileK = Config::TileK;
    constexpr int NumThreads = Config::NumThreads;

    // Shared memory layout - TMA requires 128-byte alignment for data buffers
    extern __shared__ __align__(128) char smem_raw[];

    // Barriers - must be 8-byte aligned (placed in static shared memory)
    __shared__ __align__(8) uint64_t tma_q_mbar;
    __shared__ __align__(8) uint64_t tma_k_mbar;

    // Data buffers in dynamic shared memory (already 128-byte aligned via __align__)
    Element* smem_q = reinterpret_cast<Element*>(smem_raw);
    Element* smem_k = smem_q + (TileM * TileK);

    int m_idx = blockIdx.x;
    int n_idx = blockIdx.y;
    int k_idx = blockIdx.z;

    if (m_idx * TileM >= M || n_idx * TileN >= N || k_idx * TileK >= dim_K)
        return;

    Tensor Q_smem = make_tensor(make_smem_ptr(smem_q), Shape<Int<TileM>, Int<TileK>>{}, LayoutRight{});
    Tensor K_smem = make_tensor(make_smem_ptr(smem_k), Shape<Int<TileN>, Int<TileK>>{}, LayoutRight{});

    if (threadIdx.x == 0) {
        initialize_barrier(tma_q_mbar, 1);
        initialize_barrier(tma_k_mbar, 1);

        set_barrier_transaction_bytes(tma_q_mbar, TileM * TileK * sizeof(Element));
        set_barrier_transaction_bytes(tma_k_mbar, TileN * TileK * sizeof(Element));

        Tensor mQ = tma_q.get_tma_tensor(make_shape(M, dim_K));
        Tensor gQ = local_tile(mQ, Shape<Int<TileM>, Int<TileK>>{}, make_coord(m_idx, k_idx));
        auto tma_q_slice = tma_q.get_slice(Int<0>{});
        copy(tma_q.with(tma_q_mbar), tma_q_slice.partition_S(gQ), tma_q_slice.partition_D(Q_smem));

        Tensor mK = tma_k.get_tma_tensor(make_shape(N, dim_K));
        Tensor gK = local_tile(mK, Shape<Int<TileN>, Int<TileK>>{}, make_coord(n_idx, k_idx));
        auto tma_k_slice = tma_k.get_slice(Int<0>{});
        copy(tma_k.with(tma_k_mbar), tma_k_slice.partition_S(gK), tma_k_slice.partition_D(K_smem));
    }
    __syncthreads();

    wait_barrier(tma_q_mbar, 0);
    wait_barrier(tma_k_mbar, 0);
    __syncthreads();

    auto thr_layout = make_layout(Int<NumThreads>{});

    Tensor tQ_smem = local_partition(Q_smem, thr_layout, threadIdx.x);
    Tensor tK_smem = local_partition(K_smem, thr_layout, threadIdx.x);

    auto coord_tensor = make_identity_tensor(Shape<Int<TileM>, Int<TileK>>{});
    Tensor tCoord = local_partition(coord_tensor, thr_layout, threadIdx.x);

    float local_max[2] = {0.f, 0.f};

    #pragma unroll
    for (int i = 0; i < size(tQ_smem); ++i) {
        local_max[0] = fmaxf(local_max[0], fabsf(static_cast<float>(tQ_smem(i))));
        local_max[1] = fmaxf(local_max[1], fabsf(static_cast<float>(tK_smem(i))));
    }

    blockReduceMax<float, 2>(local_max);

    __shared__ float shared_q_inv_scale;
    __shared__ float shared_k_inv_scale;

    if (threadIdx.x == 0) {
        float q_max = local_max[0];
        float k_max = local_max[1];

        float q_scale = (q_max > 0.f) ? (q_max * inv_sqrt_d / 127.f) : 1.f;
        float k_scale = (k_max > 0.f) ? (k_max / 127.f) : 1.f;

        shared_q_inv_scale = (q_max > 0.f) ? (127.f / q_max) : 0.f;
        shared_k_inv_scale = (k_max > 0.f) ? (127.f / k_max) : 0.f;

        int num_k_tiles = (dim_K + TileK - 1) / TileK;
        q_scales[m_idx * num_k_tiles + k_idx] = q_scale;
        k_scales[n_idx * num_k_tiles + k_idx] = k_scale;
    }
    __syncthreads();

    float q_inv_scale = shared_q_inv_scale;
    float k_inv_scale = shared_k_inv_scale;

    int q_row_base = m_idx * TileM;
    int k_row_base = n_idx * TileN;
    int col_base = k_idx * TileK;

    #pragma unroll
    for (int i = 0; i < size(tQ_smem); ++i) {
        int local_row = get<0>(tCoord(i));
        int local_col = get<1>(tCoord(i));

        int global_q_row = q_row_base + local_row;
        int global_k_row = k_row_base + local_row;
        int global_col = col_base + local_col;

        // Quantize Q
        if (global_q_row < M && global_col < dim_K) {
            float v = static_cast<float>(tQ_smem(i));
            float q = rintf(v * q_inv_scale);
            q = fmaxf(-127.f, fminf(127.f, q));
            Q_q_ptr[global_q_row * dim_K + global_col] = static_cast<int8_t>(q);
        }

        // Quantize K
        if (global_k_row < N && global_col < dim_K) {
            float v = static_cast<float>(tK_smem(i));
            float q = rintf(v * k_inv_scale);
            q = fmaxf(-127.f, fminf(127.f, q));
            K_q_ptr[global_k_row * dim_K + global_col] = static_cast<int8_t>(q);
        }
    }
}

template <typename Element>
void launch_tma_load_qk(
    Element* Q,
    Element* K,
    int8_t* Q_q,
    int8_t* K_q,
    float* q_scales,
    float* k_scales,
    int M, int N, int dim_K,
    cudaStream_t stream = 0)
{
    using Config = TileConfig<Element, 64, 64, 64>;
    constexpr int TileM = Config::TileM;
    constexpr int TileN = Config::TileN;
    constexpr int TileK = Config::TileK;

    auto Q_gmem_layout = make_layout(make_shape(M, dim_K), LayoutRight{});
    auto Q_gmem_tensor = make_tensor(make_gmem_ptr(Q), Q_gmem_layout);
    auto Q_smem_layout = make_layout(Shape<Int<TileM>, Int<TileK>>{}, LayoutRight{});
    auto tma_q = make_tma_copy(SM90_TMA_LOAD{}, Q_gmem_tensor, Q_smem_layout);

    auto K_gmem_layout = make_layout(make_shape(N, dim_K), LayoutRight{});
    auto K_gmem_tensor = make_tensor(make_gmem_ptr(K), K_gmem_layout);
    auto K_smem_layout = make_layout(Shape<Int<TileN>, Int<TileK>>{}, LayoutRight{});
    auto tma_k = make_tma_copy(SM90_TMA_LOAD{}, K_gmem_tensor, K_smem_layout);

    dim3 grid(
        (M + TileM - 1) / TileM,
        (N + TileN - 1) / TileN,
        (dim_K + TileK - 1) / TileK
    );
    dim3 block(Config::NumThreads);

    size_t smem_size = (TileM * TileK + TileN * TileK) * sizeof(Element);
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim_K));

    tma_quantize_kernel<Config><<<grid, block, smem_size, stream>>>(
        tma_q, tma_k,
        Q_q, K_q,
        q_scales, k_scales,
        M, N, dim_K,
        inv_sqrt_d);
}

template void launch_tma_load_qk<cutlass::bfloat16_t>(
    cutlass::bfloat16_t* Q,
    cutlass::bfloat16_t* K,
    int8_t* Q_q,
    int8_t* K_q,
    float* q_scales,
    float* k_scales,
    int M, int N, int dim_K,
    cudaStream_t stream);