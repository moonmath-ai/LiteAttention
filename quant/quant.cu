#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h> 

#include <cub/cub.cuh>

#include <assert.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

using namespace cute;

constexpr int kNumThreads = 256;

// 4D Shape: [seqlen, head_dim, num_heads, batch]
using Shape4D = cute::Shape<int, int, int, int>;

template<int ROWS, int HEAD_DIM, typename Element>
struct SmemConfig {
    // Input: row-major for TMA load
    using SmemLayoutIn = Layout<Shape<Int<ROWS>, Int<HEAD_DIM>>,
                                Stride<Int<HEAD_DIM>, _1>>;
    // Output: row-major for TMA store
    using SmemLayoutOut = Layout<Shape<Int<ROWS>, Int<HEAD_DIM>>,
                                 Stride<Int<HEAD_DIM>, _1>>;
};

template <int HEAD_DIM, typename Element, typename SmemLayoutIn, typename SmemLayoutOut>
struct __align__(128) TmaSharedStorage {
    using BlockReduce = cub::BlockReduce<float, kNumThreads>;
    
    __align__(128) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutIn>>   smem_in;
    __align__(128) cute::ArrayEngine<int8_t,  cute::cosize_v<SmemLayoutOut>>  smem_out;
    
    __align__(16) typename BlockReduce::TempStorage cub_storage;

    __align__(16) cute::uint64_t tma_load_barrier;
    __align__(16)  float tile_inv_scale;
    __align__(16)  float tile_means[HEAD_DIM];
};


// Helper struct for reducing 8 floats simultaneously in CUB
struct Array8 {
    float data[8];
    
    __device__ __forceinline__ static Array8 add(const Array8& a, const Array8& b) {
        Array8 res;
        #pragma unroll
        for(int i=0; i<8; ++i) {
            res.data[i] = a.data[i] + b.data[i];
        }
        return res;
    }
};

struct SumArray8 {
    __device__ __forceinline__ Array8 operator()(const Array8& a, const Array8& b) const {
        return Array8::add(a, b);
    }
};

// -----------------------------------------------------------------------------
// Kernel 1: Compute K Mean
// -----------------------------------------------------------------------------
template <typename Element>
__global__ void compute_k_mean_kernel(
    const Element* __restrict__ K,
    float* __restrict__ k_mean,
    int seqlen, 
    int num_heads, 
    int head_dim,
    int64_t stride_b, 
    int64_t stride_s, 
    int64_t stride_h)
{
    constexpr int kVecElem = 8; 
    
    int bh_idx    = blockIdx.x; // Batch * NumHeads
    int chunk_idx = blockIdx.y; // Channel Chunk Index
    int tid       = threadIdx.x;
    
    int b = bh_idx / num_heads;
    int h = bh_idx % num_heads;
    
    int d_start = chunk_idx * kVecElem;
    
    // Base Pointer Arithmetic (in bytes)
    // K is [Batch, Seq, Head, Dim] conceptually for these strides
    const char* ptr_base = reinterpret_cast<const char*>(K) + 
    (b * stride_b + h * stride_h) * sizeof(Element) + 
    d_start * sizeof(Element);
    
    // Accumulate 8 sums in registers
    Array8 local_sums;
    #pragma unroll
    for(int i=0; i<kVecElem; ++i) local_sums.data[i] = 0.0f;
    
    // Grid-Stride Loop over Sequence
    for (int s = tid; s < seqlen; s += blockDim.x) {
        // Calculate address for this sequence step, s * stride_s jumps over the sequence dimension
        const char* curr_ptr = ptr_base + s * stride_s * sizeof(Element);
        
        // Vector Load (16 bytes)
        uint4 loaded = *reinterpret_cast<const uint4*>(curr_ptr);
        
        // Interpret bits as Elements and accumulate
        const Element* vals = reinterpret_cast<const Element*>(&loaded);
        
        #pragma unroll
        for(int i=0; i<kVecElem; ++i) {
            local_sums.data[i] += static_cast<float>(vals[i]);
        }
    }

    // Block Reduction
    using BlockReduce = cub::BlockReduce<Array8, kNumThreads>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Array8 block_sum = BlockReduce(temp_storage).Reduce(local_sums, SumArray8());

    // Write result
    if (tid == 0) {
        float inv_seqlen = 1.0f / static_cast<float>(seqlen);
        int out_offset = bh_idx * head_dim + d_start;

        #pragma unroll
        for (int i = 0; i < kVecElem; ++i) {
            if (d_start + i < head_dim) {
                k_mean[out_offset + i] = block_sum.data[i] * inv_seqlen;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Quantize Q 
// -----------------------------------------------------------------------------
template <
    int HEAD_DIM,
    int BLOCK_M,
    typename Element,
    typename TmaLoad,
    typename TmaStore>
__global__ void __launch_bounds__(kNumThreads)
quantize_q_kernel(
    CUTE_GRID_CONSTANT TmaLoad const tma_load,
    CUTE_GRID_CONSTANT TmaStore const tma_store,
    int seqlen_q,
    int num_heads,
    int batch,
    float* __restrict__ q_scales,
    int num_seq_tiles_q,
    float inv_sqrt_d)
{
    using SmemLayoutIn = typename SmemConfig<BLOCK_M, HEAD_DIM, Element>::SmemLayoutIn;
    using SmemLayoutOut = typename SmemConfig<BLOCK_M, HEAD_DIM, Element>::SmemLayoutOut;
    using SharedStorage = TmaSharedStorage<HEAD_DIM, Element, SmemLayoutIn, SmemLayoutOut>;
    using BlockReduce = typename SharedStorage::BlockReduce;

    extern __shared__ char shared_mem[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_mem);

    int seq_tile_idx = blockIdx.x;
    int bidh         = blockIdx.y;
    int bidb         = blockIdx.z;

    // 1. TMA Load
    Shape4D shape_Q = make_shape(seqlen_q, HEAD_DIM, num_heads, batch);
    Tensor mQ       = tma_load.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
    Tensor gQ       = local_tile(mQ, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));
    Tensor sQ_in    = make_tensor(make_smem_ptr(storage.smem_in.begin()), SmemLayoutIn{});

    auto tma_load_slice = tma_load.get_slice(_0{});
    Tensor tAgQ = tma_load_slice.partition_S(gQ);
    Tensor tAsQ = tma_load_slice.partition_D(sQ_in);

    if (threadIdx.x == 0) {
        initialize_barrier(storage.tma_load_barrier, 1);
        set_barrier_transaction_bytes(storage.tma_load_barrier, sizeof(Element) * size(sQ_in));
        copy(tma_load.with(storage.tma_load_barrier), tAgQ, tAsQ);
    }
    __syncthreads();
    wait_barrier(storage.tma_load_barrier, 0);
    __syncthreads();

    // 2. Data Partitioning
    Tensor sQ_flat = make_tensor(sQ_in.data(), make_layout(size(sQ_in)));
    Tensor tTsQ    = local_partition(sQ_flat, Layout<Shape<Int<kNumThreads>>>{}, threadIdx.x);

    int tile_row_start = seq_tile_idx * BLOCK_M;
    int rows_remaining = seqlen_q - tile_row_start;
    int valid_rows     = (rows_remaining > 0) ? min(BLOCK_M, rows_remaining) : 0;
    int valid_elems    = valid_rows * HEAD_DIM;

    // Find Max
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < size(tTsQ); ++i) {
        int original_idx = threadIdx.x + i * kNumThreads;
        if (original_idx < valid_elems) {
            float v = static_cast<float>(tTsQ(i));
            local_max = fmaxf(local_max, fabsf(v));
        }
    }

    float tile_max_raw = BlockReduce(storage.cub_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0) {
        float real_scale = (tile_max_raw * inv_sqrt_d) / 127.f;
        real_scale = fmaxf(real_scale, 1e-6f); 

        int idx_scale = bidb * num_heads * num_seq_tiles_q + bidh * num_seq_tiles_q + seq_tile_idx;
        q_scales[idx_scale] = real_scale;

        storage.tile_inv_scale = 127.f / fmaxf(tile_max_raw, 1e-6f);
    }
    __syncthreads(); 

    // Quantize & Store
    Tensor sQ_out     = make_tensor(make_smem_ptr(storage.smem_out.begin()), SmemLayoutOut{});
    Tensor sQ_out_flat= make_tensor(sQ_out.data(), make_layout(size(sQ_out)));
    Tensor tTsQ_out   = local_partition(sQ_out_flat, Layout<Shape<Int<kNumThreads>>>{}, threadIdx.x);

    float effective_inv_scale = storage.tile_inv_scale;

    #pragma unroll
    for (int i = 0; i < size(tTsQ_out); ++i) {
        int original_idx = threadIdx.x + i * kNumThreads;
        if (original_idx < valid_elems) {
            float v = static_cast<float>(tTsQ(i));
            tTsQ_out(i) = static_cast<int8_t>(max(-127, min(127, __float2int_rn(v * effective_inv_scale))));
        } else {
            tTsQ_out(i) = 0;
        }
    }
    __syncthreads();

    // 3. TMA Store
    Tensor mQ_out = tma_store.get_tma_tensor(shape_Q)(_, _, bidh, bidb);
    Tensor gQ_out = local_tile(mQ_out, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));

    auto tma_store_slice = tma_store.get_slice(_0{});
    Tensor tAsQ_out = tma_store_slice.partition_S(sQ_out);
    Tensor tAgQ_out = tma_store_slice.partition_D(gQ_out);

    if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store, tAsQ_out, tAgQ_out);
        tma_store_wait<0>();
    }
}

// -----------------------------------------------------------------------------
// Kernel 3: Quantize K
// -----------------------------------------------------------------------------

template <
    int HEAD_DIM,
    int BLOCK_N,
    typename Element,
    typename TmaLoad,
    typename TmaStore>
__global__ void __launch_bounds__(kNumThreads)
quantize_k_kernel(
    CUTE_GRID_CONSTANT TmaLoad const tma_load,
    CUTE_GRID_CONSTANT TmaStore const tma_store,
    int seqlen_k,
    int num_heads,
    int batch,
    Shape4D shape_K,
    const float* __restrict__ k_mean,
    float* __restrict__ k_scales,
    int num_seq_tiles_k)
{
    using SmemLayoutIn = typename SmemConfig<BLOCK_N, HEAD_DIM, Element>::SmemLayoutIn;
    using SmemLayoutOut = typename SmemConfig<BLOCK_N, HEAD_DIM, Element>::SmemLayoutOut;
    using SharedStorage = TmaSharedStorage<HEAD_DIM, Element, SmemLayoutIn, SmemLayoutOut>;
    using BlockReduce = typename SharedStorage::BlockReduce;

    extern __shared__ char shared_mem[];
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_mem);

    int seq_tile_idx = blockIdx.x;
    int bidh         = blockIdx.y;
    int bidb         = blockIdx.z;

    int bh_idx = bidb * num_heads + bidh;
    if (threadIdx.x < HEAD_DIM) {
        storage.tile_means[threadIdx.x] = k_mean[bh_idx * HEAD_DIM + threadIdx.x];
    }
    __syncthreads();

    // 1. TMA Load
    Tensor mK = tma_load.get_tma_tensor(shape_K)(_, _, bidh, bidb);
    Tensor gK = local_tile(mK, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));
    Tensor sK_in = make_tensor(make_smem_ptr(storage.smem_in.begin()), SmemLayoutIn{});

    auto tma_load_slice = tma_load.get_slice(_0{});
    Tensor tAgK = tma_load_slice.partition_S(gK);
    Tensor tAsK = tma_load_slice.partition_D(sK_in);

    if (threadIdx.x == 0) {
        initialize_barrier(storage.tma_load_barrier, 1);
        set_barrier_transaction_bytes(storage.tma_load_barrier, sizeof(Element) * size(sK_in));
        copy(tma_load.with(storage.tma_load_barrier), tAgK, tAsK);
    }
    __syncthreads();
    wait_barrier(storage.tma_load_barrier, 0);
    __syncthreads();

    // 2. Compute
    Tensor sK_flat = make_tensor(sK_in.data(), make_layout(size(sK_in)));
    Tensor tTsK    = local_partition(sK_flat, Layout<Shape<Int<kNumThreads>>>{}, threadIdx.x);

    int tile_row_start = seq_tile_idx * BLOCK_N;
    int rows_remaining = seqlen_k - tile_row_start;
    int valid_rows     = (rows_remaining > 0) ? min(BLOCK_N, rows_remaining) : 0;
    int valid_elems    = valid_rows * HEAD_DIM;

    // Find Max (Centered)
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < size(tTsK); ++i) {
        int original_idx = threadIdx.x + i * kNumThreads;
        if (original_idx < valid_elems) {
            int d = original_idx % HEAD_DIM;
            float v = static_cast<float>(tTsK(i)) - storage.tile_means[d];
            local_max = fmaxf(local_max, fabsf(v));
        }
    }
    
    float tile_max = BlockReduce(storage.cub_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0) {
        float scale = tile_max / 127.f;
        scale = fmaxf(scale, 1e-6f);
        storage.tile_inv_scale = 1.f / scale;
        
        int idx_scale = bh_idx * num_seq_tiles_k + seq_tile_idx;
        k_scales[idx_scale] = scale;
    }
    __syncthreads();

    // Quantize & Store
    Tensor sK_out     = make_tensor(make_smem_ptr(storage.smem_out.begin()), SmemLayoutOut{});
    Tensor sK_out_flat= make_tensor(sK_out.data(), make_layout(size(sK_out)));
    Tensor tTsK_out   = local_partition(sK_out_flat, Layout<Shape<Int<kNumThreads>>>{}, threadIdx.x);
    
    float inv_scale = storage.tile_inv_scale;

    #pragma unroll
    for (int i = 0; i < size(tTsK_out); ++i) {
        int original_idx = threadIdx.x + i * kNumThreads;
        if (original_idx < valid_elems) {
            int d = original_idx % HEAD_DIM;
            float v = static_cast<float>(tTsK(i)) - storage.tile_means[d];
            tTsK_out(i) = static_cast<int8_t>(max(-127, min(127, __float2int_rn(v * inv_scale))));
        } else {
            tTsK_out(i) = 0;
        }
    }
    __syncthreads();

    // 3. TMA Store
    Tensor mK_out = tma_store.get_tma_tensor(shape_K)(_, _, bidh, bidb);
    Tensor gK_out = local_tile(mK_out, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, make_coord(seq_tile_idx, 0));

    auto tma_store_slice = tma_store.get_slice(_0{});
    Tensor tAsK_out = tma_store_slice.partition_S(sK_out);
    Tensor tAgK_out = tma_store_slice.partition_D(gK_out);

    if (threadIdx.x == 0) {
        tma_store_fence();
        copy(tma_store, tAsK_out, tAgK_out);
        tma_store_wait<0>();
    }
}

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
struct QKConfigAllowed {
    static constexpr bool value =
        (HEAD_DIM ==  64 && BLOCK_M == 128 && BLOCK_N == 224) ||
        (HEAD_DIM ==  96 && BLOCK_M == 128 && BLOCK_N == 208) ||
        (HEAD_DIM == 128 && BLOCK_M == 128 && BLOCK_N == 128) ||
        (HEAD_DIM == 128 && BLOCK_M == 128 && BLOCK_N == 176);
};

template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, typename Element>
void launch_quantize_qk_config(
    const Element* Q, const Element* K,
    int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, float* k_mean,
    int batch, int seqlen_q, int seqlen_k, int num_heads,
    cudaStream_t stream)
{
    static_assert(QKConfigAllowed<HEAD_DIM, BLOCK_M, BLOCK_N>::value, "Unsupported Config");

    using SmemConfigQ = SmemConfig<BLOCK_M, HEAD_DIM, Element>;
    using SmemConfigK = SmemConfig<BLOCK_N, HEAD_DIM, Element>;

    int num_seq_tiles_q = (seqlen_q + BLOCK_M - 1) / BLOCK_M;
    int num_seq_tiles_k = (seqlen_k + BLOCK_N - 1) / BLOCK_N;
    float inv_sqrt_d    = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // 1. Compute K Mean
    // -------------------------------------------------
    // Strides: [Batch, Seq, Head, Dim] mapping
    int64_t stride_s_k = (int64_t)num_heads * HEAD_DIM;
    int64_t stride_b_k = (int64_t)seqlen_k * stride_s_k;
    int64_t stride_h_k = (int64_t)HEAD_DIM;

    // Grid Strategy: 
    // X: Batch * NumHeads
    // Y: Chunks of HeadDim (8 elements per chunk)
    // This dramatically increases occupancy (e.g. 64 -> 1024 blocks).
    dim3 grid_mean(batch * num_heads, (HEAD_DIM + 7) / 8); 

    compute_k_mean_kernel<Element><<<grid_mean, kNumThreads, 0, stream>>>(
        K, k_mean, seqlen_k, num_heads, HEAD_DIM,
        stride_b_k, stride_s_k, stride_h_k
    );

    // 2. Prepare Shapes & Tensors
    Shape4D shape_Q = make_shape(seqlen_q, HEAD_DIM, num_heads, batch);
    Shape4D shape_K = make_shape(seqlen_k, HEAD_DIM, num_heads, batch);

    int64_t stride_s_q = (int64_t)num_heads * HEAD_DIM;
    int64_t stride_b_q = (int64_t)seqlen_q * stride_s_q;
    auto stride_Q = make_stride(stride_s_q, Int<1>{}, Int<HEAD_DIM>{}, stride_b_q);

    // Re-use stride_K calculation logic for TMA
    auto stride_K = make_stride(stride_s_k, Int<1>{}, Int<HEAD_DIM>{}, stride_b_k);

    Tensor mQ   = make_tensor(make_gmem_ptr(Q),   make_layout(shape_Q, stride_Q));
    Tensor mK   = make_tensor(make_gmem_ptr(K),   make_layout(shape_K, stride_K));
    Tensor mQ_q = make_tensor(make_gmem_ptr(Q_q), make_layout(shape_Q, stride_Q));
    Tensor mK_q = make_tensor(make_gmem_ptr(K_q), make_layout(shape_K, stride_K));

    using SmemLayoutInQ  = typename SmemConfigQ::SmemLayoutIn;
    using SmemLayoutOutQ = typename SmemConfigQ::SmemLayoutOut;
    using SmemLayoutInK  = typename SmemConfigK::SmemLayoutIn;
    using SmemLayoutOutK = typename SmemConfigK::SmemLayoutOut;

    // 3. Create TMA Objects
    auto tma_load_Q = make_tma_copy<Element>(
        SM90_TMA_LOAD{}, mQ, SmemLayoutInQ{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, _1{});
    auto tma_store_Q = make_tma_copy<int8_t>(
        SM90_TMA_STORE{}, mQ_q, SmemLayoutOutQ{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}, _1{});

    auto tma_load_K = make_tma_copy<Element>(
        SM90_TMA_LOAD{}, mK, SmemLayoutInK{}, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, _1{});
    auto tma_store_K = make_tma_copy<int8_t>(
        SM90_TMA_STORE{}, mK_q, SmemLayoutOutK{}, Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{}, _1{});

    // 4. Launch Q Kernel
    using SharedStorageQ = TmaSharedStorage<HEAD_DIM, Element, SmemLayoutInQ, SmemLayoutOutQ>;
    int smem_size_q = sizeof(SharedStorageQ);

    if (smem_size_q >= 48 * 1024) {
        cudaFuncSetAttribute(quantize_q_kernel<HEAD_DIM, BLOCK_M, Element, decltype(tma_load_Q), decltype(tma_store_Q)>, 
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_q);
    }

    dim3 grid_q(num_seq_tiles_q, num_heads, batch);
    quantize_q_kernel<HEAD_DIM, BLOCK_M, Element>
        <<<grid_q, kNumThreads, smem_size_q, stream>>>(
        tma_load_Q, tma_store_Q,
        seqlen_q, num_heads, batch,
        q_scales, num_seq_tiles_q, inv_sqrt_d
    );

    // 5. Launch K Kernel
    using SharedStorageK = TmaSharedStorage<HEAD_DIM, Element, SmemLayoutInK, SmemLayoutOutK>;
    int smem_size_k = sizeof(SharedStorageK);

    if (smem_size_k >= 48 * 1024) {
        cudaFuncSetAttribute(quantize_k_kernel<HEAD_DIM, BLOCK_N, Element, decltype(tma_load_K), decltype(tma_store_K)>, 
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k);
    }

    dim3 grid_k(num_seq_tiles_k, num_heads, batch);
    quantize_k_kernel<HEAD_DIM, BLOCK_N, Element>
        <<<grid_k, kNumThreads, smem_size_k, stream>>>(
        tma_load_K, tma_store_K,
        seqlen_k, num_heads, batch,
        shape_K, k_mean, k_scales, num_seq_tiles_k
    );
}

// Runtime Dispatcher
template <typename Element>
void launch_quantize_qk_runtime(
    const Element* Q, const Element* K,
    int8_t* Q_q, int8_t* K_q,
    float* q_scales, float* k_scales, float* k_mean,
    int batch, int seqlen_q, int seqlen_k, int num_heads,
    int head_dim, int block_m, int block_n,
    cudaStream_t stream)
{
    if (head_dim == 64 && block_m == 128 && block_n == 224) {
        launch_quantize_qk_config<64, 128, 224, Element>(
            Q, K, Q_q, K_q, q_scales, k_scales, k_mean,
            batch, seqlen_q, seqlen_k, num_heads, stream);
    } else if (head_dim == 96 && block_m == 128 && block_n == 208) {
        launch_quantize_qk_config<96, 128, 208, Element>(
            Q, K, Q_q, K_q, q_scales, k_scales, k_mean,
            batch, seqlen_q, seqlen_k, num_heads, stream);
    } else if (head_dim == 128 && block_m == 128 && block_n == 128) {
        launch_quantize_qk_config<128, 128, 128, Element>(
            Q, K, Q_q, K_q, q_scales, k_scales, k_mean,
            batch, seqlen_q, seqlen_k, num_heads, stream);
    } else if (head_dim == 128 && block_m == 128 && block_n == 176) {
        launch_quantize_qk_config<128, 128, 176, Element>(
            Q, K, Q_q, K_q, q_scales, k_scales, k_mean,
            batch, seqlen_q, seqlen_k, num_heads, stream);
    } else {
        assert(false && "Unsupported config");
    }
}

template void launch_quantize_qk_runtime<cutlass::half_t>(
    const cutlass::half_t*, const cutlass::half_t*,
    int8_t*, int8_t*,
    float*, float*, float*,
    int, int, int, int,
    int, int, int,
    cudaStream_t);

template void launch_quantize_qk_runtime<cutlass::bfloat16_t>(
    const cutlass::bfloat16_t*, const cutlass::bfloat16_t*,
    int8_t*, int8_t*,
    float*, float*, float*,
    int, int, int, int,
    int, int, int,
    cudaStream_t);