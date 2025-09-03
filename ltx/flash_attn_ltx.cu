#pragma once

#include <torch/types.h>
#include <torch/torch.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "flash_attn_ltx.cuh"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

/*
this function runs as if we used flash attention 3 with the following template values:
void device_kernel<
    enable_sm90_or_later<
        FlashAttnFwdSm90<
            CollectiveMainloopFwdSm90<
                2, // Stages
                tuple<C<1>, C<1>, C<1>>, // ClusterShape_
                tuple<C<128>, C<176>, C<128>>, // TileShape_MNK_
                128, // kHeadDimV
                bfloat16_t, // Element_
                float, // ElementAccum_
                Sm90, // ArchTag_
                0, // Is_causal_
                0, // Is_local_
                0, // Has_softcap_
                0, // Varlen_
                0, // PagedKVNonTMA_
                0, // AppendKV_
                0, // HasQv_
                1, // MmaPV_is_RS
                1, // IntraWGOverlap
                0, // PackGQA_
                0, // Split_
                0 // V_colmajor_
            >,
            CollectiveEpilogueFwd<
                tuple<C<128>, C<128>, C<176>>, // TileShape_MNK_PV
                tuple<C<1>, C<1>, C<1>>, // ClusterShape
                bfloat16_t, // ElementOut
                Sm90, // ArchTag
                256, // NumMmaThreads
                0, // Varlen
                0, // PackGQA
                0, // Split
                0 // FP8_TransposeV
            >,
            StaticPersistentTileScheduler<
                0 // Split
            >
        >
    >
>(Params)
*/
void flash_attn_ltx(
    torch::Tensor q, // (batch_size, seqlen_q, nheads, headdim)
    torch::Tensor k,
    torch::Tensor v,
    // consider: working with skip lists (idx and count)
    torch::Tensor skip_mask // (batch_size, nheads, CEIL_DIV(T_c, 64))
){
    // sequence length
    const int N = q.size(1);

    // height of each tile of q
    const int B_r = 128; // todo: find the correct value
    // number of q tiles
    const int T_r = CEIL_DIV(N, B_r);

    // height of each tile of k and v
    const int B_c = 128; // todo: find the correct value
    // number of k and v tiles
    const int T_c = CEIL_DIV(N, B_c);

    // number of heads
    const int nheads = q.size(2); // should be 32 in ltx
    const int batch_size = q.size(0);

    // head dimension
    const int D = q.size(3); // should be 128
    
    /* Flash Attention performs two main matrix multiplications per block:
     
     1. ATTENTION SCORES: Q @ K^T -> S
        TileShape_MNK = (128, 176, 128) represents:
        - M=128: Number of query tokens processed per thread block  
        - N=176: Number of key tokens processed per thread block
        - K=128: Head dimension for Q and K matrices
        Matrix sizes: Q_tile[128, 128] @ K_tile^T[176, 128] = S_tile[128, 176]
    
     2. OUTPUT COMPUTATION: P @ V -> O  (where P = softmax(S))
        TileShape_MNK_PV = (128, 128, 176) represents:
        - M=128: Number of query tokens (same as above)
        - N=128: Head dimension for V and output O matrices  
        - K=176: Number of value tokens (same as key tokens)
        Matrix sizes: P_tile[128, 176] @ V_tile[176, 128] = O_tile[128, 128]
    
     The kBlockN=176 value is determined by tile_size_fwd_sm90() in hopper/tile_size.h:33
     For headdim=128, non-causal, non-local attention: returns {128, use_blockN_128 ? 128 : 176, true, true}
     176 is chosen for optimal performance balancing:
     - Shared memory constraints (hopper/tile_size.h:35 comment: "128 x 144 hits the limit if !MmaPV_is_RS")
     - Tile quantization efficiency (hopper/tile_size.h:28 comment: "Good for long seqlen but suffers from tile quantization at short seqlen")
     - Memory throughput optimization for SM90+ architecture
     This value was empirically optimized for general attention workloads without causal/local masking
    */
    using TileShape_MNK = cute::Shape<cute::Int<128>, cute::Int<176>, cute::Int<128>>; // QK^T computation: (M,N,K) = (seqlen_q, seqlen_k, head_dim)
    using TileShape_MNK_PV = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<176>>;  // PV computation: (M,N,K) = (seqlen_q, head_dim_v, seqlen_k)

    // in original implementation, the grid size is the number of SMs (because we use presistent kernel)
    dim3 grid(T_r, nheads, batch_size); // x, y, z
    /*
    in the original implementation, the block size is 1D and ditermined by the line:
        uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
        dim3(MaxThreadsPerBlock, 1, 1);
    which translates to:
        uint32_t MaxThreadsPerBlock = 256 + (1 * 128) = 384;
        dim3(384, 1, 1);
    meaning, 1 warp group for loading, 2 warp groups for mma
    */
    // 3 warp groups, 1 consumer, 2 producer -> (32 * 4) * 3 = 384 threads per block
    dim3 block(128, 3); // x - number of threads per warp group, y - number of warp groups


    // struct SharedStorage {
    //     struct TensorStorage : cute::aligned_struct<128, _1> {
    //         union {
    //             struct {
    //                 // DOR: why do we even need smem_o? since o's temp results are already stored in registers?

    //                 // Padding to align epilogue smem_o with mainloop smem_v start
    //                 // mainloop_smem_padding = max(0, sizeof(epilogue) - sizeof(mainloop.smem_v))
    //                 // Since epilogue (~32KB) < mainloop.smem_v (~180KB), padding = 0
    //                 cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    
    //                 // Mainloop tensor storage: Since MmaPV_is_RS=1, uses TensorStorageWithoutPNoTranspose
    //                 // Contains: smem_v (~180KB), smem_q (~32KB), smem_k (~180KB), smem_qv (0 bytes)
    //                 // Total mainloop size: ~392KB
    //                 typename CollectiveMainloop::TensorStorage mainloop;
    //             };
                
    //             // Epilogue tensor storage: smem_o for output (~32KB)
    //             // Union allows epilogue to reuse mainloop memory space after computation
    //             // We want smem_o to line up with the start of smem_v
    //             typename CollectiveEpilogue::TensorStorage epilogue;
    //         };
    //         // Effective tensor storage size: max(mainloop, epilogue) = ~392KB
    //     } tensors;
        
    //     struct PipelineStorage : cute::aligned_struct<16, _1> {
    //         // Query barrier for Q matrix synchronization (8 bytes)
    //         alignas(16) BarrierQ barrier_Q;
            
    //         // Commented out barriers not needed due to template parameters:
    //         // alignas(16) BarrierQ barrier_Qv; // not needed since HasQv=0
    //         // alignas(16) cutlass::arch::ClusterBarrier barrier_O; // not needed since ClusterShape=(1,1,1)
            
    //         // Pipeline storage for each data stream (2 stages each):
    //         // Each PipelineTmaAsync<2> contains: 2×ClusterTransactionBarrier + 2×ClusterBarrier = 32 bytes
    //         alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;    // 32 bytes
    //         alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;    // 32 bytes
    //         alignas(16) typename CollectiveMainloop::MainloopPipelineVt::SharedStorage pipeline_vt;  // 32 bytes
    //         alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_k_new; // 32 bytes
    //         alignas(16) typename CollectiveMainloop::MainloopPipelineKVNew::SharedStorage pipeline_v_new; // 32 bytes
            
    //         // Tile scheduler storage for persistent kernel work distribution (~16-32 bytes)
    //         alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
    //     } pipelines; 
    //     // Total pipeline storage: ~200-220 bytes
        
    //     // Overall SharedStorage size: ~392KB (tensor) + ~220 bytes (pipeline) + alignment = ~392KB total
    //     // This approaches the GPU's shared memory limit, affecting occupancy but optimized for Flash Attention's memory access patterns
    // };
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    int smem_size = AttnKernel::SharedStorageSize;

    flash_attn_ltx_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<uint64_t*>(skip_mask.data_ptr())
    );
}