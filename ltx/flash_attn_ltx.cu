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
    const int nheads = q.size(2); // should be 32
    const int batch_size = q.size(0);

    // head dimension
    const int D = q.size(3); // should be 128

    dim3 grid(T_r, nheads, batch_size); // x, y, z
    // 2 warp groups, 1 consumer, 1 producer -> (32 * 4) * 2 = 256 threads per block
    dim3 block(128, 2); // x - number of threads per warp group, y - number of warp groups
    
    // Flash Attention performs two main matrix multiplications per block:
    // 
    // 1. ATTENTION SCORES: Q @ K^T -> S
    //    TileShape_MNK = (128, 176, 128) represents:
    //    - M=128: Number of query tokens processed per thread block  
    //    - N=176: Number of key tokens processed per thread block
    //    - K=128: Head dimension for Q and K matrices
    //    Matrix sizes: Q_tile[128, 128] @ K_tile^T[176, 128] = S_tile[128, 176]
    //
    // 2. OUTPUT COMPUTATION: P @ V -> O  (where P = softmax(S))
    //    TileShape_MNK_PV = (128, 128, 176) represents:
    //    - M=128: Number of query tokens (same as above)
    //    - N=128: Head dimension for V and output O matrices  
    //    - K=176: Number of value tokens (same as key tokens)
    //    Matrix sizes: P_tile[128, 176] @ V_tile[176, 128] = O_tile[128, 128]
    //
    // The kBlockN=176 value is determined by tile_size_fwd_sm90() in hopper/tile_size.h:33
    // For headdim=128, non-causal, non-local attention: returns {128, use_blockN_128 ? 128 : 176, true, true}
    // 176 is chosen for optimal performance balancing:
    // - Shared memory constraints (hopper/tile_size.h:35 comment: "128 x 144 hits the limit if !MmaPV_is_RS")
    // - Tile quantization efficiency (hopper/tile_size.h:28 comment: "Good for long seqlen but suffers from tile quantization at short seqlen")
    // - Memory throughput optimization for SM90+ architecture
    // This value was empirically optimized for general attention workloads without causal/local masking
    using TileShape_MNK = cute::Shape<cute::Int<128>, cute::Int<176>, cute::Int<128>>; // QK^T computation: (M,N,K) = (seqlen_q, seqlen_k, head_dim)
    using TileShape_MNK_PV = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<176>>;  // PV computation: (M,N,K) = (seqlen_q, head_dim_v, seqlen_k)


    flash_attn_ltx_kernel<<<grid, block>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<uint64_t*>(skip_mask.data_ptr())
    );
}