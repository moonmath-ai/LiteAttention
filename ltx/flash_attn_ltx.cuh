#include <cuda_bf16.h>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

__global__ 
 // consider: setting it to lower value (the number of registers per thread changes between producer and consumer)
 // or maybe somehow tell the compiler that there's only one block per SM
__launch_bounds__(128 * 3)
template <typename TileShape_MNK_PV>
void flash_attn_ltx_kernel(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    // uint64_t* skip_mask
    __nv_bfloat16* output
){
    // consider: creating k anv v already in thair tailed version in global memory (requires only to reorder the weight rows in the linear layer who produces them)
    // (the rotational embedding and rms norm could complicate things though)
    // we keep this to remaind ourself that the amount of needed register derived from the number of mma warp groups
    static constexpr uint32_t NumMmaWarpGroups = 2;
    // consider: fine tunning these values according to the changes in this fa3 version
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 240 : 160;

    // the first warp group is the producer
    if (threadIdx.y == 0){ // producer
        cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    }
    // the other 2 warp groups are the consumers
    else{ // consumer
        cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

        // todo: minimize the use of this rage inducing and arrogant cutlass patterns
        static constexpr int kBlockM = get<0>(TileShape_MNK{}); // number of rows in a tile of Q (128)
        // static constexpr int kBlockN = get<1>(TileShape_MNK{});
        // static constexpr int kHeadDim = get<2>(TileShape_MNK{});
        using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
        using AtomLayoutPV = AtomLayoutQK;
        // define the gmma operation needed for doing P_tile @ V_tile
        using TiledMmaPV = decltype(
            cute::make_tiled_mma(
                decltype(cute::GMMA::rs_op_selector<__nv_bfloat16, __nv_bfloat16, float, TileShape_MNK_PV, GMMA::Major::K, GMMA::Major::MN>()){},
                AtomLayoutPV{}
            )
        );

        // Initialize matmul objects, here we actually define the registers needed for the operation
        TiledMmaPV tiled_mma_pv;

        /*
        taken from the answer here: https://youtu.be/JwUcZwPOCpA?t=3152
        DOR: thread wise accumulator for the attention output
        O - signifay this is the O matrix
        r - signifay this is stored in registers
        t - signifay this is the thread wise view
        */
        // Attention output (GEMM-II) accumulator.
        Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
    }
}