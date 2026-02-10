#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/gemm/collective/builders/sm90_common.inl"

using namespace cute;

constexpr static int kStages = 2;

constexpr static int kBlockM = 128;
constexpr static int kBlockN = 176;
constexpr static int kHeadDim = 128;
using Element = int8_t;
using ElementV = cutlass::bfloat16_t;
constexpr static auto TmaMajorV = GMMA::Major::MN;

using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>; // QK^T computation: (M,N,K) = (seqlen_q, seqlen_k, head_dim)
using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>; // PV computation: (M,N,K) = (seqlen_q, head_dim_v, seqlen_k)
/*
(M x K) x (K x N) = (M x N) (QK)
(M x N) x (N x K) = (M x K) (PV)
*/

// Sw<3,4,3> o _0 o (_8,_128):(_128,_1)
using SmemLayoutAtomQ = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, // major
        Element, // data type
        decltype(cute::get<0>(TileShape_MNK{})), // M dimension
        decltype(cute::get<2>(TileShape_MNK{})) // K dimension
    >() // smem layout
); // should be a layout with some dim of size 128

// Sw<3,4,3> o _0 o ((_8,_16),(_128,_1)):((_128,_1024),(_1,_0))
using SmemLayoutQ = decltype(
    tile_to_shape(
        SmemLayoutAtomQ{},
        select<0, 2>(TileShape_MNK{}) // Layout of shape (M, K)
    )
);

using SmemLayoutAtomK = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, // major
        Element, // data type
        decltype(cute::get<1>(TileShape_MNK{})), // N dimension
        decltype(cute::get<2>(TileShape_MNK{})) // K dimension
    >() // smem layout
);


// Sw<3,4,3> o smem_ptr[8b](unset) o ((_8,_22),(_128,_1),(_1,_2)):((_128,_1024),(_1,_0),(_0,_22528))
// Shape structure: ((_8,_22),(_128,_1),(_1,_2))
//   - N dimension: 8 x 22 = 176 (kBlockN)
//   - K dimension: 128 x 1 = 128 (kHeadDim)
//   - Stages: 1 x 2 = 2 (kStages)
//
// ASCII illustration of 3D structure:
//
//         K dimension (128)
//         <----------->
//     ┌─────────────────┐
//     │                 │
//   N │                 │  Stage 0
//   ( │   [176 x 128]   │
//   1 │                 │
//   7 │                 │
//   6 │                 │
//   ) └─────────────────┘
//     ┌─────────────────┐
//     │                 │
//   N │                 │  Stage 1
//   ( │   [176 x 128]   │
//   1 │                 │
//   7 │                 │
//   6 │                 │
//   ) └─────────────────┘
//
//   N dimension breakdown: 8 tiles × 22 elements = 176
//   K dimension breakdown: 128 tiles × 1 element = 128
//

using SmemLayoutK = decltype(
    tile_to_shape(
        // Sw<3,4,3> o _0 o (_8,_128):(_128,_1)
        SmemLayoutAtomK{},
        make_shape(
            shape<1>(TileShape_MNK{}), // N dimension
            shape<2>(TileShape_MNK{}), // K dimension
            Int<kStages>{} // stages (2)
        )
    ) // Layout ((),(),()):((),(),())
);

// Sw<3,3,3> o _0 o (_64,_8):(_1,_64)
using SmemLayoutAtomVt = decltype(
    cutlass::gemm::collective::detail::ss_smem_selector<
        TmaMajorV, // major
        ElementV, // data type
        Int<kHeadDim>, // head_dim_v dimension
        decltype(cute::get<2>(TileShape_MNK_PV{})) // N dimension (seqlen_k)
    >() // smem layout
); // should be a layout with some dim of size 128

// Sw<3,4,3> o smem_ptr[16b](unset) o ((_64,_2),(_8,_22),(_1,_2)):((_1,_11264),(_64,_512),(_0,_22528))
// Shape structure: ((_64,_2),(_8,_22),(_1,_2))
//   - head_dim_v dimension: 64 x 2 = 128 (kHeadDim)
//   - N dimension: 8 x 22 = 176 (kBlockN, seqlen_k)
//   - Stages: 1 x 2 = 2 (kStages)
//
// ASCII illustration of 3D structure:
//
//         N dimension (176)
//         <----------->
//     ┌─────────────────┐
//     │                 │
//   K │                 │  Stage 0
//   ( │   [128 x 176]   │
//   1 │                 │
//   2 │                 │
//   8 │                 │
//   ) └─────────────────┘
//     ┌─────────────────┐
//     │                 │
//   K │                 │  Stage 1
//   ( │   [128 x 176]   │
//   1 │                 │
//   2 │                 │
//   8 │                 │
//   ) └─────────────────┘
//
//   head_dim_v dimension breakdown: 64 tiles × 2 elements = 128
//   N dimension breakdown: 8 tiles × 22 elements = 176
//
using SmemLayoutVt = decltype(
    tile_to_shape(
        SmemLayoutAtomVt{},
        make_shape(
            Int<kHeadDim>{}, // head_dim_v dimension 128
            shape<2>(TileShape_MNK_PV{}), // N dimension (seqlen_k) 176
            Int<kStages>{} // stages (2)
        ),
        std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}
    ) // Layout with stages for double buffering
);


using GmemTiledCopyQ = cute::SM90_TMA_LOAD;

using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen, d, head, batch)
using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
using GmemTensorQ = decltype(make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)), ShapeQKV{}, StrideQK{}));

using TMA_Q = decltype(
    make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        GmemTensorQ{},
        SmemLayoutQ{},
        TileShape_MNK{},
        Int<1>{}
    )
);

using TMA_K = decltype(
    make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)), ShapeQKV{}, StrideQK{}),
        take<0, 2>(SmemLayoutK{}),
        TileShape_MNK{},
        Int<1>{}
    )
); // mcast along M mode for this N load, if any

using StrideV = std::conditional_t<!V_colmajor, StrideQK, cute::Stride<_1, int64_t, int64_t, int64_t>>;

using TMA_V = decltype(
    make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<ElementV const *>(nullptr)), ShapeQKV{}, select<1, 0, 2, 3>(StrideV{})),
        take<0, 2>(SmemLayoutVt{}),
        select<1, 2>(TileShape_MNK_PV{}),
        Int<1>{}
    )
); // mcast along M mode for this N load, if any

int main() {

    printf("SmemLayoutAtomQ:\n");
    print_layout(SmemLayoutAtomQ{});

    printf("SmemLayoutQ:\n");
    print_layout(SmemLayoutQ{});

    printf("SmemLayoutAtomK:\n");
    print_layout(SmemLayoutAtomK{});

    printf("SmemLayoutK:\n");
    // print_layout(SmemLayoutK{});
    print(SmemLayoutK{}); printf("\n");

    printf("SmemLayoutAtomVt:\n");
    print_layout(SmemLayoutAtomVt{});

    printf("SmemLayoutVt:\n");
    // print_layout(SmemLayoutVt{});
    print(SmemLayoutVt{}); printf("\n");

    return 0;
}
/*
nvcc -std=c++20 -O3 --use_fast_math -I./csrc/cutlass/include -arch=sm_90 -o int8_bf16_TMA_MMA int8_bf16_TMA_MMA.cu
*/