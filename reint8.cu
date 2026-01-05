#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/gemm/collective/builders/sm90_common.inl"

using namespace cute;

static constexpr bool ReInt8 = true;

static constexpr int kStages = 2;

static constexpr int kBlockM = 256;
static constexpr int kBlockMI = 128;
static constexpr int kBlockN = 128;
static constexpr int kHeadDim = 128;
static constexpr int InnerDimSize = 2;

using Element = int8_t;
using ElementAccumQK = int32_t;
using ElementAccum = float;
using ElementV = cutlass::bfloat16_t;
static constexpr cute::GMMA::Major MmaMajorV = GMMA::Major::MN;

using TileShape_MNK = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MINK = Shape<Int<kBlockMI>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MNK_PV = std::conditional_t<ReInt8,
    Shape<decltype(get<0>(TileShape_MINK{})), Int<kHeadDim>, decltype(get<1>(TileShape_MINK{}))>,
    Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDim>, decltype(get<1>(TileShape_MNK{}))>>;

using AtomLayoutQK = Layout<std::conditional_t<ReInt8,
    Shape<Int<kBlockMI / 64>, Int<InnerDimSize>, _1>,
    Shape<Int<kBlockM / 64>, Int<InnerDimSize>, _1>>>; // (num mma wg, inner dim size, 1)

using TiledMmaQK = decltype(cute::make_tiled_mma(
    std::conditional_t<ReInt8,
        decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MINK>()),
        decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MNK>())>{},
    AtomLayoutQK{})
);

using AtomLayoutPV = AtomLayoutQK;
using TiledMmaPV = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<ElementV, ElementV, ElementAccum,
                               TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(),
    AtomLayoutPV{}));

static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
static constexpr int NumMmaThreads = size(TiledMmaPV{});

using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 GMMA::Major::K,
                                 Element,
                                 decltype(cute::get<0>(TileShape_MINK{})),
                                 decltype(cute::get<2>(TileShape_MINK{}))>());

// using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
using SmemLayoutQ = decltype(tile_to_shape(
    SmemLayoutAtomQ{},
    make_shape(
        shape<0>(TileShape_MINK{}),
        shape<2>(TileShape_MINK{}),
        Int<InnerDimSize>{})));

using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 GMMA::Major::K,
                                 Element,
                                 decltype(cute::get<1>(TileShape_MINK{})),
                                 decltype(cute::get<2>(TileShape_MINK{}))>());

using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomK{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                     MmaMajorV, ElementV,
                                     Int<kHeadDim>,
                                     decltype(cute::get<2>(TileShape_MNK_PV{}))>());
using SmemLayoutVtMma = decltype(tile_to_shape(
    SmemLayoutAtomVtMma{},
    make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
    std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 GMMA::Major::K,
                                 ElementV,
                                 decltype(cute::get<0>(TileShape_MINK{})),
                                 decltype(cute::get<1>(TileShape_MINK{}))>());

using SmemLayoutP = decltype(tile_to_shape(
    SmemLayoutAtomP{},
    make_shape(shape<0>(TileShape_MINK{}), shape<1>(TileShape_MINK{}), Int<InnerDimSize>{})));

int main()
{

    printf("AtomLayoutQK:\n");
    // print_layout(AtomLayoutQK{});  // 3D layout, use print() instead
    print(AtomLayoutQK{});
    printf("\n");

    printf("TiledMmaQK:\n");
    // print_layout(TiledMmaQK{});  // Complex type, use print() instead
    print(TiledMmaQK{});
    printf("\n");

    printf("AtomLayoutPV:\n");
    // print_layout(AtomLayoutPV{});  // 3D layout, use print() instead
    print(AtomLayoutPV{});
    printf("\n");

    printf("TiledMmaPV:\n");
    // print_layout(TiledMmaPV{});  // Complex type, use print() instead
    print(TiledMmaPV{});
    printf("\n");

    printf("SmemLayoutAtomQ:\n");
    print_layout(SmemLayoutAtomQ{});

    printf("SmemLayoutQ:\n");
    // print_layout(SmemLayoutQ{});  // 3D layout, use print() instead
    print(SmemLayoutQ{});
    printf("\n");

    printf("SmemLayoutAtomK:\n");
    print_layout(SmemLayoutAtomK{});

    printf("SmemLayoutK:\n");
    // print_layout(SmemLayoutK{});  // 3D layout, use print() instead
    print(SmemLayoutK{});
    printf("\n");

    printf("SmemLayoutAtomVt:\n");
    print_layout(SmemLayoutAtomVtMma{});

    printf("SmemLayoutVt:\n");
    // print_layout(SmemLayoutVtMma{});  // 3D layout, use print() instead
    print(SmemLayoutVtMma{});
    printf("\n");

    printf("SmemLayoutAtomP:\n");
    print_layout(SmemLayoutAtomP{});

    printf("SmemLayoutP:\n");
    // print_layout(SmemLayoutP{});  // 3D layout, use print() instead
    print(SmemLayoutP{});
    printf("\n");

    // ~~~~~~~~~~~~~~~~~~~~~~~~~ inner kernel code ~~~~~~~~~~~~~~~~~~~~~~~~~

    static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
    printf("MmaWarpGroups: %d\n", MmaWarpGroups);
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                  make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    printf("warp_group_thread_layout:\n");
    print(warp_group_thread_layout);
    printf("\n");

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    printf("tiled_mma_qk:\n");
    print(tiled_mma_qk); printf("\n");
    printf("tiled_mma_pv:\n");
    print(tiled_mma_pv); printf("\n");

    // (thread_idx, value ) -> index in some op or memory
    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(0));
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(0));
    printf("wg_mma_qk:\n");
    print(wg_mma_qk); printf("\n");
    printf("wg_mma_pv:\n");
    print(wg_mma_pv); printf("\n");
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return 0;
}

/*
nvcc -std=c++20 -O3 --use_fast_math -I./csrc/cutlass/include -arch=sm_90 -o reint8 reint8.cu && \
./reint8 > shapes_and_such.txt 2>&1
*/