
#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/gemm/collective/builders/sm90_common.inl"

using namespace cute;

static constexpr int kStages = 2;

static constexpr int kBlockM = 192;
static constexpr int kBlockN = 128;
static constexpr int kHeadDim = 128;

static constexpr bool MmaPV_is_RS = false;
using Element = int8_t;
using ElementAccumQK = int32_t;
using ElementAccum = float;
using ElementV = cutlass::bfloat16_t;
static constexpr cute::GMMA::Major MmaMajorV = GMMA::Major::MN;
static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::K;

using TileShape_MNK = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;

using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>; // (num mma wg, 1, 1)

using TileShape_MNK_QK = TileShape_MNK;
using TiledMmaQK = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MNK_QK>(), AtomLayoutQK{}));

using AtomLayoutPV = AtomLayoutQK;
using TiledMmaPV = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<ElementV, ElementV, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(),
    AtomLayoutPV{}));

using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

// using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 GMMA::Major::K,
                                 Element,
                                 decltype(cute::get<1>(TileShape_MNK{})),
                                 decltype(cute::get<2>(TileShape_MNK{}))>());

using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomK{},
    make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, ElementV, Int<kHeadDim>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());

using SmemLayoutVt = decltype(tile_to_shape(
    SmemLayoutAtomVt{},
    make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
    std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

using SmemLayoutAtomVtMma = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                     MmaMajorV, ElementV,
                                     Int<kHeadDim>,
                                     decltype(cute::get<2>(TileShape_MNK_PV{}))>());
using SmemLayoutVtMma = decltype(tile_to_shape(
    SmemLayoutAtomVtMma{},
    make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
    std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementV, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));

using RmemShapeO = decltype(select<0, 1>(TileShape_MNK_PV{}));

static constexpr bool Use_TMA_Q = true;
static constexpr bool MmaQK_is_RS = false;
static constexpr bool Use_TMA_KV = true;
static constexpr bool AppendKV = false;

// If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q to be aligned
// and have sQ being position_independent_swizzle_tensor.
// If !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want smem_k and smem_v to be aligned.
static constexpr size_t SmemAlignmentQ = Use_TMA_Q && !MmaQK_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
static constexpr size_t SmemAlignmentK = Use_TMA_KV && !AppendKV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<ElementV, 0>, cute::array_aligned<ElementV, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;

struct TensorStorageWithPNoTranspose : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0>
{
    cute::array_aligned<ElementV, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    // SmemQv_t smem_qv;
    SmemP_t smem_p;
};

using TensorStorage = TensorStorageWithPNoTranspose;

using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, ElementV>;