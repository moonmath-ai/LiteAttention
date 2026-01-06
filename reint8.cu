#include "cute/tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/gemm/collective/builders/sm90_common.inl"

using namespace cute;

#ifndef REINT8_ENABLED
static constexpr bool ReInt8 = false;
#else
static constexpr bool ReInt8 = true;
#endif

static constexpr int kStages = 2;

static constexpr int kBlockM = ReInt8 ? 256 : 128;
static constexpr int kBlockMI = 128;
static constexpr int kBlockN = 128;
static constexpr int kHeadDim = 128;
static constexpr int InnerDimSize = 2;

static constexpr bool MmaPV_is_RS = false;
using Element = int8_t;
using ElementAccumQK = int32_t;
using ElementAccum = float;
using ElementV = cutlass::bfloat16_t;
static constexpr cute::GMMA::Major MmaMajorV = GMMA::Major::MN;
static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::K;

using TileShape_MNK = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MINK = Shape<Int<kBlockMI>, Int<kHeadDim>, Int<kBlockN>>;
using TileShape_MNK_PV = std::conditional_t<ReInt8,
                                            Shape<decltype(get<0>(TileShape_MINK{})), Int<kHeadDim>, decltype(get<1>(TileShape_MINK{}))>,
                                            Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDim>, decltype(get<1>(TileShape_MNK{}))>>;

using AtomLayoutQK = Layout<std::conditional_t<ReInt8,
                                            //    Shape<Int<kBlockMI / 64>, Int<InnerDimSize>, _1>,
                                               Shape<Int<kBlockMI / 64>, _1, _1>,
                                               Shape<Int<kBlockM / 64>, _1, _1>>>; // (num mma wg, inner dim size, 1)

using TiledMmaQK = decltype(cute::make_tiled_mma(
    std::conditional_t<ReInt8,
                       decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MINK>()),
                       decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccumQK, TileShape_MNK>())>{},
    AtomLayoutQK{}));

using AtomLayoutPV = AtomLayoutQK;
using TiledMmaPV = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<ElementV, ElementV, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(),
    AtomLayoutPV{}));

// static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
// static constexpr int NumMmaThreads = size(TiledMmaPV{});

using SmemLayoutAtomQ = decltype(
    std::conditional_t<ReInt8,
                       decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MINK{})), decltype(cute::get<2>(TileShape_MINK{}))>()),
                       decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>())>{});

// using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
using SmemLayoutQ = std::conditional_t<ReInt8,
                                       decltype(tile_to_shape(SmemLayoutAtomQ{}, make_shape(shape<0>(TileShape_MINK{}), shape<2>(TileShape_MINK{}), Int<InnerDimSize>{}))),
                                       decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})))>;

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

using SmemLayoutAtomP = std::conditional_t<ReInt8,
                                           decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementV, decltype(cute::get<0>(TileShape_MINK{})), decltype(cute::get<1>(TileShape_MINK{}))>()),
                                           decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementV, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>())>;

using SmemLayoutP = std::conditional_t<ReInt8,
                                       decltype(tile_to_shape(SmemLayoutAtomP{}, make_shape(shape<0>(TileShape_MINK{}), shape<1>(TileShape_MINK{}), Int<InnerDimSize>{}))),
                                       decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})))>;

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

__global__ void kernel_inspect_layouts()
{
    // Use extern shared memory pattern: declare as char array and cast to our type
    extern __shared__ char shared_memory[];
    TensorStorage &shared_storage = *reinterpret_cast<TensorStorage *>(shared_memory);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});     // (kBlockM, kHeadSize, InnerDimSize[optional])
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});     // (kBlockN, kHeadSize, kStages)
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVtMma{}); // (kHeadSize, kBlockN, kStages)
    Tensor sP = [&]
    {
        if constexpr (MmaPV_is_RS)
        {
            // We might not have smem_p if !MmaPV_is_RS, just use smem_q as a placeholder since we don't use it
            // Cast to ElementV* since SmemLayoutP expects ElementV type (this placeholder is never actually used)
            return make_tensor(make_smem_ptr(reinterpret_cast<ElementV *>(shared_storage.smem_q.data())), SmemLayoutP{});
        }
        else
        {
            return make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
        }
    }();

    static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
    printf("MmaWarpGroups: %d\n", MmaWarpGroups);
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    printf("warp_group_thread_layout:\n");
    print(warp_group_thread_layout);
    printf("\n");

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    printf("tiled_mma_qk:\n");
    print(tiled_mma_qk);
    printf("\n");
    printf("tiled_mma_pv:\n");
    print(tiled_mma_pv);
    printf("\n");

    // (thread_idx, value ) -> index in some op or memory
    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(1));
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(1));
    printf("wg_mma_qk:\n");
    print(wg_mma_qk);
    printf("\n");
    printf("wg_mma_pv:\n");
    print(wg_mma_pv);
    printf("\n");


    printf("smem_tiled_copy_P:\n");
    auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
    print(smem_tiled_copy_P);
    printf("\n");

    const int thread_idx = 0;
    printf("smem_thr_copy_P:\n");
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
    print(smem_thr_copy_P);
    printf("\n");

    // Allocate "fragments/descriptors"
    printf("tSrQ (partition_fragment_A(sQ)):\n");
    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
    print(tSrQ);
    printf("\n");

    printf("tSrK (partition_fragment_B(sK)):\n");
    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
    print(tSrK);
    printf("\n");

    printf("tOrV (partition_fragment_B(sV)):\n");
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
    print(tOrV);
    printf("\n");

    printf("tOsP (partition_fragment_A(sP)):\n");
    Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
    print(tOsP);
    printf("\n");

    printf("tPsP (partition_D(as_position_independent_swizzle_tensor(sP))):\n");
    Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));
    print(tPsP);
    printf("\n");

    printf("tOrO (partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}))):\n");
    Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
    print(tOrO);
    printf("\n");
}

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

    // Launch kernel to run device code that uses smem_ptr
    // Calculate exact shared memory size needed
    constexpr size_t smem_size = sizeof(TensorStorage);

    // Set maximum dynamic shared memory for this function
    // Use the exact size needed (not the maximum), as per CUTLASS examples
    // Only set if size >= 48KB (default limit)
    if (smem_size >= (48 << 10))
    {
        cudaError_t err = cudaFuncSetAttribute(
            kernel_inspect_layouts,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size); // Use exact size needed
        if (err != cudaSuccess)
        {
            printf("ERROR: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }

    // Launch kernel with exact shared memory size needed
    kernel_inspect_layouts<<<1, 1, smem_size>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize and check for runtime errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("ERROR: cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

/*
# Compile with ReInt8 = true
nvcc -std=c++20 --use_fast_math -I./csrc/cutlass/include -arch=sm_90a -DREINT8_ENABLED -o reint8 reint8.cu && \
./reint8 > shapes_and_such.txt 2>&1

# Compile with ReInt8 = false
nvcc -std=c++20 --use_fast_math -I./csrc/cutlass/include -arch=sm_90a -o reint8 reint8.cu && \
./reint8 > shapes_and_such_no_reint8.txt 2>&1
*/