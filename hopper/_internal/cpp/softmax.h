/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"
#include "skip_list.h"

namespace flash
{

    using namespace cute;

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool const zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
    {
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni++)
        {
#pragma unroll
            for (int mi = 0; mi < size<0>(tensor); mi++)
            {
                if constexpr (zero_init){
                    summary(mi) = ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
                }else{
                    summary(mi) = op(summary(mi), tensor(mi, ni));
                }
                // summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
            }
        }
    }

    template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op)
    {
        CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
        for (int i = 0; i < size(dst); i++)
        {
            dst(i) = Allreduce<4>::run(src(i), op);
        }
    }

    template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
    __device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op)
    {
        thread_reduce_<zero_init>(tensor, summary, op);
        quad_allreduce_(summary, summary, op);
    }

    template <bool const zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &max)
    {
        MaxOp<float> max_op;
        reduce_<zero_init>(tensor, max, max_op);
    }

    template <bool const zero_init = true, bool warp_reduce = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &sum)
    {
        SumOp<float> sum_op;
        thread_reduce_<zero_init>(tensor, sum, sum_op);
        if constexpr (warp_reduce)
        {
            quad_allreduce_(sum, sum, sum_op);
        }
    }

    // Apply the exp to all the elements.
    template <bool const Scale_max = true, bool const Check_inf = true, int const Max_offset = 0,
              typename Engine0, typename Layout0, typename Engine1, typename Layout1>
    __forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale)
    {
        // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
        // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
        static constexpr float max_offset = float(Max_offset); // We can only template on int, not float
        static_assert(Layout0::rank == 2, "Only support 2D Tensor");
        static_assert(Layout1::rank == 1, "Only support 1D Tensor");
        CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi)
        {
            // If max is -inf, then all elements must have been -inf (possibly due to masking).
            // We don't want (-inf - (-inf)) since that would give NaN.
            // const float max_scaled = Check_inf
            //                              ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            //                              : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
            if constexpr (Check_inf){
                const float max_scaled = max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
    #pragma unroll
                for (int ni = 0; ni < size<1>(tensor); ++ni)
                {
                    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    // max * log_2(e)). This allows the compiler to use the ffma
                    // instruction instead of fadd and fmul separately.
                    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
                }
            }else{
                const float max_scaled = (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
    #pragma unroll
                for (int ni = 0; ni < size<1>(tensor); ++ni)
                {
                    // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    // max * log_2(e)). This allows the compiler to use the ffma
                    // instruction instead of fadd and fmul separately.
                    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int kNRows, int Max_offset = 0>
    struct Softmax
    {

        using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
        TensorT row_max, row_sum;
        float const softmax_scale_log2; // (log2(e) * 1/sqrt(128)) * q_dequant * k_dequant
        // int const warp_idx_in_warpgroup = (threadIdx.x / 32) % 4;
        int const warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        bool const is_warp_leader = (threadIdx.x % 32) == 0;
        // bool const is_warp_leader = cute::elect_one_sync();
        // int const row_mask;
        // int const local_row_idx;
        int const seqlen_q;
        int const thread_idx;

        // CUTLASS_DEVICE Softmax(float const softmax_scale_log2_, int const row_mask_, int const local_row_idx_) : softmax_scale_log2(softmax_scale_log2_), row_mask(row_mask_), local_row_idx(local_row_idx_) {};
        CUTLASS_DEVICE Softmax(float const softmax_scale_log2_, int const seqlen_q_, int const thread_idx_) 
            : softmax_scale_log2(softmax_scale_log2_), seqlen_q(seqlen_q_), thread_idx(thread_idx_) {};

        template <int kBlockM, typename TiledMma, bool const Is_first, bool const Check_inf = false, typename Tensor0>
        __forceinline__ __device__ TensorT max_get_scale_detect_qk_skip(
            Tensor0 &acc_s,
            const float thr,
            auto &skip_reader,
            const int m_block)
        {
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            TensorT scores_scale;
            if constexpr (Is_first)
            {
                flash::template reduce_max</*zero_init=*/true>(scores, row_max);
                cute::fill(scores_scale, 1.f);
                if (is_warp_leader)
                {
                    skip_reader.update_skip(false, warp_idx_in_warpgroup);
                }
            }
            else
            {
                Tensor scores_max_prev = make_fragment_like(row_max);
                cute::copy(row_max, scores_max_prev);

                // find the local max for each row
                Tensor scores_max_local = make_fragment_like(row_max);
                /*
                inside reduce_max, each thread reduces over the columns he holds.
                each thread holds a querter of the columns, for example:
                for headdim == 128, we have 176 columns so each thread holds 176 / 4 = 44 columns.
                each 4 consecutive threads hold TOGTHER the full row.
                after the thread level reduction, we reduce across each 4 consecutive threads and get the local max for the row.
                */
                flash::template reduce_max</*zero_init=*/true>(scores, scores_max_local);

                // update row max
                // thread_reduce_<true>(scores_max_local, row_max, MaxOp<float>());
                // flash::template reduce_max</*zero_init=*/false>(scores, row_max);

                // Compute row bounds following the same pattern as mask.h
                // Create identity tensor and partition it to get row coordinates
                auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
                auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
                Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockM>>{});  // Dummy shape, only need row info
                Tensor tScS = thread_mma.partition_C(cS);
                Tensor t0ScS = thread0_mma.partition_C(cS);
                Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol(tScS.layout()));
                Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol(t0ScS.layout()));
                
                // Compute thread_row_offset and seqlenq_row_limit following mask.h pattern
                int const thread_row_offset = get<0>(tScS_rowcol(_0{}, _0{}));
                int const seqlenq_row_limit = seqlen_q - m_block * kBlockM - thread_row_offset;
                
                bool do_qk = false;
#pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi)
                {
                    // Check if this row is out of bounds, following mask.h pattern
                    // Use t0ScS_rowcol to get compile-time known row indices
                    const bool row_not_out_of_bounds = !(int(get<0>(t0ScS_rowcol(mi, _0{}))) >= seqlenq_row_limit);
                    
                    // update row max
                    row_max(mi) = max(row_max(mi), scores_max_local(mi));
                    // float cur = !Check_inf
                    //                 ? row_max(mi)
                    //                 : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                    float cur;
                    if constexpr (Check_inf){
                        cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                    }else{
                        cur = row_max(mi);
                    }
                    float prev = scores_max_prev(mi);
                    scores_scale(mi) = exp2f((prev - cur) * softmax_scale_log2);
                    row_sum(mi) *= scores_scale(mi);

                    // do_qk |= (((scores_max_local(mi) - prev) * softmax_scale_log2) > thr) & row_not_out_of_bounds;

                    // do_qk |= ((scores_max_local(mi) * thr) >= prev) & row_not_out_of_bounds;

                    // bool cond1 = (scores_max_local(mi) - prev + abs(scores_max_local(mi)) * 0.5f >= 0); // if the current max is at least 1.5 times the previous max
                    bool cond1 = true;
                    bool cond2 = ((scores_max_local(mi) - prev) * softmax_scale_log2) > thr; // if the current max is more than thr times the previous max
                    do_qk |= cond1 & cond2 & row_not_out_of_bounds; // if both conditions are true and the row is not out of bounds, then set do_qk to true
                }

                // (warp = 32) * 4 = warpgroup, 2 * warpgroup
                const bool skip = !__any_sync(0xffffffffu, do_qk);
                if (is_warp_leader)
                {
                    skip_reader.update_skip(skip, warp_idx_in_warpgroup);
                }
            }

            return scores_scale;
        };

        // TONY: acc_s is Q times K for one tile
        template <bool const Is_first, bool const Check_inf = false, typename Tensor0>
        __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s)
        { // pass in a bool ref
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            TensorT scores_scale;
            if constexpr (Is_first)
            {
                flash::template reduce_max</*zero_init=*/true>(scores, row_max);
                cute::fill(scores_scale, 1.f);
            }
            else
            {
                Tensor scores_max_prev = make_fragment_like(row_max);
                cute::copy(row_max, scores_max_prev);
                flash::template reduce_max</*zero_init=*/false>(scores, row_max);
#pragma unroll
                for (int mi = 0; mi < size(row_max); ++mi)
                {
                    // float scores_max_cur = !Check_inf
                    //                            ? row_max(mi)
                    //                            : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));

                    float scores_max_cur;
                    if constexpr (Check_inf){
                        scores_max_cur = row_max(mi) == -INFINITY ? 0.0f : row_max(mi);
                    }else{
                        scores_max_cur = row_max(mi);
                    }

                    scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                    row_sum(mi) *= scores_scale(mi);
                }
            }
            return scores_scale;
        };

        template <bool const Is_first, bool const Check_inf = false, typename Tensor0>
        __forceinline__ __device__ void online_softmax(Tensor0 &acc_s)
        {
            // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
            static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
            flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
        };

        __forceinline__ __device__ TensorT finalize(float const final_scale = 1.f)
        {
            SumOp<float> sum_op;
            quad_allreduce_(row_sum, row_sum, sum_op);
            TensorT scores_scale;
#pragma unroll
            for (int mi = 0; mi < size(row_sum); ++mi)
            {
                float sum = row_sum(mi);
                // float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
                float inv_sum = (sum == 0.f | sum != sum) ? 0.f : 1.f / sum;
                scores_scale(mi) = inv_sum * final_scale;
                // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
                if constexpr (Max_offset != 0)
                {
                    static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                    sum *= sum_scale;
                }
                row_sum(mi) = ((sum == 0.f) | (sum != sum)) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
            }
            return scores_scale;
        };

        template <typename Tensor1>
        __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale)
        {
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
#pragma unroll
            for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
            {
#pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
                {
                    acc_o_rowcol(mi, ni) *= scores_scale(mi);
                }
            }
        };
    };

} // namespace flash
