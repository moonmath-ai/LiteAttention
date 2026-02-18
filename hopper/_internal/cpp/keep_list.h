#pragma once

#include "cute/tensor.hpp"

namespace flash
{

    using namespace cute;

    static constexpr int kNumMmaWarpGroups = 4;

    // --------------------------------------------------------------------------
    // Computes per-query-block offset into keep list buffers
    // --------------------------------------------------------------------------
    template <typename TileShape_MNK, typename ParamsType>
    __device__ uint64_t compute_keep_list_offset(
        const ParamsType& params, int bidb, int bidh, int m_block)
    {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const num_heads = get<2>(params.shape_Q);
        uint64_t const num_q_blocks = cute::ceil_div(get<0>(params.shape_Q), kBlockM);
        uint64_t const num_k_blocks = cute::ceil_div(get<0>(params.shape_K), kBlockN) + 1;
        uint32_t const q_i = static_cast<uint32_t>(m_block);
        return (static_cast<uint64_t>(bidb) * num_heads * num_q_blocks * num_k_blocks) +
               (static_cast<uint64_t>(bidh) * num_q_blocks * num_k_blocks) +
               (static_cast<uint64_t>(q_i) * num_k_blocks);
    }

    // ============================================================================
    // Unified helper struct for reading keep lists and must-do lists
    // Encapsulates all the logic for iterating through list ranges
    // Template parameters:
    //   - IsKeepList: true for KeepList, false for MustDoList
    //   - Reverse: whether to read the list in reverse
    //   - Phase: only used when IsKeepList=true, controls step direction
    // ============================================================================
    template <bool IsKeepList, bool Reverse, bool Phase = true>
    struct ListReader
    {
        const int16_t *list_ptr;
        int list_len;
        int read_idx;
        int start_idx;
        int end_idx;

        static constexpr int step = Phase ? 1 : -1;
        /*
        For KeepList reverse with phase=1 (add 1):
        [2, 30, -1] -> [2, -1, 30] -> [2, 0, 31]
        For KeepList reverse with phase=0 (subtract 1):
        [2, 0, 31] -> [2, 31, 0] -> [2, 30, -1]
        For MustDoList reverse:
        Uses -1 offset
        */

        // Initialize the reader with calculated offset
        template <typename TileShape_MNK, typename ParamsType>
        __device__
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            if constexpr (IsKeepList) {
                uint64_t const mask_offset = compute_keep_list_offset<TileShape_MNK>(params, bidb, bidh, m_block);
                list_ptr = &params.qk_skip_mask_args.attn_read_list[mask_offset];
            } else {
                // MustDoList initialization: single global list
                list_ptr = &params.qk_skip_mask_args.attn_must_do_list[0];
            }

            list_len = list_ptr[0];
            read_idx = Reverse ? list_len : 1;

            // we ignore the edge case which list_len == 0 because even in this case
            // we will be better off loading the first range because it's like to use the first range 2 timesteps ago
            advance();
        }

        // Advance to the next list range: load start_idx/end_idx from current read_idx, then move read_idx
        __device__
        void advance()
        {
            if constexpr (!Reverse) {
                start_idx = flash::warp_uniform(list_ptr[read_idx]);
                end_idx = flash::warp_uniform(list_ptr[read_idx + 1]);
                read_idx += 2;
            } else {
                start_idx = flash::warp_uniform(list_ptr[read_idx] + step);
                end_idx = flash::warp_uniform(list_ptr[read_idx - 1] + step);
                read_idx -= 2;
            }
        }

        __device__
        int last_n_block() const
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(list_ptr[list_len] + 1);
            } else {
                return flash::warp_uniform(list_ptr[1]);
            }
        }

        // True if n_block is still within the current range (before end_idx in iteration order).
        // Phase/step: forward (step=1) -> n_block < end_idx; backward (step=-1) -> n_block > end_idx.
        __device__
        bool in_range(int n_block) const
        {
            if constexpr (Phase) {
                return n_block < end_idx;
            } else {
                return n_block > end_idx;
            }
        }

        // === MustDoList methods ===

        // Check if we have more ranges to process
        __device__
        bool has_more()
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(read_idx <= list_len);
            } else {
                return flash::warp_uniform(read_idx >= 1);
            }
        }

        __device__
        bool is_n_block_in_range(int n_block) const
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(n_block >= start_idx && n_block < end_idx);
            } else {
                return flash::warp_uniform(n_block <= start_idx && n_block > end_idx);
            }
        }

        __device__
        bool has_passed_current_range(int n_block) const
        {
            if constexpr (!Reverse) {
                return flash::warp_uniform(n_block >= end_idx);
            } else {
                return flash::warp_uniform(n_block <= end_idx);
            }
        }

        __device__
        bool find_range(int n_block)
        {
            bool found = is_n_block_in_range(n_block);
            while (has_more() && !found && has_passed_current_range(n_block)) {
                advance();
                found = is_n_block_in_range(n_block);
            }
            return found;
        }
    };

    // ============================================================================
    // Type aliases for backward compatibility
    // ============================================================================
    template <bool ReverseMustDoList>
    using MustDoListReader = ListReader<false, ReverseMustDoList, !ReverseMustDoList>;

    template <bool Phase = true>
    using KeepListReader = ListReader<true, /*Reverse=*/true, Phase>;

    // ============================================================================
    // Helper struct for writing keep lists
    // Encapsulates all the logic for updating keep lists based on skip detection
    // ============================================================================
    struct KeepListWriter
    {
        // int *list_ptr;
        int16_t *list_ptr;
        int write_idx = 1;
        bool is_skipping = true;

        // Initialize the writer with calculated offset
        template <typename TileShape_MNK, typename ParamsType>
        __device__
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            uint64_t const mask_offset = compute_keep_list_offset<TileShape_MNK>(params, bidb, bidh, m_block);
            list_ptr = &params.qk_skip_mask_args.attn_write_list[mask_offset];
        }

        // Record a transition in skip state
        __device__
        void maybe_record_transition(bool skip, int n_block)
        {
            if (skip != is_skipping)
            {
                list_ptr[write_idx] = n_block;
                write_idx++;
                is_skipping = skip;
            }
        }

        // Record the end of a range (force transition to skipping)
        __device__
        void record_range_end(bool skip, int end_idx)
        {
            is_skipping = true;
            if (skip != is_skipping)
            {
                list_ptr[write_idx] = end_idx;
                write_idx++;
            }
        }

        // Finalize the keep list by writing the count
        __device__
        void finalize()
        {
            list_ptr[0] = write_idx - 1;
        }
    };

    // ============================================================================
    // Reader of the delayed keep list (not a "delayed reader").
    // Reads from the circular buffer that the producer fills with DelayAmount lag.
    // Consumer uses this to iterate over n_blocks that were recorded by the producer.
    // ============================================================================
    template <int DelayAmount, int NumMmaWarpGroups>
    struct DelayedKeepListReader
    {
        static constexpr int BufferSize = DelayAmount * 2;
        
        // Pointers to shared memory buffers
        int* n_blocks_buffer;
        int (*skip_tests)[kNumMmaWarpGroups];
        int last_n_block;

        // we start with -1 because the first call to next_n_block will increment it to 0.
        int index = -1;

        // Constructor to initialize with shared memory pointers
        __device__
        DelayedKeepListReader(int* n_blocks, int (*skip)[kNumMmaWarpGroups], int* final_n_block)
            : n_blocks_buffer(n_blocks), 
              skip_tests(skip), last_n_block(*final_n_block) {}


        __device__
        int next_n_block()
        {
            // issue: many uniform instructions!
            index = (index + 1) % BufferSize;
            return flash::warp_uniform(n_blocks_buffer[index]);
        }

        __device__
        void update_skip(bool skip, int warp_idx_in_warpgroup){
            if constexpr (NumMmaWarpGroups > 2) {
                atomicAnd(&(skip_tests[index][warp_idx_in_warpgroup]), static_cast<int>(skip));
            }else{
                skip_tests[index][warp_idx_in_warpgroup] &= static_cast<int>(skip);
            }
        }

        __device__
        bool has_more(int n_block)
        {
            return flash::warp_uniform(last_n_block != n_block);
        }

    };

    // ============================================================================
    // Delayed wrapper for KeepListWriter using circular buffer
    // Buffers operations and replays them after a specified delay
    // This allows the writer to lag behind the reader by DelayAmount iterations
    // ============================================================================
    template <int DelayAmount, bool Phase, bool HasMustDoList>
    struct DelayedKeepListWriter
    {
        static constexpr int BufferSize = DelayAmount * 2;
        
        // Pointers to shared memory buffers
        int* n_blocks_buffer;
        int* end_range_buffer;
        int (*skip_tests)[kNumMmaWarpGroups];

        //should reside in thread registers.
        KeepListWriter writer;
        bool replayed_skip;
        int record_idx = -1;
        int replay_idx = DelayAmount - 1;

        // Constructor to initialize with shared memory pointers
        __device__
        DelayedKeepListWriter(int* n_blocks, int* end_range, int (*skip)[kNumMmaWarpGroups])
            : n_blocks_buffer(n_blocks), end_range_buffer(end_range), 
              skip_tests(skip) {}

        /*
        DelayAmount = 4, and we iterate over the range [5, 0] example:
        Producer - record n_block=5 with record_idx = 0 -> load K0
        Consumer - waits for K0 to load -> load n_block=5 with index = 0 -> QK0 -> release K0 -> update skip with index 0
        Producer - record n_block=4 with record_idx = 1 -> load K1
        Consumer - waits for K1 to load -> load n_block=4 with index = 1 -> QK1 -> release K1 -> update skip with index 1
        Producer - replay n_block=nothing with replay_idx=2 -> load V0
        Consumer - waits for V0 to load -> PV0 -> release V0
        Producer - waits for K0 release -> record n_block=3 with record_idx = 2 -> load K2
        Consumer - waits for K2 to load -> load n_block=3 with index = 2 -> QK2 -> release K2 -> update skip with index 2
        Producer - replay n_block=nothing with replay_idx=3 -> load V1
        Consumer - waits for V1 to load -> PV1 -> release V1
        Producer - waits for K1 release -> record n_block=2 with record_idx = 3 -> load K3
        Consumer - waits for K3 to load -> load n_block=3 with index = 3 -> QK3 -> release K3 -> update skip with index 3
        Producer - waits for V0 release -> replay n_block=5 with replay_idx=0 -> load V2
        Consumer - waits for V2 to load -> PV2 -> release V2
        Producer - waits for K2 release -> record n_block=1 with record_idx = 0 -> load K4
        Consumer - waits for K4 to load -> load n_block=1 with index = 0 -> QK4 -> release K4 -> update skip with index 0
        */
        
        // Initialize the underlying writer
        template <typename TileShape_MNK, typename ParamsType>
        __device__
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            writer.template init<TileShape_MNK>(params, bidb, bidh, m_block);
            for (int i = 0; i < BufferSize; ++i) {
                #pragma unroll
                for (int j = 0; j < kNumMmaWarpGroups; ++j) {
                    skip_tests[i][j] = 1;
                }
                end_range_buffer[i] = -2;
            }
        }

        // consider: calling this when acquiring K for loading.
        // is_must_do: if true, this n_block is in the MustDoList and should not be skipped
        __device__
        void record_n_block(int n_block, bool is_must_do = false)
        {
            record_idx = (record_idx + 1) % BufferSize;
            // record the current n_block for replay in DelayAmount iterations from now.
            n_blocks_buffer[record_idx] = n_block;
            
            // If this n_block is in the MustDoList, initialize skip_tests to 0 (force computation)
            // Otherwise, initialize to 1 (allow skipping)
            int init_value = is_must_do ? 0 : 1;
            #pragma unroll
            for (int j = 0; j < kNumMmaWarpGroups; ++j) {
                skip_tests[record_idx][j] = init_value;
            }
        }

        __device__
        void record_range_end(int end_idx)
        {
            end_range_buffer[record_idx] = end_idx;
        }

        __device__
        void replay()
        {
            replay_idx = (replay_idx + 1) % BufferSize;

            replayed_skip = 1;
            #pragma unroll
            for (int j = 0; j < kNumMmaWarpGroups; ++j) {
                replayed_skip &= skip_tests[replay_idx][j];
            }

            int replayed_n_block = n_blocks_buffer[replay_idx];
            writer.maybe_record_transition(replayed_skip, replayed_n_block);

            int replayed_end_idx = end_range_buffer[replay_idx];
            if (replayed_end_idx != -2) {
                writer.record_range_end(replayed_skip, replayed_end_idx);
            }
            end_range_buffer[replay_idx] = -2;
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        // Finalize by flushing all remaining queue entries
        __device__
        void finalize()
        {
            // replay all of the buffer.
            // we don't need to warry about the buffer not being full becuase we init skip_tests
            // in such a way that it woudn't effect the resulting write keep list.
            for (int i = 0; i < DelayAmount; ++i) {
                replay();
            }

            writer.finalize();
        }
    };

    template <int BufferSize, bool Phase, bool HasMustDoList>
    struct KeepListStorage
    {
        alignas(16) int n_blocks_buffer[BufferSize]; // 4
        alignas(16) int end_range_buffer[BufferSize]; // 4
        alignas(16) int skip_tests[BufferSize][kNumMmaWarpGroups];
        int last_n_block[1]; // 4
        KeepListReader<Phase> reader;
        MustDoListReader<!Phase> must_do_reader;
    };

} // namespace flash
