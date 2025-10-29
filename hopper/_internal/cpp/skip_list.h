#pragma once

#include "cute/tensor.hpp"

namespace flash
{

    using namespace cute;

    // ============================================================================
    // Helper struct for reading skip lists
    // Encapsulates all the logic for iterating through skip list ranges
    // ============================================================================
    struct SkipListReader
    {
        const int *list_ptr;
        int skip_list_len;
        int read_idx = 1;
        int start_idx;
        int end_idx;

        // Initialize the reader with calculated offset
        template <typename TileShape_MNK, typename ParamsType>
        __device__ __forceinline__ 
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            
            int const num_heads = get<2>(params.shape_Q);
            uint64_t const num_q_blocks = cute::ceil_div(get<0>(params.shape_Q), kBlockM);
            uint64_t const num_k_blocks = cute::ceil_div(get<0>(params.shape_K), kBlockN) + 1;
            const uint32_t q_i = ((uint32_t)m_block);
            uint64_t mask_offset = (bidb * num_heads * num_q_blocks * num_k_blocks) + 
                                   (bidh * num_q_blocks * num_k_blocks) + 
                                   (q_i * num_k_blocks);
            
            list_ptr = &params.qk_skip_mask_args.attn_read_list[mask_offset];
            skip_list_len = list_ptr[0];

            // we ignore the edge case which skip_list_len == 0 because even in this case
            // we will be better off loading the first range because it's like to use the first range 2 timesteps ago
            load_range();
        }

        __device__ __forceinline__ 
        void load_range()
        {
            start_idx = list_ptr[read_idx];
            end_idx = list_ptr[read_idx + 1];
        }

        // Advance to the next skip list range
        __device__ __forceinline__ 
        void advance()
        {
            read_idx += 2;
        }

        // Check if we have more ranges to process
        __device__ __forceinline__ 
        bool has_more() const
        {
            return read_idx <= skip_list_len;
        }

        // Check if we have more ranges to process
        __device__ __forceinline__ 
        bool has_more_n_block(int const n_block) const
        {
            // return has_more() & (n_block - 1 >= end_idx);
            return has_more() & (n_block > end_idx);
        }
    };

    // ============================================================================
    // Helper struct for writing skip lists
    // Encapsulates all the logic for updating skip lists based on skip detection
    // ============================================================================
    struct SkipListWriter
    {
        int *list_ptr;
        int write_idx = 1;
        bool is_skipping = true;

        // Initialize the writer with calculated offset
        template <typename TileShape_MNK, typename ParamsType>
        __device__ __forceinline__ 
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            
            int const num_heads = get<2>(params.shape_Q);
            uint64_t const num_q_blocks = cute::ceil_div(get<0>(params.shape_Q), kBlockM);
            uint64_t const num_k_blocks = cute::ceil_div(get<0>(params.shape_K), kBlockN) + 1;
            const uint32_t q_i = ((uint32_t)m_block);
            uint64_t mask_offset = (bidb * num_heads * num_q_blocks * num_k_blocks) + 
                                   (bidh * num_q_blocks * num_k_blocks) + 
                                   (q_i * num_k_blocks);
            
            list_ptr = &params.qk_skip_mask_args.attn_write_list[mask_offset];
        }

        // Record a transition in skip state
        __device__ __forceinline__ 
        void record_transition(bool skip, int n_block)
        {
            if (skip != is_skipping)
            {
                list_ptr[write_idx] = n_block;
                write_idx++;
                is_skipping = skip;
            }
        }

        // Record the end of a range (force transition to skipping)
        __device__ __forceinline__ 
        void record_range_end(bool skip, int end_idx)
        {
            is_skipping = true;
            if (skip != is_skipping)
            {
                list_ptr[write_idx] = end_idx;
                write_idx++;
            }
        }

        // Finalize the skip list by writing the count
        __device__ __forceinline__ 
        void finalize()
        {
            list_ptr[0] = write_idx - 1;
        }
    };

    // ============================================================================
    // Delayed wrapper for SkipListReader using circular buffer
    // Buffers operations and replays them after a specified delay
    // This allows the reader to lag behind the writer by DelayAmount iterations
    // ============================================================================
    template <int DelayAmount>
    struct DelayedSkipListReader
    {
        static constexpr int BufferSize = DelayAmount * 2;
        
        // Pointers to shared memory buffers
        int* n_blocks_buffer;
        int* end_range_buffer;
        int (*skip_tests)[4];
        int* stop_condition_buffer;

        // we start with -1 because the first call to next_n_block will increment it to 0.
        int index = -1;

        // Constructor to initialize with shared memory pointers
        __device__ __forceinline__
        DelayedSkipListReader(int* n_blocks, int* end_range, int (*skip)[4], int* stop_cond)
            : n_blocks_buffer(n_blocks), end_range_buffer(end_range), 
              skip_tests(skip), stop_condition_buffer(stop_cond) {}

        // Default constructor
        __device__ __forceinline__
        DelayedSkipListReader() = default;

        __device__ __forceinline__ int next_n_block()
        {
            index = (index + 1) % BufferSize;
            return n_blocks_buffer[index];
        }

        __device__ __forceinline__ void update_skip(bool skip, int warp_idx_in_warpgroup){
            // consider: using atomic here
            atomicAnd(&(skip_tests[index][warp_idx_in_warpgroup]), static_cast<int>(skip));
        }

        __device__ __forceinline__ bool has_more()
        {
            return stop_condition_buffer[index];
        }

    };

    // ============================================================================
    // Delayed wrapper for SkipListWriter using circular buffer
    // Buffers operations and replays them after a specified delay
    // This allows the writer to lag behind the reader by DelayAmount iterations
    // ============================================================================
    template <int DelayAmount>
    struct DelayedSkipListWriter
    {
        static constexpr int BufferSize = DelayAmount * 2;
        
        // Pointers to shared memory buffers
        int* n_blocks_buffer;
        int* end_range_buffer;
        int (*skip_tests)[4];
        int* stop_condition_buffer;

        //should reside in thread registers.
        SkipListWriter writer;
        bool replayed_skip;
        int record_idx = 0;
        int replay_idx = DelayAmount;

        // Constructor to initialize with shared memory pointers
        __device__ __forceinline__
        DelayedSkipListWriter(int* n_blocks, int* end_range, int (*skip)[4], int* stop_cond)
            : n_blocks_buffer(n_blocks), end_range_buffer(end_range), 
              skip_tests(skip), stop_condition_buffer(stop_cond) {}

        // Default constructor
        __device__ __forceinline__
        DelayedSkipListWriter() = default;

        /*
        DelayAmount = 4, example:
        K0 - record_n_block -> record_idx = 0 -> record_idx = 1
        K1 - record_n_block -> record_idx = 1 -> record_idx = 2
        V0 - replay -> replay_idx = 2 -> replay_idx = 3
        K2 - record_n_block -> record_idx = 2 -> record_idx = 3
        V1 - replay -> replay_idx = 3 -> replay_idx = 0
        K3 - record_n_block -> record_idx = 3 -> record_idx = 0
        V2 - replay -> replay_idx = 0 -> replay_idx = 1
        */
        
        // Initialize the underlying writer
        template <typename TileShape_MNK, typename ParamsType>
        __device__ __forceinline__ 
        void init(const ParamsType &params, int bidb, int bidh, int m_block)
        {
            writer.template init<TileShape_MNK>(params, bidb, bidh, m_block);
            for (int i = 0; i < BufferSize; ++i) {
                // init everything to true because we would write something only when we
                // encounter a false skip result. and now it could happen only after the buffers are full.
                skip_tests[i][0] = 1;
                skip_tests[i][1] = 1;
                skip_tests[i][2] = 1;
                skip_tests[i][3] = 1;
                end_range_buffer[i] = -2;
                stop_condition_buffer[i] = true;
            }
        }

        // consider: calling this when acquiring K for loading.
        __device__ __forceinline__ 
        void record_n_block(int n_block)
        {
            // record the current n_block for replay in DelayAmount iterations from now.
            n_blocks_buffer[record_idx] = n_block;
            record_idx = (record_idx + 1) % BufferSize;
        }

        __device__ __forceinline__ 
        void record_range_end(int end_idx)
        {
            // we save into previous index!
            // end_range_buffer[(record_idx - 1) % DelayAmount] = end_idx;
            end_range_buffer[(record_idx + BufferSize - 1) % BufferSize] = end_idx;
        }

        __device__ __forceinline__ 
        // void record_final_iter()
        void record_final_iter(bool not_final_iter)
        {
            // stop_condition_buffer[(record_idx + BufferSize - 1) % BufferSize] = false;
            stop_condition_buffer[record_idx] = not_final_iter;
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // consider: making these functions private and making replay() public.

        __device__ __forceinline__ 
        void replay_transition()
        {
            // calculate the replayed skip result from DelayAmount ago.
            replayed_skip = 
                skip_tests[replay_idx][0] &
                skip_tests[replay_idx][1] &
                skip_tests[replay_idx][2] &
                skip_tests[replay_idx][3];
            
            // replay the n_block from DelayAmount ago.
            int replayed_n_block = n_blocks_buffer[replay_idx];

            // replay record_transition with the values from DelayAmount ago.
            writer.record_transition(replayed_skip, replayed_n_block);

            // reset the skip tests for reuse by the consumers.
            skip_tests[replay_idx][0] = 1;
            skip_tests[replay_idx][1] = 1;
            skip_tests[replay_idx][2] = 1;
            skip_tests[replay_idx][3] = 1;
        }

        __device__ __forceinline__ 
        void replay_end_range()
        {
            int replayed_end_idx = end_range_buffer[replay_idx];
            if (replayed_end_idx != -2) {
                writer.record_range_end(replayed_skip, replayed_end_idx);
            }
            end_range_buffer[replay_idx] = -2;
        }

        __device__ __forceinline__ 
        void replay()
        {
            replay_transition();
            replay_end_range();
            replay_idx = (replay_idx + 1) % BufferSize;
        }

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        // Finalize by flushing all remaining queue entries
        __device__ __forceinline__ 
        void finalize()
        {
            // replay all of the buffer.
            // we don't need to warry about the buffer not being full becuase we init skip_tests
            // in such a way that it woudn't effect the resulting write skip list.
            for (int i = 0; i < DelayAmount; ++i) {
                replay();
            }

            writer.finalize();
        }
    };

} // namespace flash

