/******************************************************************************
 * Simplified versions of reduce_max for illustration purposes
 * 
 * This file contains:
 * 1. Optimized version: overlaps int-to-float conversion with max finding
 * 2. Naive version: converts all rows to float first, then finds max
 ******************************************************************************/

#pragma once

// ============================================================================
// OPTIMIZED VERSION: Overlaps conversion with max finding
// ============================================================================
// Strategy: 
// - Row 0: Find max on int32_t values directly
// - Row 1: Convert int32_t to float while finding max on floats
// This overlaps conversion with max finding and avoids register pressure
// ============================================================================
template <int N>
void reduce_max_optimized(
    int32_t tensor[2][N],  // Input: 2 rows, N columns (int32_t)
    float max[2]            // Output: max values per row (float)
) {
    // Row 0: Find max on int32_t
    int32_t row0_max = tensor[0][0];
    for (int ni = 1; ni < N; ni++) {
        row0_max = max(row0_max, tensor[0][ni]);
    }
    max[0] = (float)row0_max;
    
    // Row 1: Convert to float while finding max
    max[1] = (float)tensor[1][0];
    tensor[1][0] = (int32_t)max[1];
    
    for (int ni = 1; ni < N; ni++) {
        max[1] = max(max[1], (float)tensor[1][ni]);
        tensor[1][ni] = (int32_t)((float)tensor[1][ni]);
    }

}

// ============================================================================
// NAIVE VERSION: Convert all rows to float first, then find max
// ============================================================================
// Strategy: Convert all int32_t values to float first, then find max
// Simpler but requires more register pressure (all floats stored)
// ============================================================================
template <int num_cols>
void reduce_max_naive(){
    int32_t tensor[2][num_cols],  // Input: 2 rows, N columns (int32_t)
    float current_max[2]            // Output: max values per row (float)

    // naive version: convert all to float first, then find max
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < num_cols; col++) {
            // inplace I2FP conversion
            tensor[row][col] = (float)tensor[row][col];
            // find max
            current_max[row] = max(current_max[row], tensor[row][col]);
        }
    }

    // optimized version: interleaving I2FP, float max, int max and IADD conversions
    for (int col = 0; col < num_cols; col++) {
        // row 0: max on int32_t
        current_max[0] = max(current_max[0], tensor[0][col]);
        // row 0: convert to float with IADD
        tensor[0][col] += magic_int32;
        // row 1: inplace I2FP conversion
        tensor[1][col] = (float)tensor[1][col];
        // row 1: find max
        current_max[1] = max(current_max[1], tensor[1][col]);
    }
}

// naive version: convert all to float first, then find max
for (int row = 0; row < 2; row++) {
    for (int col = 0; col < num_cols; col++) {
        // inplace int to float (I2FP) conversion
        tensor[row][col] = (float)tensor[row][col];
        // find max
        current_max[row] = max(current_max[row], tensor[row][col]);
    }
}

// optimized version: interleaving int max, IADD, I2FP, float max.
for (int col = 0; col < num_cols; col++) {
    // row 0: max on int's and after that convert to float (with IADD)
    current_max[0] = max(current_max[0], tensor[0][col]);
    tensor[0][col] += magic_int32;
    // row 1: convert to float (I2FP) and then find max
    tensor[1][col] = (float)tensor[1][col];
    current_max[1] = max(current_max[1], tensor[1][col]);
}