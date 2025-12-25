#pragma once

// __device__ float pow2_neg(int32_t neg_t) {
//     // Calculate 2^{-t} where neg_t = -t (t >= 0), so neg_t <= 0
//     // This computes 2^{neg_t} = 2^{-t} using bit manipulation
//     // IEEE 754 single precision: sign(1) + exponent(8) + mantissa(23)
//     // For 2^{neg_t}: sign=0, exponent=127+neg_t (biased), mantissa=0
    
//     // Handle underflow: if neg_t < -127, result underflows to 0
//     if (neg_t < -127) {
//         return 0.0f;
//     }
    
//     // Construct float bit pattern: (exponent << 23) | mantissa
//     // Exponent is stored in bits 30-23, biased by 127
//     // uint32_t exponent = 127 + neg_t;  // biased exponent (neg_t <= 0, so exponent <= 127)
//     uint32_t exponent = neg_t;  // biased exponent (neg_t <= 0, so exponent <= 127) we added 127 already in the max_local_dequan_s_int
//     uint32_t bits = exponent << 23;  // mantissa is 0, so just shift exponent
    
//     // Reinterpret bits as float
//     return *reinterpret_cast<float*>(&bits);
    
//     // Previous solution using ldexp:
//     // return ldexpf(1.0f, neg_t);
// }

// // question: is it legal to assume max_local >= 0? (because we normalized Q and K)
// __device__ float exp2_emulated(int32_t x, int32_t max_local, float max_global, float dequan_s, uint32_t dequan_s_uint, uint32_t dequan_s_frac) {
//     /*
//     this is what we calculate:
//     exp2f((x - max_local)*dequan_s) * expf(max_local*dequan_s - max_global)
//     = exp2f((x - max_local)*(dequan_s_int + dequan_s_frac)) * expf(max_local*dequan_s - max_global)
//     = exp2f((x - max_local)*dequan_s_int) * exp2f((x - max_local)*dequan_s_frac) * expf(max_local*dequan_s - max_global)

//     calculated once per row
//     max_correction = expf(max_local*dequan_s - max_global)

//     easy becuase we work with int's so we only need to calculate the exponent.
//     dequan_s_int_correction = exp2f((x - max_local)*dequan_s_int)

//     more hevy where we need to emulate exp2
//     exp2f((x - max_local)*dequan_s_frac)
//     */

//     // TODO: pass as argument, it's shared between all the elements in the row
//     float max_correction = exp2f(max_local*dequan_s - max_global);

//     // dequan x - max_local in fixed point
//     int64_t dequan_x_fixed_point = -(max_local - x)*dequan_s_fixed;

//     // TODO: pass as argument, it's shared between all the elements in the row
//     int32_t max_local_dequan_s_int = max_local*dequan_s_int + 127; // 127 is the bias for the exponent

//     // ~~~~~~~~~~ calculate exponent for exp2((x - local_max) * dequan_s_uint) ~~~~~~~~~~
//     // should be added to the exponent of exponent_x_dequan_s_frac
//     int32_t exponent_x_dequan_s_uint = x*dequan_s_uint - max_local_dequan_s_int; // MAD operation

//     // ~~~~~~~~~~ calculate exp2((x - local_max) * dequan_s_frac) ~~~~~~~~~~

//     // TODO: pass as argument, it's shared between all the elements in the row
//     int64_t max_local_dequan_s_frac = max_local*dequan_s_frac;

//     // higher 32 bits are the 
//     int64_t dequan_x_fixed_point = x*dequan_s_frac - max_local_dequan_s_frac;


// }

__device__ float exp2_emulated(int32_t x, int32_t max_local, float max_global, float dequan_s, int64_t dequan_s_fixed) {
    /*
    this is what we calculate:
    exp2f(x*dequan_s - max_global) =
    exp2f(x*dequan_s - max_local*dequan_s + max_local*dequan_s - max_global) =
    exp2f((x - max_local)*dequan_s + (max_local*dequan_s - max_global)) =
    exp2f((x - max_local)*dequan_s) * exp2f(max_local*dequan_s - max_global)

    y = exp2f((x - max_local)*dequan_s)
    // need to calculate once per row
    correction = exp2f(max_local*dequan_s - max_global)

    exp2f(x*dequan_s - max_global) = y * correction
    */

    // TODO: pass as argument, it's shared between all the elements in the row
    float max_correction = exp2f(max_local*dequan_s - max_global);

    // dequan (x - max_local) in fixed point
    // consider: try to use MAD operation and calculate max_local*dequan_s_fixed in advance
    // consider: using float64 MAD operation
    // int64_t dequan_x_fixed_point = (x - max_local)*dequan_s_fixed + (127LL << 32); // 127 is the bias for the exponent
    int64_t dequan_x_fixed_point = (x - max_local)*dequan_s_fixed + (127LL << 32); // 127 is the bias for the exponent
    dequan_x_fixed_point *= 1ULL << 23; // now we have the correct exponent (need to think about the sign bit interpretation)
    // consider: using this instead (need to think about overflows)
    // int64_t dequan_x_fixed_point = (x - max_local)*(dequan_s_fixed * (1ULL << 23)) + ((127LL << 32) * (1ULL << 23)); // 127 is the bias for the exponent
    // in this case 

}
