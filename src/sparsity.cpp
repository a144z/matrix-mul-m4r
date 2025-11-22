#include "dsta/sparsity.h"
#include <cmath>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace dsta {

    std::vector<uint64_t> SparsityPredictor::generate_mask(const std::vector<float>& input, float threshold) {
        size_t size = input.size();
        size_t num_u64 = (size + 63) / 64;
        std::vector<uint64_t> mask(num_u64, 0);

        // Optimized: Use SIMD to process 8 floats at a time
        #ifdef __AVX2__
        size_t i = 0;
        __m256 threshold_vec = _mm256_set1_ps(threshold);
        __m256 neg_threshold_vec = _mm256_set1_ps(-threshold);
        
        for (; i + 8 <= size; i += 8) {
            __m256 vals = _mm256_loadu_ps(&input[i]);
            // Check if |vals| > threshold using SIMD
            __m256 abs_vals = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vals); // abs
            __m256 cmp_gt = _mm256_cmp_ps(abs_vals, threshold_vec, _CMP_GT_OQ);
            int mask_bits = _mm256_movemask_ps(cmp_gt);
            
            if (mask_bits != 0) {
                size_t u64_idx = i / 64;
                size_t bit_offset = i % 64;
                // Set bits in the mask
                for (int bit = 0; bit < 8; ++bit) {
                    if (mask_bits & (1 << bit)) {
                        size_t global_idx = i + bit;
                        size_t u64_idx_local = global_idx / 64;
                        size_t bit_idx_local = global_idx % 64;
                        mask[u64_idx_local] |= (1ULL << bit_idx_local);
                    }
                }
            }
        }
        #endif
        
        // Scalar tail
        for (; i < size; ++i) {
            if (std::abs(input[i]) > threshold) {
                size_t u64_idx = i / 64;
                size_t bit_idx = i % 64;
                mask[u64_idx] |= (1ULL << bit_idx);
            }
        }
        return mask;
    }

    size_t SparsityPredictor::count_active_blocks(const std::vector<uint64_t>& mask) {
        size_t count = 0;
        for (uint64_t val : mask) {
            // standard popcount
            // in C++20: std::popcount(val)
            // portable fallback:
            uint64_t v = val;
            v = v - ((v >> 1) & 0x5555555555555555);
            v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333);
            count += (((v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56;
        }
        return count;
    }

} // namespace dsta

