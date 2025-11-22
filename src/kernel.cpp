#include "dsta/kernel.h"
#include "dsta/common.h"
#include <immintrin.h>
#include <cstdint>

namespace dsta {

    // Pre-computed lookup table for weight expansion (byte -> 4 floats)
    // Maps 2-bit weights to float values: 00->0.0, 01->1.0, 10->-1.0
    // This replaces the decoding overhead with a simple memory lookup
    alignas(64) static const float weight_lut[256][4] = {
        #define V(b, s) ((((b)>>(s))&3)==1 ? 1.0f : ((((b)>>(s))&3)==2 ? -1.0f : 0.0f))
        #define M(n) { V(n,0), V(n,2), V(n,4), V(n,6) }

        M(0), M(1), M(2), M(3), M(4), M(5), M(6), M(7), M(8), M(9), M(10), M(11), M(12), M(13), M(14), M(15),
        M(16), M(17), M(18), M(19), M(20), M(21), M(22), M(23), M(24), M(25), M(26), M(27), M(28), M(29), M(30), M(31),
        M(32), M(33), M(34), M(35), M(36), M(37), M(38), M(39), M(40), M(41), M(42), M(43), M(44), M(45), M(46), M(47),
        M(48), M(49), M(50), M(51), M(52), M(53), M(54), M(55), M(56), M(57), M(58), M(59), M(60), M(61), M(62), M(63),
        M(64), M(65), M(66), M(67), M(68), M(69), M(70), M(71), M(72), M(73), M(74), M(75), M(76), M(77), M(78), M(79),
        M(80), M(81), M(82), M(83), M(84), M(85), M(86), M(87), M(88), M(89), M(90), M(91), M(92), M(93), M(94), M(95),
        M(96), M(97), M(98), M(99), M(100), M(101), M(102), M(103), M(104), M(105), M(106), M(107), M(108), M(109), M(110), M(111),
        M(112), M(113), M(114), M(115), M(116), M(117), M(118), M(119), M(120), M(121), M(122), M(123), M(124), M(125), M(126), M(127),
        M(128), M(129), M(130), M(131), M(132), M(133), M(134), M(135), M(136), M(137), M(138), M(139), M(140), M(141), M(142), M(143),
        M(144), M(145), M(146), M(147), M(148), M(149), M(150), M(151), M(152), M(153), M(154), M(155), M(156), M(157), M(158), M(159),
        M(160), M(161), M(162), M(163), M(164), M(165), M(166), M(167), M(168), M(169), M(170), M(171), M(172), M(173), M(174), M(175),
        M(176), M(177), M(178), M(179), M(180), M(181), M(182), M(183), M(184), M(185), M(186), M(187), M(188), M(189), M(190), M(191),
        M(192), M(193), M(194), M(195), M(196), M(197), M(198), M(199), M(200), M(201), M(202), M(203), M(204), M(205), M(206), M(207),
        M(208), M(209), M(210), M(211), M(212), M(213), M(214), M(215), M(216), M(217), M(218), M(219), M(220), M(221), M(222), M(223),
        M(224), M(225), M(226), M(227), M(228), M(229), M(230), M(231), M(232), M(233), M(234), M(235), M(236), M(237), M(238), M(239),
        M(240), M(241), M(242), M(243), M(244), M(245), M(246), M(247), M(248), M(249), M(250), M(251), M(252), M(253), M(254), M(255)
        #undef M
        #undef V
    };

    // AVX-512 optimized kernel (16 floats per iteration)
    static void dsta_kernel_avx512(
        float* output,
        const float* input,
        const uint8_t* packed_weights,
        const uint64_t* mask,
        size_t input_dim,
        size_t output_dim
    ) {
        // Number of 64-bit blocks in the mask
        size_t mask_len = (input_dim + 63) / 64;
        
        // Bytes per row in packed weights (4 weights per byte)
        size_t row_stride = output_dim / 4;

        // Iterate over the sparsity mask
        for (size_t m = 0; m < mask_len; ++m) {
            uint64_t current_mask = mask[m];
            
            // Skip empty blocks entirely
            if (current_mask == 0) continue;

            // Iterate over set bits (active inputs)
            while (current_mask) {
                // Find index of first set bit
                int bit_idx = _tzcnt_u64(current_mask);
                int global_input_idx = m * 64 + bit_idx;

                // Load the activation value (scalar -> broadcast vector)
                float activation_val = input[global_input_idx];
                __m512 val_vec = _mm512_set1_ps(activation_val);

                // Pointer to the corresponding weight row
                const uint8_t* w_row_ptr = packed_weights + (global_input_idx * row_stride);
                float* out_ptr = output;

                // Inner loop: Process output vector in chunks of 16
                // 16 weights = 4 bytes (32 bits)
                size_t j = 0;
                
                for (; j < output_dim; j += 16) {
                    // 2. Load Packed Weights (32 bits = 16 weights)
                    uint32_t w_packed = *(const uint32_t*)(w_row_ptr + (j / 4));

                    // Optimization: Skip if all weights in this block are zero
                    if (w_packed == 0) continue;

                    // 1. Load Accumulator (Y)
                    __m512 acc = _mm512_load_ps(out_ptr + j);

                    // 3. Decode Ternary Weights to Masks using PEXT
                    // Encoding: 00=0, 01=+1, 10=-1
                    uint16_t low_bits = (uint16_t)_pext_u32(w_packed, 0x55555555);
                    uint16_t high_bits = (uint16_t)_pext_u32(w_packed, 0xAAAAAAAA);
                    
                    // k_pos: pattern 01 (low=1, high=0) -> +1
                    // k_neg: pattern 10 (low=0, high=1) -> -1
                    __mmask16 k_pos = (__mmask16)(low_bits & ~high_bits);
                    __mmask16 k_neg = (__mmask16)(~low_bits & high_bits);

                    // 4. Select-and-Add / Select-and-Sub
                    acc = _mm512_mask_add_ps(acc, k_pos, acc, val_vec);
                    acc = _mm512_mask_sub_ps(acc, k_neg, acc, val_vec);

                    // 5. Store Accumulator
                    _mm512_store_ps(out_ptr + j, acc);
                }

                // Clear the processed bit
                current_mask = _blsr_u64(current_mask);
            }
        }
    }

    // AVX2 fallback kernel (8 floats per iteration) - OPTIMIZED
    // Now uses direct FMA and Table Lookup instead of pext/blend decoding
    static void dsta_kernel_avx2(
        float* output,
        const float* input,
        const uint8_t* packed_weights,
        const uint64_t* mask,
        size_t input_dim,
        size_t output_dim
    ) {
        size_t mask_len = (input_dim + 63) / 64;
        size_t row_stride = output_dim / 4;

        // Iterate over the sparsity mask
        for (size_t m = 0; m < mask_len; ++m) {
            uint64_t current_mask = mask[m];
            if (current_mask == 0) continue;

            // OPTIMIZATION: Process multiple active inputs from same mask block together
            // This improves cache locality and reduces loop overhead
            int base_idx = m * 64;
            
            // Iterate over set bits (active inputs)
            while (current_mask) {
                int bit_idx = _tzcnt_u64(current_mask);
                int global_input_idx = base_idx + bit_idx;

                float activation_val = input[global_input_idx];
                __m256 val_vec = _mm256_set1_ps(activation_val);

                const uint8_t* w_row_ptr = packed_weights + (global_input_idx * row_stride);
                float* out_ptr = output;
                
                // Prefetch next weight row if there are more active inputs
                if (_blsr_u64(current_mask) != 0) {
                    int next_bit = _tzcnt_u64(_blsr_u64(current_mask));
                    int next_input_idx = base_idx + next_bit;
                    _mm_prefetch((const char*)(packed_weights + (next_input_idx * row_stride)), _MM_HINT_T0);
                }

                // Inner loop: Process output vector in chunks of 8 (AVX2)
                // AGGRESSIVELY OPTIMIZED: Unroll 4 blocks at once + FMA
                size_t j = 0;
                
                // Process 4 blocks (32 outputs) per iteration
                for (; j + 32 <= output_dim; j += 32) {
                    // Prefetch next
                    _mm_prefetch((const char*)(out_ptr + j + 32), _MM_HINT_T1);
                    
                    // Block 0 (Bytes 0,1 -> Weights 0..7)
                    __m256 acc0 = _mm256_load_ps(out_ptr + j);
                    uint8_t b0_0 = w_row_ptr[j / 4];
                    uint8_t b0_1 = w_row_ptr[j / 4 + 1];
                    if (b0_0 | b0_1) { // Optimization: skip if both bytes zero
                        __m128 w0_lo = _mm_load_ps(weight_lut[b0_0]);
                        __m128 w0_hi = _mm_load_ps(weight_lut[b0_1]);
                        __m256 w0 = _mm256_insertf128_ps(_mm256_castps128_ps256(w0_lo), w0_hi, 1);
                        acc0 = _mm256_fmadd_ps(w0, val_vec, acc0);
                        _mm256_store_ps(out_ptr + j, acc0);
                    }

                    // Block 1 (Bytes 2,3 -> Weights 8..15)
                    __m256 acc1 = _mm256_load_ps(out_ptr + j + 8);
                    uint8_t b1_0 = w_row_ptr[j / 4 + 2];
                    uint8_t b1_1 = w_row_ptr[j / 4 + 3];
                    if (b1_0 | b1_1) {
                        __m128 w1_lo = _mm_load_ps(weight_lut[b1_0]);
                        __m128 w1_hi = _mm_load_ps(weight_lut[b1_1]);
                        __m256 w1 = _mm256_insertf128_ps(_mm256_castps128_ps256(w1_lo), w1_hi, 1);
                        acc1 = _mm256_fmadd_ps(w1, val_vec, acc1);
                        _mm256_store_ps(out_ptr + j + 8, acc1);
                    }
                    
                    // Block 2 (Bytes 4,5 -> Weights 16..23)
                    __m256 acc2 = _mm256_load_ps(out_ptr + j + 16);
                    uint8_t b2_0 = w_row_ptr[j / 4 + 4];
                    uint8_t b2_1 = w_row_ptr[j / 4 + 5];
                    if (b2_0 | b2_1) {
                        __m128 w2_lo = _mm_load_ps(weight_lut[b2_0]);
                        __m128 w2_hi = _mm_load_ps(weight_lut[b2_1]);
                        __m256 w2 = _mm256_insertf128_ps(_mm256_castps128_ps256(w2_lo), w2_hi, 1);
                        acc2 = _mm256_fmadd_ps(w2, val_vec, acc2);
                        _mm256_store_ps(out_ptr + j + 16, acc2);
                    }

                    // Block 3 (Bytes 6,7 -> Weights 24..31)
                    __m256 acc3 = _mm256_load_ps(out_ptr + j + 24);
                    uint8_t b3_0 = w_row_ptr[j / 4 + 6];
                    uint8_t b3_1 = w_row_ptr[j / 4 + 7];
                    if (b3_0 | b3_1) {
                        __m128 w3_lo = _mm_load_ps(weight_lut[b3_0]);
                        __m128 w3_hi = _mm_load_ps(weight_lut[b3_1]);
                        __m256 w3 = _mm256_insertf128_ps(_mm256_castps128_ps256(w3_lo), w3_hi, 1);
                        acc3 = _mm256_fmadd_ps(w3, val_vec, acc3);
                        _mm256_store_ps(out_ptr + j + 24, acc3);
                    }
                }
                
                // Handle remaining blocks
                for (; j < output_dim; j += 8) {
                    uint8_t b0 = w_row_ptr[j / 4];
                    uint8_t b1 = w_row_ptr[j / 4 + 1];
                    if ((b0 | b1) == 0) continue;
                    
                    __m256 acc = _mm256_load_ps(out_ptr + j);
                    __m128 w_lo = _mm_load_ps(weight_lut[b0]);
                    __m128 w_hi = _mm_load_ps(weight_lut[b1]);
                    __m256 w = _mm256_insertf128_ps(_mm256_castps128_ps256(w_lo), w_hi, 1);
                    
                    acc = _mm256_fmadd_ps(w, val_vec, acc);
                    _mm256_store_ps(out_ptr + j, acc);
                }

                current_mask = _blsr_u64(current_mask);
            }
        }
    }

    // Main dispatcher function (Input-Stationary)
    void dsta_kernel_forward(
        float* output,
        const float* input,
        const uint8_t* packed_weights,
        const uint64_t* mask,
        size_t input_dim,
        size_t output_dim
    ) {
        static SIMDLevel level = detect_simd_level();
        
        switch (level) {
            case SIMDLevel::AVX512:
                dsta_kernel_avx512(output, input, packed_weights, mask, input_dim, output_dim);
                break;
            case SIMDLevel::AVX2:
                dsta_kernel_avx2(output, input, packed_weights, mask, input_dim, output_dim);
                break;
            default:
                dsta_kernel_avx2(output, input, packed_weights, mask, input_dim, output_dim);
                break;
        }
    }

} // namespace dsta
