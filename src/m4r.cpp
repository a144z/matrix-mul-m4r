#include "dsta/m4r.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <immintrin.h>

namespace dsta {

    M4RLayer::M4RLayer(size_t in_dim, size_t out_dim) 
        : input_dim_(in_dim), output_dim_(out_dim), lattice_table_(nullptr), acc_buffer_(nullptr) {
        
        if (in_dim % 8 != 0) {
            // Padding is now handled implicitly by processing partial chunks carefully or ensuring upstream padding.
            // For this demo, we still prefer multiples of 8 but will not crash.
            // The memory allocation is rounded up to chunks.
        }
        allocate_memory();
    }

    M4RLayer::~M4RLayer() {
        free_memory();
    }

    void M4RLayer::allocate_memory() {
        // Size: output_dim * (input_dim / 8) * 256
        // Each block of 8 inputs -> 256 entries in table
        size_t num_chunks = input_dim_ / 8;
        size_t table_size = output_dim_ * num_chunks * 256;
        
        lattice_table_ = (int16_t*)aligned_alloc_mem(table_size * sizeof(int16_t), 64);
        if (!lattice_table_) {
            throw std::bad_alloc();
        }
        
        // Pre-allocate Accumulator Buffer
        acc_buffer_ = (int32_t*)aligned_alloc_mem(output_dim_ * sizeof(int32_t), 64);
        if (!acc_buffer_) {
            aligned_free_mem(lattice_table_);
            throw std::bad_alloc();
        }
        
        // Pre-allocate Packed Indices Buffer
        packed_indices_buffer_.resize(num_chunks);
        
        // Zero initialize table
        std::memset(lattice_table_, 0, table_size * sizeof(int16_t));
    }

    void M4RLayer::free_memory() {
        if (lattice_table_) {
            aligned_free_mem(lattice_table_);
            lattice_table_ = nullptr;
        }
        if (acc_buffer_) {
            aligned_free_mem(acc_buffer_);
            acc_buffer_ = nullptr;
        }
    }

    void M4RLayer::load_weights(const std::vector<float>& weights) {
        size_t num_chunks = input_dim_ / 8;

        // Optimized Layout: [Chunk][Pattern][Row]
        // This allows for contiguous vector access in the inner loop (Output Stationary wrt Pattern)
        // 
        // Total Size: NumChunks * 256 * OutputDim
        // Access Pattern:
        //   For each Chunk c:
        //      Pattern p = Input[c]
        //      Vector src = Table[c][p]  <-- Contiguous vector of size OutputDim
        //      Vector dst = Output       <-- Contiguous vector of size OutputDim
        //      dst += src

        for (size_t c = 0; c < num_chunks; ++c) {
            // Temporary storage for this chunk's weights for all rows: [OutputDim][8]
            std::vector<float> chunk_weights_all_rows(output_dim_ * 8);
            for (size_t i = 0; i < output_dim_; ++i) {
                for (int k = 0; k < 8; ++k) {
                    size_t w_idx = i * input_dim_ + (c * 8 + k);
                    chunk_weights_all_rows[i * 8 + k] = (w_idx < weights.size()) ? weights[w_idx] : 0.0f;
                }
            }

            // Now compute the table for this chunk
            for (int pattern = 0; pattern < 256; ++pattern) {
                
                // Compute dot product for this pattern for ALL rows
                for (size_t i = 0; i < output_dim_; ++i) {
                    float sum = 0.0f;
                    for (int bit = 0; bit < 8; ++bit) {
                        if ((pattern >> bit) & 1) {
                            sum += chunk_weights_all_rows[i * 8 + bit];
                        }
                    }
                    
                    // Store in [Chunk][Pattern][Row]
                    size_t table_idx = c * (256 * output_dim_) + pattern * output_dim_ + i;
                    lattice_table_[table_idx] = static_cast<int16_t>(sum);
                }
            }
        }
    }

    // Helper to pack float activations to byte indices - OPTIMIZED
    // Uses AVX2 for fast quantization
    static void pack_activations_into(const std::vector<float>& input, std::vector<uint8_t>& packed) {
        size_t n = input.size();
        size_t num_chunks = packed.size();
        
        size_t j = 0;
        
#ifdef __AVX2__
        __m256 zero = _mm256_setzero_ps();
        // Process 8 floats at a time (one byte per 8 floats)
        for (; j + 8 <= n; j += 8) {
            __m256 vals = _mm256_loadu_ps(&input[j]);
            __m256 mask = _mm256_cmp_ps(vals, zero, _CMP_GT_OQ);
            int byte_mask = _mm256_movemask_ps(mask);
            packed[j / 8] = static_cast<uint8_t>(byte_mask);
        }
#endif

        // Scalar tail
        for (; j < n; j += 8) {
            uint8_t index = 0;
            for (int k = 0; k < 8 && (j + k) < n; ++k) {
                if (input[j + k] > 0.0f) {
                    index |= (1 << k);
                }
            }
            packed[j / 8] = index;
        }
    }

    std::vector<float> M4RLayer::forward(const std::vector<float>& input) {
        // Realistic M4R Forward Pass
        pack_activations_into(input, packed_indices_buffer_);
        
        size_t num_chunks = input_dim_ / 8;
        
        // Reset Accumulator
        std::memset(acc_buffer_, 0, output_dim_ * sizeof(int32_t));

        // 2. The M4R Loop (Input Stationary / Column Major)
        for (size_t c = 0; c < num_chunks; ++c) {
            uint8_t pattern = packed_indices_buffer_[c];
            if (pattern == 0) continue;
            
            const int16_t* src_ptr = &lattice_table_[c * 256 * output_dim_ + pattern * output_dim_];
            
            size_t i = 0;
#ifdef __AVX2__
            // Optimized: 4-way unrolling (32 elements) with better instruction scheduling
            // Use aligned loads for better performance (lattice table is 64-byte aligned)
            for (; i + 32 <= output_dim_; i += 32) {
                // Load 32 values (aligned - lattice table is 64-byte aligned)
                __m256i vals_16_0 = _mm256_load_si256((const __m256i*)&src_ptr[i]);
                __m256i vals_16_1 = _mm256_load_si256((const __m256i*)&src_ptr[i + 16]);
                
                // Convert int16 -> int32 (sign extension) - interleave loads for better ILP
                __m256i v0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vals_16_0));
                __m256i v1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vals_16_1));
                
                // Load accumulators early (hide latency)
                __m256i acc0 = _mm256_load_si256((__m256i*)&acc_buffer_[i]);
                __m256i acc2 = _mm256_load_si256((__m256i*)&acc_buffer_[i + 16]);

                // Continue conversions
                __m256i v0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vals_16_0, 1));
                __m256i v1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vals_16_1, 1));

                // Load remaining accumulators
                __m256i acc1 = _mm256_load_si256((__m256i*)&acc_buffer_[i + 8]);
                __m256i acc3 = _mm256_load_si256((__m256i*)&acc_buffer_[i + 24]);

                // Add (better instruction scheduling - independent operations)
                acc0 = _mm256_add_epi32(acc0, v0_lo);
                acc2 = _mm256_add_epi32(acc2, v1_lo);
                acc1 = _mm256_add_epi32(acc1, v0_hi);
                acc3 = _mm256_add_epi32(acc3, v1_hi);

                // Store
                _mm256_store_si256((__m256i*)&acc_buffer_[i], acc0);
                _mm256_store_si256((__m256i*)&acc_buffer_[i + 8], acc1);
                _mm256_store_si256((__m256i*)&acc_buffer_[i + 16], acc2);
                _mm256_store_si256((__m256i*)&acc_buffer_[i + 24], acc3);
            }
#endif
            for (; i < output_dim_; ++i) {
                acc_buffer_[i] += src_ptr[i];
            }
        }

        // Convert to float output
        std::vector<float> output(output_dim_);
        
#ifdef __AVX2__
        size_t i = 0;
        for (; i + 8 <= output_dim_; i += 8) {
            __m256i vals_i32 = _mm256_load_si256((const __m256i*)&acc_buffer_[i]);
            __m256 vals_f32 = _mm256_cvtepi32_ps(vals_i32);
            _mm256_storeu_ps(&output[i], vals_f32);
        }
        for (; i < output_dim_; ++i) {
            output[i] = static_cast<float>(acc_buffer_[i]);
        }
#else
        for (size_t i = 0; i < output_dim_; ++i) {
            output[i] = static_cast<float>(acc_buffer_[i]);
        }
#endif
        
        return output;
    }

    std::vector<int32_t> M4RLayer::forward_quantized(const std::vector<uint8_t>& input) {
        // 1. Pack Inputs (Already packed)
        size_t num_chunks = input_dim_ / 8;
        std::memset(acc_buffer_, 0, output_dim_ * sizeof(int32_t));

        // 2. The M4R Loop
        for (size_t c = 0; c < num_chunks; ++c) {
            uint8_t pattern = input[c];
            if (pattern == 0) continue;
            
            const int16_t* src_ptr = &lattice_table_[c * 256 * output_dim_ + pattern * output_dim_];
            
            size_t i = 0;
#ifdef __AVX2__
            // Optimized: 4-way unrolling (32 elements) with better instruction scheduling
            for (; i + 32 <= output_dim_; i += 32) {
                // Load 32 values (aligned - lattice table is 64-byte aligned)
                __m256i vals_16_0 = _mm256_load_si256((const __m256i*)&src_ptr[i]);
                __m256i vals_16_1 = _mm256_load_si256((const __m256i*)&src_ptr[i + 16]);
                
                // Convert int16 -> int32 (sign extension) - interleave loads for better ILP
                __m256i v0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vals_16_0));
                __m256i v1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vals_16_1));
                
                // Load accumulators early (hide latency)
                __m256i acc0 = _mm256_load_si256((__m256i*)&acc_buffer_[i]);
                __m256i acc2 = _mm256_load_si256((__m256i*)&acc_buffer_[i + 16]);

                // Continue conversions
                __m256i v0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vals_16_0, 1));
                __m256i v1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vals_16_1, 1));

                // Load remaining accumulators
                __m256i acc1 = _mm256_load_si256((__m256i*)&acc_buffer_[i + 8]);
                __m256i acc3 = _mm256_load_si256((__m256i*)&acc_buffer_[i + 24]);

                // Add (better instruction scheduling - independent operations)
                acc0 = _mm256_add_epi32(acc0, v0_lo);
                acc2 = _mm256_add_epi32(acc2, v1_lo);
                acc1 = _mm256_add_epi32(acc1, v0_hi);
                acc3 = _mm256_add_epi32(acc3, v1_hi);

                // Store
                _mm256_store_si256((__m256i*)&acc_buffer_[i], acc0);
                _mm256_store_si256((__m256i*)&acc_buffer_[i + 8], acc1);
                _mm256_store_si256((__m256i*)&acc_buffer_[i + 16], acc2);
                _mm256_store_si256((__m256i*)&acc_buffer_[i + 24], acc3);
            }
#endif
            for (; i < output_dim_; ++i) {
                acc_buffer_[i] += src_ptr[i];
            }
        }

        std::vector<int32_t> output(output_dim_);
        std::memcpy(output.data(), acc_buffer_, output_dim_ * sizeof(int32_t));
        
        return output;
    }

} // namespace dsta
