#pragma once

#include "common.h"
#include <vector>
#include <cstdint>

namespace dsta {

    class M4RLayer {
    public:
        // Constructor: Dimension must be multiple of 8 for the lattice optimization
        M4RLayer(size_t in_dim, size_t out_dim);
        ~M4RLayer();

        // Initialize weights: Takes float weights {-1, 0, 1}
        // Pre-computes the "Russian Lattice" tables
        void load_weights(const std::vector<float>& weights);

        // Forward pass with floating point input (will be quantized on the fly)
        // This is for compatibility with the benchmark
        std::vector<float> forward(const std::vector<float>& input);

        // Forward pass with pre-quantized binary input (0/1 stored in uint8_t)
        // This is the "pure" M4R path
        std::vector<int32_t> forward_quantized(const std::vector<uint8_t>& input);

    private:
        size_t input_dim_;
        size_t output_dim_;
        
        // The "Russian Lattice" Table
        // Memory Layout: [Chunk][Pattern][Row] (Column-Major optimized)
        // Size: (input_dim / 8) * 256 * output_dim
        // This layout ensures contiguous vector access for each chunk accumulation.
        int16_t* lattice_table_;

        // Pre-allocated buffers for inference
        int32_t* acc_buffer_;
        std::vector<uint8_t> packed_indices_buffer_;

        void allocate_memory();
        void free_memory();
    };

} // namespace dsta

