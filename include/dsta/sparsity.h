#pragma once

#include "common.h"
#include <vector>

namespace dsta {

    // Sparsity predictor engine
    class SparsityPredictor {
    public:
        // Given an input activation vector, generate a bitmask of "active" blocks.
        // Returns a uint64_t array where each bit represents a block of neurons.
        // For simple simulation, we threshold the values.
        // In a real PowerInfer setup, this would run a small MLP predictor.
        static std::vector<uint64_t> generate_mask(const std::vector<float>& input, float threshold = 0.001f);
        
        // Helper to count active blocks
        static size_t count_active_blocks(const std::vector<uint64_t>& mask);
    };

} // namespace dsta

