#pragma once

#include "common.h"
#include "packing.h"
#include "sparsity.h"
#include "kernel.h"

namespace dsta {

    class DSTALayer {
    public:
        DSTALayer(size_t in_dim, size_t out_dim);
        ~DSTALayer();

        // Initialize weights (takes float weights -1, 0, 1 and packs them)
        void load_weights(const std::vector<float>& weights);

        // Forward pass
        // Returns the output vector
        std::vector<float> forward(const std::vector<float>& input);

    private:
        size_t input_dim_;
        size_t output_dim_;
        
        // Memory managed pointers
        uint8_t* packed_weights_; // Column-major or Row-major optimized storage
        
        // Helper for memory management
        void allocate_memory();
        void free_memory();
    };

} // namespace dsta

