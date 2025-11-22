#pragma once

#include "common.h"

namespace dsta {

    // The core DSTA Kernel (Input-Stationary: processes all outputs per active input)
    // Computes: Output += Weights * Input (Sparse Ternary)
    // Uses dynamic input sparsity (bitmask) to skip zero inputs
    void dsta_kernel_forward(
        float* output,
        const float* input,
        const uint8_t* packed_weights,
        const uint64_t* mask,
        size_t input_dim,
        size_t output_dim
    );

} // namespace dsta

