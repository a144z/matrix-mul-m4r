#pragma once

#include "common.h"
#include <vector>

namespace dsta {

    // Packing utility class
    class Packer {
    public:
        // Compress standard float weights (-1.0, 0.0, 1.0) into 2-bit packed format.
        // Returns a byte vector where each byte holds 4 weights.
        static std::vector<uint8_t> pack_weights(const std::vector<float>& weights);

        // Debug utility to unpack for verification
        static std::vector<float> unpack_weights(const std::vector<uint8_t>& packed, size_t original_size);
    };

} // namespace dsta

