#include "dsta/packing.h"
#include <cmath>
#include <stdexcept>

namespace dsta {

    std::vector<uint8_t> Packer::pack_weights(const std::vector<float>& weights) {
        size_t num_weights = weights.size();
        // 4 weights per byte (2 bits each)
        size_t packed_size = (num_weights + 3) / 4;
        std::vector<uint8_t> packed(packed_size, 0);

        for (size_t i = 0; i < num_weights; ++i) {
            uint8_t val = 0; // 00 = 0
            float w = weights[i];
            
            if (w > 0.5f) val = TernaryValue::POS_ONE;       // 01 = +1
            else if (w < -0.5f) val = TernaryValue::NEG_ONE; // 10 = -1
            else val = TernaryValue::ZERO;                   // 00 = 0

            // Position in byte: 0, 1, 2, 3
            // 0 -> bits 0-1
            // 1 -> bits 2-3
            // 2 -> bits 4-5
            // 3 -> bits 6-7
            size_t byte_idx = i / 4;
            size_t shift = (i % 4) * 2;

            packed[byte_idx] |= (val << shift);
        }

        return packed;
    }

    std::vector<float> Packer::unpack_weights(const std::vector<uint8_t>& packed, size_t original_size) {
        std::vector<float> weights(original_size);

        for (size_t i = 0; i < original_size; ++i) {
            size_t byte_idx = i / 4;
            size_t shift = (i % 4) * 2;

            uint8_t val = (packed[byte_idx] >> shift) & 0x03;

            if (val == TernaryValue::POS_ONE) weights[i] = 1.0f;
            else if (val == TernaryValue::NEG_ONE) weights[i] = -1.0f;
            else weights[i] = 0.0f;
        }

        return weights;
    }

} // namespace dsta

