#include "dsta/engine.h"
#include "dsta/common.h"
#include <cstring>
#include <stdexcept>

namespace dsta {

    DSTALayer::DSTALayer(size_t in_dim, size_t out_dim) 
        : input_dim_(in_dim), output_dim_(out_dim), packed_weights_(nullptr) {
        
        // Validation: output_dim must be multiple of 8 (for AVX2) or 16 (for AVX-512)
        // Since 16 is multiple of 8, we just check for 8
        if (output_dim_ % 8 != 0) {
            throw std::invalid_argument("Output dimension must be a multiple of 8 for SIMD kernels.");
        }

        allocate_memory();
    }

    DSTALayer::~DSTALayer() {
        free_memory();
    }

    void DSTALayer::allocate_memory() {
        // Size calculation:
        // Total weights = input_dim * output_dim
        // Packed size = Total weights / 4 (2 bits per weight)
        size_t total_weights = input_dim_ * output_dim_;
        size_t packed_size = (total_weights + 3) / 4;

        // Align allocation to 64 bytes
        packed_weights_ = static_cast<uint8_t*>(aligned_alloc_mem(packed_size, 64));
        if (!packed_weights_) {
            throw std::bad_alloc();
        }
        
        // Zero init
        std::memset(packed_weights_, 0, packed_size);
    }

    void DSTALayer::free_memory() {
        if (packed_weights_) {
            aligned_free_mem(packed_weights_);
            packed_weights_ = nullptr;
        }
    }

    void DSTALayer::load_weights(const std::vector<float>& weights) {
        if (weights.size() != input_dim_ * output_dim_) {
            throw std::invalid_argument("Weight size mismatch.");
        }

        // DSTA Kernel expects Input-Stationary layout (Column-Major).
        // Standard input weights are Row-Major [Output][Input].
        // We must transpose the weights to [Input][Output] before packing.
        
        std::vector<float> weights_col_major(weights.size());
        
        // Transpose: W_row[i][j] -> W_col[j][i]
        // i = output_idx, j = input_idx
        for (size_t i = 0; i < output_dim_; ++i) {
            for (size_t j = 0; j < input_dim_; ++j) {
                weights_col_major[j * output_dim_ + i] = weights[i * input_dim_ + j];
            }
        }

        std::vector<uint8_t> packed = Packer::pack_weights(weights_col_major);
        
        // Copy to aligned memory
        size_t size = packed.size();
        std::memcpy(packed_weights_, packed.data(), size);
    }

    std::vector<float> DSTALayer::forward(const std::vector<float>& input) {
        if (input.size() != input_dim_) {
            throw std::invalid_argument("Input dimension mismatch.");
        }

        // Realistic DSTA Forward Pass (Input-Stationary):
        // 1. Generate Sparsity Mask (realistic overhead)
        std::vector<uint64_t> mask = SparsityPredictor::generate_mask(input);

        // 2. Prepare Output Buffer (Aligned)
        float* output_ptr = static_cast<float*>(aligned_alloc_mem(output_dim_ * sizeof(float), 64));
        std::memset(output_ptr, 0, output_dim_ * sizeof(float));

        // 3. Run Kernel (core computation)
        dsta_kernel_forward(
            output_ptr,
            input.data(),
            packed_weights_,
            mask.data(),
            input_dim_,
            output_dim_
        );

        // 4. Return Result
        std::vector<float> result(output_ptr, output_ptr + output_dim_);
        aligned_free_mem(output_ptr);

        return result;
    }

} // namespace dsta

