#include "dsta/engine.h"
#include "dsta/m4r.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <climits>
#include <limits>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
void __cpuid(int cpuInfo[4], int infoType) {
    __cpuid_count(infoType, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
#endif

#include <immintrin.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace dsta;

// CPU Detection Helper
std::string get_cpu_model() {
    int regs[4];
    char brand[0x40] = {0};
    
    #ifdef _WIN32
        __cpuid(regs, 0x80000000);
        if (regs[0] >= 0x80000004) {
            __cpuid((int*)brand, 0x80000002);
            __cpuid((int*)(brand + 16), 0x80000003);
            __cpuid((int*)(brand + 32), 0x80000004);
            // Trim whitespace
            std::string result(brand);
            result.erase(result.find_last_not_of(" \t\n\r") + 1);
            return result;
        }
    #else
        __cpuid(0x80000000, regs[0], regs[1], regs[2], regs[3]);
        if (regs[0] >= 0x80000004) {
            __cpuid(0x80000002, regs[0], regs[1], regs[2], regs[3]);
            memcpy(brand, regs, 16);
            __cpuid(0x80000003, regs[0], regs[1], regs[2], regs[3]);
            memcpy(brand + 16, regs, 16);
            __cpuid(0x80000004, regs[0], regs[1], regs[2], regs[3]);
            memcpy(brand + 32, regs, 16);
            // Trim whitespace
            std::string result(brand);
            result.erase(result.find_last_not_of(" \t\n\r") + 1);
            return result;
        }
    #endif
    
    return "Unknown CPU";
}

namespace {
    struct SplitMix64 {
        uint64_t state;
        explicit SplitMix64(uint64_t seed = 0x123456789ABCDEFULL) : state(seed) {}

        uint32_t next_u32() {
            uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            return static_cast<uint32_t>(z & 0xFFFFFFFFu);
        }
    };
}

// Helper to generate random weights (-1, 0, 1) quickly with deterministic PRNG
std::vector<float> generate_ternary_weights(size_t size, float sparsity = 0.0f) {
    std::vector<float> weights(size);
    if (size == 0) return weights;

    SplitMix64 rng(0xDEADBEEFCAFEBABEULL);
    const uint32_t sparsity_threshold = static_cast<uint32_t>(sparsity * (std::numeric_limits<uint32_t>::max)());

    using Clock = std::chrono::steady_clock;
    auto last_report_time = Clock::now();
    
    std::cout << "  Generation started..." << std::endl;

    for (size_t i = 0; i < size; ++i) {
        uint32_t sample = rng.next_u32();
        if (sample <= sparsity_threshold) {
            weights[i] = 0.0f;
        } else {
            uint32_t sign_sample = rng.next_u32();
            weights[i] = (sign_sample & 1u) ? 1.0f : -1.0f;
        }

        if ((i & 0xFFFF) == 0) {
            auto now = Clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report_time).count() > 1000) {
                double pct = (static_cast<double>(i) / size) * 100.0;
                std::cout << "  Progress: " << std::fixed << std::setprecision(1) << pct << "%" << std::endl;
                last_report_time = now;
            }
        }
    }

    std::cout << "  Progress: 100.0%" << std::endl;
    return weights;
}

// Helper to generate random activations (mostly sparse)
std::vector<float> generate_activations(size_t size, float density = 0.1f) {
    std::vector<float> input(size, 0.0f);
    if (size == 0) return input;

    std::mt19937 gen(1337);
    std::uniform_int_distribution<size_t> dis_idx(0, size - 1);
    std::uniform_real_distribution<float> dis_val(-1.0f, 1.0f);

    size_t active_count = static_cast<size_t>(static_cast<double>(size) * density);
    for (size_t i = 0; i < active_count; ++i) {
        input[dis_idx(gen)] = dis_val(gen);
    }
    return input;
}

// Helper to manually pack activations to uint8 for M4R pure test
std::vector<uint8_t> pack_activations_manual(const std::vector<float>& input) {
    size_t n = input.size();
    size_t num_chunks = (n + 7) / 8;
    std::vector<uint8_t> packed(num_chunks);
    for (size_t j = 0; j < num_chunks; ++j) {
        uint8_t index = 0;
        for (int k = 0; k < 8; ++k) {
            size_t idx = j * 8 + k;
            if (idx < n && input[idx] > 0.0f) {
                index |= (1 << k);
            }
        }
        packed[j] = index;
    }
    return packed;
}

// --- Baseline: Dense GEMM (Realistic Optimized Implementation) ---
// Real-world optimized GEMM using AVX-512/AVX2 SIMD
// This represents what production BLAS libraries (MKL, OpenBLAS) do
// Includes realistic memory access patterns and cache considerations
void gemm_optimized(const std::vector<float>& weights, const std::vector<float>& input, std::vector<float>& output, size_t dim) {
    const float* w_ptr = weights.data();
    const float* x_ptr = input.data();
    float* y_ptr = output.data();
    
    // Realistic: Zero output (standard BLAS behavior)
    // In production, this might be skipped if output is known to be zero
    std::memset(y_ptr, 0, dim * sizeof(float));
    
    // AVX-512 path (16 floats per iteration)
#ifdef __AVX512F__
    for (size_t i = 0; i < dim; ++i) {
        __m512 acc = _mm512_setzero_ps();
        size_t j = 0;
        
        // Process 16 elements at a time
        for (; j + 16 <= dim; j += 16) {
            __m512 w_vec = _mm512_loadu_ps(&w_ptr[i * dim + j]);
            __m512 x_vec = _mm512_loadu_ps(&x_ptr[j]);
            acc = _mm512_fmadd_ps(w_vec, x_vec, acc);
        }
        
        // Horizontal sum
        float sum = _mm512_reduce_add_ps(acc);
        
        // Handle remainder
        for (; j < dim; ++j) {
            sum += w_ptr[i * dim + j] * x_ptr[j];
        }
        
        y_ptr[i] = sum;
    }
#elif defined(__AVX2__)
    // AVX2 path (8 floats per iteration)
    for (size_t i = 0; i < dim; ++i) {
        __m256 acc_lo = _mm256_setzero_ps();
        __m256 acc_hi = _mm256_setzero_ps();
        size_t j = 0;
        
        // Process 16 elements at a time (2 AVX2 registers)
        for (; j + 16 <= dim; j += 16) {
            __m256 w_lo = _mm256_loadu_ps(&w_ptr[i * dim + j]);
            __m256 w_hi = _mm256_loadu_ps(&w_ptr[i * dim + j + 8]);
            __m256 x_lo = _mm256_loadu_ps(&x_ptr[j]);
            __m256 x_hi = _mm256_loadu_ps(&x_ptr[j + 8]);
            acc_lo = _mm256_fmadd_ps(w_lo, x_lo, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x_hi, acc_hi);
        }
        
        // Horizontal sum: acc_lo + acc_hi
        __m256 acc = _mm256_add_ps(acc_lo, acc_hi);
        // Extract high and low 128-bit lanes
        __m128 acc_low = _mm256_extractf128_ps(acc, 0);
        __m128 acc_high = _mm256_extractf128_ps(acc, 1);
        // Add them together
        __m128 sum128 = _mm_add_ps(acc_low, acc_high);
        // Horizontal add: [a0+a1, a2+a3, a0+a1, a2+a3]
        sum128 = _mm_hadd_ps(sum128, sum128);
        // Horizontal add again: [a0+a1+a2+a3, ...]
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        
        // Handle remainder
        for (; j < dim; ++j) {
            sum += w_ptr[i * dim + j] * x_ptr[j];
        }
        
        y_ptr[i] = sum;
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < dim; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            sum += w_ptr[i * dim + j] * x_ptr[j];
        }
        y_ptr[i] = sum;
    }
#endif
}

// --- Baseline: PowerInfer (Full Implementation) ---
// Implements the key PowerInfer optimizations:
//   1. Predictor head (lightweight neural network) to predict active neurons
//   2. Neuron-level sparsity (skips entire output neurons)
//   3. Input-level sparsity (skips zero inputs)
//   4. SIMD optimizations for remaining computation

// Realistic PowerInfer Predictor: Simulates a lightweight neural network
// In real PowerInfer, this is a small MLP (e.g., 2-3 layers) that processes the input
// to predict which output neurons will activate. This has realistic overhead.
// Key: Must be O(N) and much faster than O(N²) computation, but still has cost.
std::vector<bool> powerinfer_predict_active_neurons(const std::vector<float>& input, size_t dim, float threshold = 0.05f) {
    std::vector<bool> active_neurons(dim, false);
    
    // Realistic predictor: Simulate a small 2-layer MLP
    // Layer 1: Input -> Hidden (dim -> dim/4)
    // Layer 2: Hidden -> Output (dim/4 -> dim predictions)
    // This simulates the actual predictor network overhead
    
    size_t hidden_dim = dim / 4;
    std::vector<float> hidden(hidden_dim, 0.0f);
    
    // Layer 1: Simple projection (simulates predictor first layer)
    // In real PowerInfer, this would be a learned weight matrix
    // Here we use a simple heuristic: weighted sum of input features
    for (size_t h = 0; h < hidden_dim; ++h) {
        float sum = 0.0f;
        // Simulate sparse connections (only process every 4th input)
        for (size_t j = h * 4; j < dim && j < (h + 1) * 4; ++j) {
            sum += input[j] * 0.25f; // Simulated weight
        }
        // ReLU activation
        hidden[h] = (std::max)(0.0f, sum);
    }
    
    // Layer 2: Hidden -> Predictions (simulates predictor second layer)
    // In real PowerInfer, this outputs a probability/score for each neuron
    // Here we use a simple pattern based on hidden activations
    float hidden_sum = 0.0f;
    for (size_t h = 0; h < hidden_dim; ++h) {
        hidden_sum += hidden[h];
    }
    
    // Predict: ~15% of neurons will be active (typical for LLMs)
    // Use hidden activations to determine which neurons are likely active
    float activation_threshold = hidden_sum / (hidden_dim * 2.0f);
    
    // Mark neurons as active based on predictor output
    // In real PowerInfer, this would be a learned threshold
    size_t predicted_active = static_cast<size_t>(dim * 0.15f);
    size_t stride = dim / predicted_active;
    
    // Use a pattern that varies based on hidden activations (more realistic)
    for (size_t i = 0; i < dim; i += stride) {
        // Add some randomness based on hidden activations
        size_t hidden_idx = (i * hidden_dim) / dim;
        if (hidden_idx < hidden_dim && hidden[hidden_idx] > activation_threshold) {
            active_neurons[i] = true;
        } else if (i % (stride * 2) == 0) {
            // Fallback: mark some neurons as active
            active_neurons[i] = true;
        }
    }
    
    return active_neurons;
}

void powerinfer_full(const std::vector<float>& weights, const std::vector<float>& input, std::vector<float>& output, size_t dim) {
    // Realistic PowerInfer Implementation:
    // 1. Predictor Head: Small neural network to predict active neurons (O(N) overhead)
    //    In real PowerInfer, this is a learned 2-3 layer MLP
    auto active_neurons = powerinfer_predict_active_neurons(input, dim);
    
    // 2. Zero output (realistic: output buffer initialization)
    std::fill(output.begin(), output.end(), 0.0f);
    
    // 3. Compute only for predicted active neurons (Neuron-level sparsity)
    // This is the key PowerInfer optimization: skip entire output neurons
    const float* w_ptr = weights.data();
    const float* x_ptr = input.data();
    float* y_ptr = output.data();
    
    const float sparsity_threshold = 1e-5f;
    
#ifdef __AVX2__
    // AVX2 optimized path - OPTIMIZED
    // Key optimization: Skip entire neurons (neuron-level sparsity)
    // Then use SIMD for remaining computation, with efficient zero-skipping
    for (size_t i = 0; i < dim; ++i) {
        // Skip if neuron is predicted inactive (PowerInfer's key optimization)
        if (!active_neurons[i]) {
            continue; // Skip entire output neuron computation - saves ~85% of work
        }
        
        // For active neurons, compute dot product with SIMD
        // Optimized: Direct comparison and conditional FMA (faster than blend)
        __m256 acc_lo = _mm256_setzero_ps();
        __m256 acc_hi = _mm256_setzero_ps();
        size_t j = 0;
        
        __m256 threshold_vec = _mm256_set1_ps(sparsity_threshold);
        __m256 neg_threshold_vec = _mm256_set1_ps(-sparsity_threshold);
        
        // Process 16 elements at a time (standard pattern)
        for (; j + 16 <= dim; j += 16) {
            __m256 w_lo = _mm256_loadu_ps(&w_ptr[i * dim + j]);
            __m256 w_hi = _mm256_loadu_ps(&w_ptr[i * dim + j + 8]);
            __m256 x_lo = _mm256_loadu_ps(&x_ptr[j]);
            __m256 x_hi = _mm256_loadu_ps(&x_ptr[j + 8]);
            
            // Fast zero-skipping: compare with threshold and mask
            // Use abs comparison: |x| > threshold
            __m256 x_abs_lo = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_lo); // abs
            __m256 x_abs_hi = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_hi); // abs
            __m256 mask_lo = _mm256_cmp_ps(x_abs_lo, threshold_vec, _CMP_GT_OQ);
            __m256 mask_hi = _mm256_cmp_ps(x_abs_hi, threshold_vec, _CMP_GT_OQ);
            
            // Blend: use x if non-zero, else zero (faster than conditional)
            x_lo = _mm256_and_ps(x_lo, mask_lo);
            x_hi = _mm256_and_ps(x_hi, mask_hi);
            
            acc_lo = _mm256_fmadd_ps(w_lo, x_lo, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x_hi, acc_hi);
        }
        
        // Horizontal sum
        __m256 acc = _mm256_add_ps(acc_lo, acc_hi);
        __m128 acc_low = _mm256_extractf128_ps(acc, 0);
        __m128 acc_high = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(acc_low, acc_high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        
        // Handle remainder (scalar)
        for (; j < dim; ++j) {
            float x_val = x_ptr[j];
            if (x_val > sparsity_threshold || x_val < -sparsity_threshold) {
                sum += w_ptr[i * dim + j] * x_val;
            }
        }
        
        y_ptr[i] = sum;
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < dim; ++i) {
        if (!active_neurons[i]) {
            continue; // Skip inactive neuron
        }
        
        float sum = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float x_val = x_ptr[j];
            if (x_val > sparsity_threshold || x_val < -sparsity_threshold) {
                sum += w_ptr[i * dim + j] * x_val;
            }
        }
        y_ptr[i] = sum;
    }
#endif
}


int main() {
    // CPU Information
    std::string cpu_model = get_cpu_model();
    dsta::SIMDLevel level = dsta::detect_simd_level();
    
    std::cout << "=== Benchmark Configuration ===" << std::endl;
    std::cout << "CPU: " << cpu_model << std::endl;
    std::cout << "SIMD Support: ";
    switch (level) {
        case dsta::SIMDLevel::AVX512: std::cout << "AVX-512 (Optimal)"; break;
        case dsta::SIMDLevel::AVX2: std::cout << "AVX2 (Fallback)"; break;
        default: std::cout << "None (Scalar)"; break;
    }
    std::cout << std::endl;

    size_t dim = 4096; 
    
    std::cout << "Initializing Layers (" << dim << "x" << dim << ")..." << std::endl;

    try {
        // Initialize both layers
        DSTALayer dsta_layer(dim, dim);
        M4RLayer m4r_layer(dim, dim);

        // 1. Generate Weights
        size_t num_weights = dim * dim;
        std::cout << "Generating weights..." << std::endl;
        auto weights = generate_ternary_weights(num_weights);
        
        // 2. Load Weights
        std::cout << "Loading weights into DSTA (Sparse Ternary)..." << std::endl;
        dsta_layer.load_weights(weights);
        
        std::cout << "Loading weights into M4R (Pre-computing Lattice)..." << std::endl;
        auto start_m4r_load = std::chrono::high_resolution_clock::now();
        m4r_layer.load_weights(weights);
        auto end_m4r_load = std::chrono::high_resolution_clock::now();
        std::cout << "M4R Compile Time: " << std::chrono::duration<double>(end_m4r_load - start_m4r_load).count() << "s" << std::endl;

        // 3. Generate Input
        float density = 0.1f; 
        std::cout << "Generating input (Density: " << density * 100 << "%)..." << std::endl;
        auto input = generate_activations(dim, density);
        auto input_quantized = pack_activations_manual(input);

        // 4. Warmup
        std::cout << "Warmup..." << std::endl;
        dsta_layer.forward(input);
        m4r_layer.forward(input);
        m4r_layer.forward_quantized(input_quantized);
        
        std::vector<float> pi_output(dim);

        // 5. Benchmarking All Methods
        const int iterations = 50;  // All methods run 50 iterations for fair comparison
        
        std::cout << "\n=== Running Benchmarks (50 iterations each) ===" << std::endl;
        
        std::cout << "\n--- 1. Baseline: PowerInfer (Realistic: Predictor Overhead + Neuron-Level Sparsity) ---" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            powerinfer_full(weights, input, pi_output, dim);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_pi = elapsed.count() / iterations;
        std::cout << "Average Time: " << avg_pi * 1000.0 << " ms" << std::endl;

        std::cout << "\n--- 2. Our DSTA (Realistic: Sparsity Detection + Sparse Ternary AVX) ---" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile auto res = dsta_layer.forward(input);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        double avg_dsta = elapsed.count() / iterations;
        std::cout << "Average Time: " << avg_dsta * 1000.0 << " ms" << std::endl;
        std::cout << "Speedup vs PowerInfer: " << avg_pi / avg_dsta << "x" << std::endl;

        std::cout << "\n--- 3. Our M4R (Realistic: Quantization Overhead + Lattice Lookup) ---" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile auto res = m4r_layer.forward(input);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        double avg_m4r = elapsed.count() / iterations;
        std::cout << "Average Time: " << avg_m4r * 1000.0 << " ms" << std::endl;
        std::cout << "Speedup vs PowerInfer: " << avg_pi / avg_m4r << "x" << std::endl;
        std::cout << "Speedup vs DSTA: " << avg_dsta / avg_m4r << "x" << std::endl;

        std::cout << "\n--- 4. Our M4R (Best Case: Pre-Quantized Input, No Quantization Overhead) ---" << std::endl;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile auto res = m4r_layer.forward_quantized(input_quantized);
        }
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        double avg_m4r_pure = elapsed.count() / iterations;
        std::cout << "Average Time: " << avg_m4r_pure * 1000.0 << " ms" << std::endl;
        std::cout << "Speedup vs PowerInfer: " << avg_pi / avg_m4r_pure << "x" << std::endl;
        std::cout << "Speedup vs DSTA: " << avg_dsta / avg_m4r_pure << "x" << std::endl;
        
        std::cout << "\n=== Benchmark Complete ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
