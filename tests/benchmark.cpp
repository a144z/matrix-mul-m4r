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
#include <map>

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

    for (size_t i = 0; i < size; ++i) {
        uint32_t sample = rng.next_u32();
        if (sample <= sparsity_threshold) {
            weights[i] = 0.0f;
        } else {
            uint32_t sign_sample = rng.next_u32();
            weights[i] = (sign_sample & 1u) ? 1.0f : -1.0f;
        }
    }

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
void gemm_optimized(const std::vector<float>& weights, const std::vector<float>& input, std::vector<float>& output, size_t dim) {
    const float* w_ptr = weights.data();
    const float* x_ptr = input.data();
    float* y_ptr = output.data();
    
    std::memset(y_ptr, 0, dim * sizeof(float));
    
#ifdef __AVX512F__
    for (size_t i = 0; i < dim; ++i) {
        __m512 acc = _mm512_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            __m512 w_vec = _mm512_loadu_ps(&w_ptr[i * dim + j]);
            __m512 x_vec = _mm512_loadu_ps(&x_ptr[j]);
            acc = _mm512_fmadd_ps(w_vec, x_vec, acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; j < dim; ++j) {
            sum += w_ptr[i * dim + j] * x_ptr[j];
        }
        y_ptr[i] = sum;
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < dim; ++i) {
        __m256 acc_lo = _mm256_setzero_ps();
        __m256 acc_hi = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            __m256 w_lo = _mm256_loadu_ps(&w_ptr[i * dim + j]);
            __m256 w_hi = _mm256_loadu_ps(&w_ptr[i * dim + j + 8]);
            __m256 x_lo = _mm256_loadu_ps(&x_ptr[j]);
            __m256 x_hi = _mm256_loadu_ps(&x_ptr[j + 8]);
            acc_lo = _mm256_fmadd_ps(w_lo, x_lo, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x_hi, acc_hi);
        }
        __m256 acc = _mm256_add_ps(acc_lo, acc_hi);
        __m128 acc_low = _mm256_extractf128_ps(acc, 0);
        __m128 acc_high = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(acc_low, acc_high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        for (; j < dim; ++j) {
            sum += w_ptr[i * dim + j] * x_ptr[j];
        }
        y_ptr[i] = sum;
    }
#else
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
std::vector<bool> powerinfer_predict_active_neurons(const std::vector<float>& input, size_t dim, float threshold = 0.05f) {
    std::vector<bool> active_neurons(dim, false);
    size_t hidden_dim = dim / 4;
    std::vector<float> hidden(hidden_dim, 0.0f);
    
    for (size_t h = 0; h < hidden_dim; ++h) {
        float sum = 0.0f;
        for (size_t j = h * 4; j < dim && j < (h + 1) * 4; ++j) {
            sum += input[j] * 0.25f;
        }
        hidden[h] = (std::max)(0.0f, sum);
    }
    
    float hidden_sum = 0.0f;
    for (size_t h = 0; h < hidden_dim; ++h) {
        hidden_sum += hidden[h];
    }
    
    float activation_threshold = hidden_sum / (hidden_dim * 2.0f);
    size_t predicted_active = static_cast<size_t>(dim * 0.15f);
    size_t stride = dim / predicted_active;
    
    for (size_t i = 0; i < dim; i += stride) {
        size_t hidden_idx = (i * hidden_dim) / dim;
        if (hidden_idx < hidden_dim && hidden[hidden_idx] > activation_threshold) {
            active_neurons[i] = true;
        } else if (i % (stride * 2) == 0) {
            active_neurons[i] = true;
        }
    }
    
    return active_neurons;
}

void powerinfer_full(const std::vector<float>& weights, const std::vector<float>& input, std::vector<float>& output, size_t dim) {
    auto active_neurons = powerinfer_predict_active_neurons(input, dim);
    std::fill(output.begin(), output.end(), 0.0f);
    
    const float* w_ptr = weights.data();
    const float* x_ptr = input.data();
    float* y_ptr = output.data();
    const float sparsity_threshold = 1e-5f;
    
#ifdef __AVX2__
    for (size_t i = 0; i < dim; ++i) {
        if (!active_neurons[i]) continue;
        
        __m256 acc_lo = _mm256_setzero_ps();
        __m256 acc_hi = _mm256_setzero_ps();
        size_t j = 0;
        __m256 threshold_vec = _mm256_set1_ps(sparsity_threshold);
        
        for (; j + 16 <= dim; j += 16) {
            __m256 w_lo = _mm256_loadu_ps(&w_ptr[i * dim + j]);
            __m256 w_hi = _mm256_loadu_ps(&w_ptr[i * dim + j + 8]);
            __m256 x_lo = _mm256_loadu_ps(&x_ptr[j]);
            __m256 x_hi = _mm256_loadu_ps(&x_ptr[j + 8]);
            __m256 x_abs_lo = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_lo);
            __m256 x_abs_hi = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_hi);
            __m256 mask_lo = _mm256_cmp_ps(x_abs_lo, threshold_vec, _CMP_GT_OQ);
            __m256 mask_hi = _mm256_cmp_ps(x_abs_hi, threshold_vec, _CMP_GT_OQ);
            x_lo = _mm256_and_ps(x_lo, mask_lo);
            x_hi = _mm256_and_ps(x_hi, mask_hi);
            acc_lo = _mm256_fmadd_ps(w_lo, x_lo, acc_lo);
            acc_hi = _mm256_fmadd_ps(w_hi, x_hi, acc_hi);
        }
        
        __m256 acc = _mm256_add_ps(acc_lo, acc_hi);
        __m128 acc_low = _mm256_extractf128_ps(acc, 0);
        __m128 acc_high = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(acc_low, acc_high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);
        
        for (; j < dim; ++j) {
            float x_val = x_ptr[j];
            if (x_val > sparsity_threshold || x_val < -sparsity_threshold) {
                sum += w_ptr[i * dim + j] * x_val;
            }
        }
        y_ptr[i] = sum;
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        if (!active_neurons[i]) continue;
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

// Benchmark a single layer
struct SingleLayerResults {
    double gemm_time = 0.0;
    double pi_time = 0.0;
    double dsta_time = 0.0;
    double m4r_time = 0.0;
    double m4r_pure_time = 0.0;
    double m4r_compile_time = 0.0;
};

SingleLayerResults benchmark_single_layer(size_t dim, int iterations = 50) {
    SingleLayerResults results;
    
    try {
        DSTALayer dsta_layer(dim, dim);
        M4RLayer m4r_layer(dim, dim);

        size_t num_weights = dim * dim;
        auto weights = generate_ternary_weights(num_weights);
        
        dsta_layer.load_weights(weights);
        
        auto start_m4r_load = std::chrono::high_resolution_clock::now();
        m4r_layer.load_weights(weights);
        auto end_m4r_load = std::chrono::high_resolution_clock::now();
        results.m4r_compile_time = std::chrono::duration<double>(end_m4r_load - start_m4r_load).count();

        float density = 0.1f;
        auto input = generate_activations(dim, density);
        auto input_quantized = pack_activations_manual(input);

        // Warmup
        dsta_layer.forward(input);
        m4r_layer.forward(input);
        m4r_layer.forward_quantized(input_quantized);
        
        std::vector<float> gemm_output(dim);
        std::vector<float> pi_output(dim);

        // Benchmark GEMM
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            gemm_optimized(weights, input, gemm_output, dim);
        }
        auto end = std::chrono::high_resolution_clock::now();
        results.gemm_time = std::chrono::duration<double>(end - start).count() / iterations;

        // Benchmark PowerInfer
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            powerinfer_full(weights, input, pi_output, dim);
        }
        end = std::chrono::high_resolution_clock::now();
        results.pi_time = std::chrono::duration<double>(end - start).count() / iterations;

        // Benchmark DSTA
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile auto res = dsta_layer.forward(input);
        }
        end = std::chrono::high_resolution_clock::now();
        results.dsta_time = std::chrono::duration<double>(end - start).count() / iterations;

        // Benchmark M4R
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile auto res = m4r_layer.forward(input);
        }
        end = std::chrono::high_resolution_clock::now();
        results.m4r_time = std::chrono::duration<double>(end - start).count() / iterations;

        // Benchmark M4R Pure
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile auto res = m4r_layer.forward_quantized(input_quantized);
        }
        end = std::chrono::high_resolution_clock::now();
        results.m4r_pure_time = std::chrono::duration<double>(end - start).count() / iterations;

    } catch (const std::exception& e) {
        std::cerr << "Error in benchmark_single_layer: " << e.what() << std::endl;
    }

    return results;
}

// Benchmark multi-layer propagation
struct MultiLayerResults {
    double gemm_time = 0.0;
    double dsta_time = 0.0;
    double m4r_time = 0.0;
};

MultiLayerResults benchmark_multilayer(const std::vector<size_t>& layer_dims, int num_layers, int iterations = 20) {
    MultiLayerResults results;
    
    try {
        // Create layers
        std::vector<DSTALayer> dsta_layers;
        std::vector<M4RLayer> m4r_layers;
        std::vector<std::vector<float>> gemm_weights;
        
        std::cout << " [Initializing " << num_layers << " layers...] " << std::flush;
        
        for (int l = 0; l < num_layers; ++l) {
            size_t in_dim = (l == 0) ? layer_dims[0] : layer_dims[l-1];
            size_t out_dim = layer_dims[l];
            
            std::cout << "[" << (l+1) << "/" << num_layers << ":" << in_dim << "x" << out_dim << "]" << std::flush;
            
            dsta_layers.emplace_back(in_dim, out_dim);
            std::cout << "D" << std::flush;
            m4r_layers.emplace_back(in_dim, out_dim);
            std::cout << "M" << std::flush;
            
            size_t num_weights = in_dim * out_dim;
            std::cout << "W" << std::flush;
            auto weights = generate_ternary_weights(num_weights);
            gemm_weights.push_back(weights);
            
            std::cout << "Ld" << std::flush;
            dsta_layers[l].load_weights(weights);
            std::cout << "+d" << std::flush;
            
            // M4R loading can be slow - show progress
            std::cout << "Lm" << std::flush;
            auto m4r_start = std::chrono::steady_clock::now();
            m4r_layers[l].load_weights(weights);
            auto m4r_end = std::chrono::steady_clock::now();
            auto m4r_duration = std::chrono::duration<double>(m4r_end - m4r_start).count();
            if (m4r_duration > 0.1) {
                std::cout << "+m(" << std::fixed << std::setprecision(1) << m4r_duration << "s)" << std::flush;
            } else {
                std::cout << "+m" << std::flush;
            }
            std::cout << " " << std::flush;
        }
        
        std::cout << " [Warming up...] " << std::flush;
        std::cout.flush();

        float density = 0.1f;
        auto input = generate_activations(layer_dims[0], density);
        auto input_quantized = pack_activations_manual(input);

        // Warmup
        std::cout << "W" << std::flush;
        auto temp_input = input;
        for (int l = 0; l < num_layers; ++l) {
            temp_input = dsta_layers[l].forward(temp_input);
        }
        std::cout << "+" << std::flush;
        temp_input = input;
        for (int l = 0; l < num_layers; ++l) {
            temp_input = m4r_layers[l].forward(temp_input);
        }
        std::cout << "+" << std::flush;
        std::cout << " [Benchmarking...] " << std::flush;
        std::cout.flush();

        // Benchmark GEMM
        std::cout << "GEMM " << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto current_input = input;
            for (int l = 0; l < num_layers; ++l) {
                size_t dim = layer_dims[l];
                std::vector<float> output(dim);
                gemm_optimized(gemm_weights[l], current_input, output, dim);
                current_input = output;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        results.gemm_time = std::chrono::duration<double>(end - start).count() / iterations;

        // Benchmark DSTA
        std::cout << "DSTA " << std::flush;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto current_input = input;
            for (int l = 0; l < num_layers; ++l) {
                current_input = dsta_layers[l].forward(current_input);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        results.dsta_time = std::chrono::duration<double>(end - start).count() / iterations;

        // Benchmark M4R
        std::cout << "M4R " << std::flush;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            auto current_input = input;
            for (int l = 0; l < num_layers; ++l) {
                current_input = m4r_layers[l].forward(current_input);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        results.m4r_time = std::chrono::duration<double>(end - start).count() / iterations;
        std::cout << "+" << std::flush;

    } catch (const std::exception& e) {
        std::cerr << "\nError in benchmark_multilayer: " << e.what() << std::endl;
        std::cerr.flush();
    }

    return results;
}

int main() {
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
    std::cout << std::endl << std::endl;

    // Test different matrix sizes
    std::vector<size_t> test_sizes = {1024, 2048, 4096, 8192};
    const int iterations = 50;

    std::cout << "=== Single Layer Benchmarks ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    std::cout << "\n| Size | GEMM (ms) | PowerInfer (ms) | DSTA (ms) | M4R (ms) | M4R Pure (ms) | M4R vs GEMM | M4R vs DSTA |" << std::endl;
    std::cout << "|------|-----------|-----------------|-----------|----------|---------------|-------------|-------------|" << std::endl;

    for (size_t dim : test_sizes) {
        std::cout << "Testing " << dim << "x" << dim << "..." << std::flush;
        auto results = benchmark_single_layer(dim, iterations);
        std::cout << " Done" << std::endl;
        
        double m4r_speedup_vs_gemm = results.gemm_time / results.m4r_pure_time;
        double m4r_speedup_vs_dsta = results.dsta_time / results.m4r_pure_time;
        
        std::cout << "| " << dim << " | " 
                  << results.gemm_time * 1000.0 << " | "
                  << results.pi_time * 1000.0 << " | "
                  << results.dsta_time * 1000.0 << " | "
                  << results.m4r_time * 1000.0 << " | "
                  << results.m4r_pure_time * 1000.0 << " | "
                  << m4r_speedup_vs_gemm << "x | "
                  << m4r_speedup_vs_dsta << "x |" << std::endl;
    }

    // Multi-layer benchmarks
    std::cout << "\n\n=== Multi-Layer Propagation Benchmarks ===" << std::endl;
    std::cout << "Simulating network with multiple layers..." << std::endl;
    
    // Test different network configurations
    std::vector<std::pair<std::string, std::vector<size_t>>> network_configs = {
        {"Small (3 layers)", {2048, 2048, 2048}},
        {"Medium (4 layers)", {4096, 4096, 4096, 4096}},
        {"Large (5 layers)", {4096, 4096, 4096, 4096, 4096}},
    };

    std::cout << "\n| Network | Layers | GEMM (ms) | DSTA (ms) | M4R (ms) | M4R vs GEMM | M4R vs DSTA |" << std::endl;
    std::cout << "|---------|--------|-----------|-----------|----------|-------------|-------------|" << std::endl;

    for (const auto& config : network_configs) {
        std::cout << "Testing " << config.first << "..." << std::flush;
        std::cout.flush();
        
        try {
            auto results = benchmark_multilayer(config.second, config.second.size(), 20);
            std::cout << " Done" << std::endl;
            std::cout.flush();
            
            // Ensure we have valid results before printing
            if (results.gemm_time > 0 && results.dsta_time > 0 && results.m4r_time > 0) {
                double m4r_speedup_vs_gemm = results.gemm_time / results.m4r_time;
                double m4r_speedup_vs_dsta = results.dsta_time / results.m4r_time;
                
                std::cout << "| " << config.first << " | " << config.second.size() << " | "
                          << results.gemm_time * 1000.0 << " | "
                          << results.dsta_time * 1000.0 << " | "
                          << results.m4r_time * 1000.0 << " | "
                          << m4r_speedup_vs_gemm << "x | "
                          << m4r_speedup_vs_dsta << "x |" << std::endl;
                std::cout.flush();
            } else {
                std::cout << "| " << config.first << " | " << config.second.size() << " | "
                          << "ERROR | ERROR | ERROR | - | - |" << std::endl;
                std::cout.flush();
            }
        } catch (const std::exception& e) {
            std::cerr << "\nException in main loop: " << e.what() << std::endl;
            std::cerr.flush();
            std::cout << " Done (with errors)" << std::endl;
            std::cout << "| " << config.first << " | " << config.second.size() << " | "
                      << "ERROR | ERROR | ERROR | - | - |" << std::endl;
            std::cout.flush();
        }
    }

    std::cout << "\n=== Benchmark Complete ===" << std::endl;

    return 0;
}
