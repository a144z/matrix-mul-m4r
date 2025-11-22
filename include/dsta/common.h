#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>

#ifdef _WIN32
#include <malloc.h> // For _aligned_malloc/_aligned_free
#include <intrin.h> // For __cpuid
#else
#include <stdlib.h> // For posix_memalign
#include <cpuid.h>  // For __cpuid
#endif

// Include SIMD intrinsics based on what's available
#ifdef __AVX512F__
#include <immintrin.h> // AVX-512 intrinsics
#elif defined(__AVX2__)
#include <immintrin.h> // AVX2 intrinsics
#else
#include <immintrin.h> // Try anyway, will fail at runtime if not available
#endif

namespace dsta {

    // 2-bit ternary encoding
    // 00: 0
    // 01: +1
    // 10: -1
    // 11: Unused (treated as 0)
    enum TernaryValue : uint8_t {
        ZERO = 0,
        POS_ONE = 1,
        NEG_ONE = 2
    };

    // Constants
    constexpr int BLOCKS_PER_CHUNK = 64; // Processing chunk size
    
    // Aligned memory allocator for AVX-512 (64-byte alignment)
    template <typename T>
    struct AlignedAllocator {
        using value_type = T;
        
        T* allocate(std::size_t n) {
            void* ptr = nullptr;
            if (posix_memalign(&ptr, 64, n * sizeof(T)) != 0) {
                throw std::bad_alloc();
            }
            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, std::size_t) {
            free(p);
        }
    };
    
    // Helper for MSVC/Windows which might not have posix_memalign
    inline void* aligned_alloc_mem(size_t size, size_t alignment) {
        #ifdef _WIN32
            return _aligned_malloc(size, alignment);
        #else
            void* ptr = nullptr;
            posix_memalign(&ptr, alignment, size);
            return ptr;
        #endif
    }
    
    inline void aligned_free_mem(void* ptr) {
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    // CPU Feature Detection
    enum class SIMDLevel {
        NONE,
        AVX2,
        AVX512
    };

    inline SIMDLevel detect_simd_level() {
        int regs[4];
        int ids;
        
        #ifdef _WIN32
            __cpuid(regs, 0);
        #else
            __cpuid(0, regs[0], regs[1], regs[2], regs[3]);
        #endif
        ids = regs[0];
        
        if (ids < 7) return SIMDLevel::NONE;
        
        #ifdef _WIN32
            __cpuidex(regs, 7, 0);
        #else
            __cpuid_count(7, 0, regs[0], regs[1], regs[2], regs[3]);
        #endif
        
        // Check AVX512F, AVX512BW, AVX512VL
        bool avx512f = (regs[1] & (1 << 16)) != 0;
        bool avx512bw = (regs[1] & (1 << 30)) != 0;
        bool avx512vl = (regs[1] & (1U << 31)) != 0;
        
        if (avx512f && avx512bw && avx512vl) {
            return SIMDLevel::AVX512;
        }
        
        // Check AVX2 (bit 5 of EBX in CPUID leaf 7)
        bool avx2 = (regs[1] & (1 << 5)) != 0;
        
        // Also check BMI2 for PEXT (bit 8 of EBX in CPUID leaf 7)
        bool bmi2 = (regs[1] & (1 << 8)) != 0;
        
        if (avx2 && bmi2) {
            return SIMDLevel::AVX2;
        }
        
        return SIMDLevel::NONE;
    }

} // namespace dsta

