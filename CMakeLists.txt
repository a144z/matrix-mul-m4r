cmake_minimum_required(VERSION 3.15)
project(dsta_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
# Use AVX2 as baseline (widely supported), runtime will detect and use AVX-512 if available
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -mavx2 -mbmi2 -mfma)
endif()

include_directories(include)

# Library Source
add_library(dsta_lib SHARED
    src/packing.cpp
    src/sparsity.cpp
    src/kernel.cpp
    src/engine.cpp
    src/m4r.cpp
)

# Windows-specific: Export all symbols for DLL
if(WIN32)
    set_target_properties(dsta_lib PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# For MSVC: Set output directories to match multi-config generator expectations
if(MSVC)
    # MSVC multi-config puts outputs in ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/
    # This ensures the .lib file is in the same directory as the .dll
    set_target_properties(dsta_lib PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo"
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/MinSizeRel"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/MinSizeRel"
    )
endif()

# Test/Benchmark Executable
add_executable(dsta_bench tests/benchmark.cpp)
target_link_libraries(dsta_bench PRIVATE dsta_lib)

# For MSVC multi-config generators, ensure proper linking
if(MSVC)
    # Ensure the library is built before the executable
    add_dependencies(dsta_bench dsta_lib)
    # Set executable output to match library location
    set_target_properties(dsta_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/MinSizeRel"
    )
endif()

