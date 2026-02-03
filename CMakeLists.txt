cmake_minimum_required(VERSION 3.26)
project(lite_attention LANGUAGES CXX CUDA)

# Require CUDA 12.8+
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
    message(FATAL_ERROR "CUDA 12.8+ required for optimal performance. Found: ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Use C++20 for CUDA (set before find_package to avoid redefinition warning)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)

# Use ccache for faster rebuilds (if available)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Feature flags (replacing env vars from setup.py)
option(FLASH_ATTENTION_DISABLE_BACKWARD "Disable backward pass" ON)
option(FLASH_ATTENTION_DISABLE_SPLIT "Disable split-KV" ON)
option(FLASH_ATTENTION_DISABLE_PAGEDKV "Disable paged KV-cache" ON)
option(FLASH_ATTENTION_DISABLE_SOFTCAP "Disable softcap" ON)
option(FLASH_ATTENTION_DISABLE_PACKGQA "Disable packed GQA" ON)
option(FLASH_ATTENTION_DISABLE_FP16 "Disable FP16 (BF16 only)" ON)
option(FLASH_ATTENTION_DISABLE_FP8 "Disable FP8" ON)
option(FLASH_ATTENTION_DISABLE_INT8 "Disable INT8" OFF)
option(FLASH_ATTENTION_DISABLE_SM80 "Disable SM80 (Ampere)" ON)
option(FLASH_ATTENTION_DISABLE_VARLEN "Disable variable length" ON)
option(FLASH_ATTENTION_DISABLE_CLUSTER "Disable cluster" ON)
option(FLASH_ATTENTION_DISABLE_HDIM64 "Disable head dim 64" OFF)
option(FLASH_ATTENTION_DISABLE_HDIM96 "Disable head dim 96" OFF)
option(FLASH_ATTENTION_DISABLE_HDIM128 "Disable head dim 128" OFF)
option(FLASH_ATTENTION_DISABLE_HDIM192 "Disable head dim 192" OFF)
option(FLASH_ATTENTION_DISABLE_HDIM256 "Disable head dim 256" OFF)

# Package source directory
set(PKG_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lite_attention")

# Generate instantiation files at configure time
set(INSTANTIATION_DIR "${CMAKE_BINARY_DIR}/instantiations")
file(MAKE_DIRECTORY ${INSTANTIATION_DIR})

message(STATUS "Generating kernel instantiations...")
execute_process(
    COMMAND ${Python_EXECUTABLE} ${PKG_SRC_DIR}/utils/generate_kernels.py
        -o ${INSTANTIATION_DIR}
    WORKING_DIRECTORY ${PKG_SRC_DIR}
    RESULT_VARIABLE GEN_RESULT
    OUTPUT_VARIABLE GEN_OUTPUT
    ERROR_VARIABLE GEN_ERROR
)
if(NOT GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate kernel instantiations:\n${GEN_ERROR}")
endif()

# Glob all generated sources
file(GLOB ALL_FWD_SM90 "${INSTANTIATION_DIR}/flash_fwd_*_sm90.cu")
file(GLOB ALL_FWD_SM80 "${INSTANTIATION_DIR}/flash_fwd_*_sm80.cu")
file(GLOB ALL_FWD_SM100 "${INSTANTIATION_DIR}/flash_fwd_*_sm100.cu")
file(GLOB ALL_BWD_SM90 "${INSTANTIATION_DIR}/flash_bwd_*_sm90.cu")
file(GLOB ALL_BWD_SM80 "${INSTANTIATION_DIR}/flash_bwd_*_sm80.cu")

# Filter sources based on feature flags
set(FWD_SM90_SOURCES ${ALL_FWD_SM90})
set(FWD_SM80_SOURCES "")
set(FWD_SM100_SOURCES ${ALL_FWD_SM100})
set(BWD_SM90_SOURCES "")
set(BWD_SM80_SOURCES "")

# Include SM80 sources if not disabled
if(NOT FLASH_ATTENTION_DISABLE_SM80)
    set(FWD_SM80_SOURCES ${ALL_FWD_SM80})
endif()

# Include backward sources if not disabled
if(NOT FLASH_ATTENTION_DISABLE_BACKWARD)
    set(BWD_SM90_SOURCES ${ALL_BWD_SM90})
    if(NOT FLASH_ATTENTION_DISABLE_SM80)
        set(BWD_SM80_SOURCES ${ALL_BWD_SM80})
    endif()
endif()

# Filter by dtype
if(FLASH_ATTENTION_DISABLE_FP16)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "_fp16_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "_fp16_")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "_fp16_")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "_fp16_")
endif()

if(FLASH_ATTENTION_DISABLE_FP8)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "_e4m3_")
endif()

# Filter by features
if(FLASH_ATTENTION_DISABLE_SPLIT)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "_split_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "_split_")
endif()

if(FLASH_ATTENTION_DISABLE_PAGEDKV)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "_paged_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "_paged_")
endif()

if(FLASH_ATTENTION_DISABLE_SOFTCAP)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "_softcap")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "_softcap")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "_softcap")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "_softcap")
endif()

if(FLASH_ATTENTION_DISABLE_PACKGQA)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "_packgqa_")
endif()

# Filter out batch files (hdimall, hdimdiff) - they include other files and cause double compilation
list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdimall_")
list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdimdiff_")

# Note: hdim64_256/512 and hdim192_128 variants are required by flash_api.cpp
# even if not used at runtime (linker needs symbols). Cannot disable without C++ changes.

# Filter by head dimensions
if(FLASH_ATTENTION_DISABLE_HDIM64)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdim64_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "hdim64_")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "hdim64_")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "hdim64_")
endif()

if(FLASH_ATTENTION_DISABLE_HDIM96)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdim96_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "hdim96_")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "hdim96_")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "hdim96_")
endif()

if(FLASH_ATTENTION_DISABLE_HDIM128)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdim128_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "hdim128_")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "hdim128_")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "hdim128_")
endif()

if(FLASH_ATTENTION_DISABLE_HDIM192)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdim192_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "hdim192_")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "hdim192_")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "hdim192_")
endif()

if(FLASH_ATTENTION_DISABLE_HDIM256)
    list(FILTER FWD_SM90_SOURCES EXCLUDE REGEX "hdim256_")
    list(FILTER FWD_SM80_SOURCES EXCLUDE REGEX "hdim256_")
    list(FILTER BWD_SM90_SOURCES EXCLUDE REGEX "hdim256_")
    list(FILTER BWD_SM80_SOURCES EXCLUDE REGEX "hdim256_")
endif()

# Set per-file architecture flags (no more Ninja monkey-patching!)
set_source_files_properties(${FWD_SM90_SOURCES} ${BWD_SM90_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-gencode;arch=compute_90a,code=sm_90a"
)
set_source_files_properties(${FWD_SM80_SOURCES} ${BWD_SM80_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-gencode;arch=compute_80,code=sm_80"
)
set_source_files_properties(${FWD_SM100_SOURCES} PROPERTIES
    COMPILE_OPTIONS "-gencode;arch=compute_100a,code=sm_100a"
)

# Collect all sources
set(SOURCES
    ${PKG_SRC_DIR}/_internal/cpp/flash_api.cpp
    ${PKG_SRC_DIR}/_internal/cpp/flash_prepare_scheduler.cu
)

# Add optional sources
if(NOT FLASH_ATTENTION_DISABLE_SPLIT)
    list(APPEND SOURCES ${PKG_SRC_DIR}/_internal/cpp/flash_fwd_combine.cu)
endif()

if(NOT FLASH_ATTENTION_DISABLE_INT8)
    list(APPEND SOURCES ${PKG_SRC_DIR}/_internal/cpp/quant.cu)
endif()

# Add instantiation sources
list(APPEND SOURCES ${FWD_SM90_SOURCES})
list(APPEND SOURCES ${FWD_SM80_SOURCES})
list(APPEND SOURCES ${FWD_SM100_SOURCES})
list(APPEND SOURCES ${BWD_SM90_SOURCES})
list(APPEND SOURCES ${BWD_SM80_SOURCES})

# Report what we're building
list(LENGTH FWD_SM90_SOURCES FWD_SM90_COUNT)
list(LENGTH FWD_SM80_SOURCES FWD_SM80_COUNT)
list(LENGTH BWD_SM90_SOURCES BWD_SM90_COUNT)
list(LENGTH BWD_SM80_SOURCES BWD_SM80_COUNT)
message(STATUS "Building: ${FWD_SM90_COUNT} fwd_sm90, ${FWD_SM80_COUNT} fwd_sm80, ${BWD_SM90_COUNT} bwd_sm90, ${BWD_SM80_COUNT} bwd_sm80 kernels")

# Build the Python extension
python_add_library(_C MODULE ${SOURCES} WITH_SOABI)

target_include_directories(_C PRIVATE
    ${PKG_SRC_DIR}
    ${PKG_SRC_DIR}/_internal/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

# C++ compile options
target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
)

# CUDA compile options
# --threads enables intra-file parallelism (nvcc uses multiple CPU threads per file)
set(NVCC_THREADS "$ENV{NVCC_THREADS}")
if(NOT NVCC_THREADS)
    set(NVCC_THREADS 2)
endif()
target_compile_options(_C PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --threads=${NVCC_THREADS}
        --ftemplate-backtrace-limit=0
        --use_fast_math
        -lineinfo
        -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
        -DCUTLASS_DEBUG_TRACE_LEVEL=0
        -DNDEBUG
    >
)

# Feature compile definitions
target_compile_definitions(_C PRIVATE
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_BACKWARD}>:FLASHATTENTION_DISABLE_BACKWARD>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_SPLIT}>:FLASHATTENTION_DISABLE_SPLIT>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_PAGEDKV}>:FLASHATTENTION_DISABLE_PAGEDKV>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_SOFTCAP}>:FLASHATTENTION_DISABLE_SOFTCAP>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_PACKGQA}>:FLASHATTENTION_DISABLE_PACKGQA>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_FP16}>:FLASHATTENTION_DISABLE_FP16>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_FP8}>:FLASHATTENTION_DISABLE_FP8>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_INT8}>:FLASHATTENTION_DISABLE_INT8>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_SM80}>:FLASHATTENTION_DISABLE_SM8x>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_VARLEN}>:FLASHATTENTION_DISABLE_VARLEN>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_CLUSTER}>:FLASHATTENTION_DISABLE_CLUSTER>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_HDIM64}>:FLASHATTENTION_DISABLE_HDIM64>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_HDIM96}>:FLASHATTENTION_DISABLE_HDIM96>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_HDIM128}>:FLASHATTENTION_DISABLE_HDIM128>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_HDIM192}>:FLASHATTENTION_DISABLE_HDIM192>
    $<$<BOOL:${FLASH_ATTENTION_DISABLE_HDIM256}>:FLASHATTENTION_DISABLE_HDIM256>
)

# Install the extension into the lite_attention package
install(TARGETS _C LIBRARY DESTINATION lite_attention)
