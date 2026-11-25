cmake_minimum_required(VERSION 3.24)

project(Eclair
    VERSION 0.1.0
    DESCRIPTION "Emulation and Compatibility Layer for Accelerated Inference and Rendering"
    LANGUAGES CXX
)

option(ECLAIR_BUILD_TESTS "Build Eclair tests" ON)
option(ECLAIR_ENABLE_CUDA "Enable optional CUDA targets" OFF)
option(ECLAIR_BUILD_RENDER_DEMO "Build the optional SDL Cornell preview" OFF)
option(ECLAIR_BUILD_GPU_QUALIFICATION "Build opt-in Volta hardware qualification tooling" OFF)
set(ECLAIR_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures to compile, for example 86 on a validation laptop or 70 on a V100")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

# Warning policy for first-party host C++ targets.
function(eclair_target_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
    )
endfunction()

# Numerical semantics: format definitions and portable reference conversion.
add_library(eclair_core STATIC
    src/eclair/formats/bf16.cpp
    src/eclair/formats/fp8.cpp
    src/eclair/formats/fp16.cpp
)
target_include_directories(eclair_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(eclair_core PUBLIC cxx_std_20)
eclair_target_warnings(eclair_core)

# Operation contracts and validation; no framework or vendor dependency.
add_library(eclair_ops STATIC
    src/eclair/ops/ray_trace.cpp
    src/eclair/ops/w4a16_linear.cpp
    src/eclair/ops/bf16_linear.cpp
    src/eclair/ops/fp8_linear.cpp
    src/eclair/ops/linear_epilogue.cpp
    src/eclair/ops/gated_delta.cpp
    src/eclair/ops/gated_delta_output.cpp
    src/eclair/ops/paged_attention.cpp
)
target_link_libraries(eclair_ops PUBLIC eclair_core)
eclair_target_warnings(eclair_ops)

# Capability contracts: the feature catalog, report types, and policy. No
# backend, vendor, or CLI dependency.
add_library(eclair_capability STATIC
    src/eclair/capability/feature.cpp
    src/eclair/capability/policy.cpp
    src/eclair/capability/report.cpp
)
target_include_directories(eclair_capability
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(eclair_capability PUBLIC cxx_std_20)
eclair_target_warnings(eclair_capability)

# Device-independent selection policy. Execution and storage stay caller-owned.
add_library(eclair_runtime STATIC
    src/eclair/runtime/bf16_linear.cpp
    src/eclair/runtime/fp8_linear.cpp
    src/eclair/runtime/linear_selection.cpp
)
target_link_libraries(eclair_runtime PUBLIC eclair_ops eclair_capability)
eclair_target_warnings(eclair_runtime)

# Host reference backend, always available as the fallback.
add_library(eclair_backend_host STATIC
    src/eclair/host/ray_trace.cpp
    src/eclair/host/w4a16_linear.cpp
    src/eclair/host/bf16_linear.cpp
    src/eclair/host/host_backend.cpp
    src/eclair/host/fp8_linear.cpp
    src/eclair/host/linear_epilogue.cpp
    src/eclair/host/gated_delta.cpp
    src/eclair/host/gated_delta_output.cpp
    src/eclair/host/paged_attention.cpp
)
target_link_libraries(eclair_backend_host PUBLIC eclair_capability eclair_core eclair_ops)
eclair_target_warnings(eclair_backend_host)

# NVIDIA compute-capability rules. Host-only C++, so its tests run without a
# CUDA toolkit. The CUDA backend feeds it live probe results.
add_library(eclair_cuda_architecture STATIC
    src/eclair/cuda/architecture.cpp
)
target_link_libraries(eclair_cuda_architecture PUBLIC eclair_capability)
eclair_target_warnings(eclair_cuda_architecture)

# These archives can also be linked into opt-in framework shared libraries.
set_target_properties(eclair_core eclair_ops eclair_capability eclair_cuda_architecture
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ECLAIR_BUILD_TESTS)
    enable_testing()
    add_executable(eclair_material_tests tests/material_tests.cpp)
    target_link_libraries(eclair_material_tests PRIVATE eclair_core)
    eclair_target_warnings(eclair_material_tests)
    add_test(NAME eclair.material COMMAND eclair_material_tests)
    add_executable(eclair_ray_preview_tests tests/ray_preview_tests.cpp)
    target_link_libraries(eclair_ray_preview_tests PRIVATE eclair_ops)
    eclair_target_warnings(eclair_ray_preview_tests)
    add_test(NAME eclair.ray_preview COMMAND eclair_ray_preview_tests)

    add_executable(eclair_ray_trace_tests tests/ray_trace_tests.cpp)
    target_link_libraries(eclair_ray_trace_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_ray_trace_tests)
    add_test(NAME eclair.ray_trace COMMAND eclair_ray_trace_tests)
    add_executable(eclair_w4a16_linear_tests tests/w4a16_linear_tests.cpp)
    target_link_libraries(eclair_w4a16_linear_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_w4a16_linear_tests)
    add_test(NAME eclair.w4a16_linear COMMAND eclair_w4a16_linear_tests)
    add_executable(eclair_fp8_tests tests/fp8_tests.cpp)
    target_link_libraries(eclair_fp8_tests PRIVATE eclair_core)
    eclair_target_warnings(eclair_fp8_tests)
    add_test(NAME eclair.fp8 COMMAND eclair_fp8_tests)

    add_executable(eclair_bf16_tests tests/bf16_tests.cpp)
    target_link_libraries(eclair_bf16_tests PRIVATE eclair_core)
    eclair_target_warnings(eclair_bf16_tests)
    add_test(NAME eclair.bf16 COMMAND eclair_bf16_tests)

    add_executable(eclair_fp16_tests tests/fp16_tests.cpp)
    target_link_libraries(eclair_fp16_tests PRIVATE eclair_core)
    eclair_target_warnings(eclair_fp16_tests)
    add_test(NAME eclair.fp16 COMMAND eclair_fp16_tests)

    add_executable(eclair_bf16_linear_tests tests/bf16_linear_tests.cpp)
    target_link_libraries(eclair_bf16_linear_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_bf16_linear_tests)
    add_test(NAME eclair.bf16_linear COMMAND eclair_bf16_linear_tests)

    add_executable(eclair_fp8_linear_tests tests/fp8_linear_tests.cpp)
    target_link_libraries(eclair_fp8_linear_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_fp8_linear_tests)
    add_test(NAME eclair.fp8_linear COMMAND eclair_fp8_linear_tests)

    add_executable(eclair_linear_epilogue_tests tests/linear_epilogue_tests.cpp)
    target_link_libraries(eclair_linear_epilogue_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_linear_epilogue_tests)
    add_test(NAME eclair.linear_epilogue COMMAND eclair_linear_epilogue_tests)

    add_executable(eclair_gated_delta_tests tests/gated_delta_tests.cpp)
    target_link_libraries(eclair_gated_delta_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_gated_delta_tests)
    add_test(NAME eclair.gated_delta COMMAND eclair_gated_delta_tests)

    add_executable(eclair_gated_delta_output_tests tests/gated_delta_output_tests.cpp)
    target_link_libraries(eclair_gated_delta_output_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_gated_delta_output_tests)
    add_test(NAME eclair.gated_delta_output COMMAND eclair_gated_delta_output_tests)

    add_executable(eclair_paged_attention_tests tests/paged_attention_tests.cpp)
    target_link_libraries(eclair_paged_attention_tests PRIVATE eclair_backend_host)
    eclair_target_warnings(eclair_paged_attention_tests)
    add_test(NAME eclair.paged_attention COMMAND eclair_paged_attention_tests)

    add_executable(eclair_runtime_tests tests/runtime_tests.cpp)
    target_link_libraries(eclair_runtime_tests PRIVATE eclair_runtime eclair_backend_host eclair_cuda_architecture)
    eclair_target_warnings(eclair_runtime_tests)
    add_test(NAME eclair.runtime COMMAND eclair_runtime_tests)

    add_executable(eclair_capability_tests tests/capability_tests.cpp)
    target_link_libraries(eclair_capability_tests
        PRIVATE eclair_backend_host eclair_cuda_architecture)
    eclair_target_warnings(eclair_capability_tests)
    add_test(NAME eclair.capability COMMAND eclair_capability_tests)
endif()

if(ECLAIR_ENABLE_CUDA)
    if(ECLAIR_CUDA_ARCHITECTURES)
        # Set this before enable_language: CMake's CUDA compiler probe needs an
        # architecture even when all targets set one explicitly later.
        set(CMAKE_CUDA_ARCHITECTURES "${ECLAIR_CUDA_ARCHITECTURES}" CACHE STRING
            "CUDA architectures selected for Eclair" FORCE)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(eclair_cuda STATIC
        src/eclair/cuda/ray_trace.cu
        src/eclair/cuda/w4a16_linear.cu
        src/eclair/cuda/bf16_convert.cu
        src/eclair/cuda/bf16_linear.cu
        src/eclair/cuda/capabilities.cpp
        src/eclair/cuda/cuda_backend.cpp
        src/eclair/cuda/fp8_convert.cu
        src/eclair/cuda/fp8_linear.cu
        src/eclair/cuda/linear_epilogue.cu
        src/eclair/cuda/gated_delta.cu
        src/eclair/cuda/gated_delta_output.cu
        src/eclair/cuda/paged_attention.cu
        src/eclair/cuda/paged_decode.cu
    )
    target_include_directories(eclair_cuda
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(eclair_cuda
        PUBLIC eclair_core eclair_ops eclair_capability CUDA::cudart
        PRIVATE eclair_cuda_architecture)
    eclair_target_warnings(eclair_cuda)
    set_source_files_properties(src/eclair/cuda/ray_trace.cu PROPERTIES
        COMPILE_OPTIONS "--fmad=false")
    # Reported by eclair-smi so a missing kernel image can be diagnosed.
    set_property(SOURCE src/eclair/cuda/cuda_backend.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        "ECLAIR_CUDA_BUILD_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES}\"")
    set_target_properties(eclair_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    if(ECLAIR_CUDA_ARCHITECTURES)
        set_property(TARGET eclair_cuda PROPERTY CUDA_ARCHITECTURES
            "${ECLAIR_CUDA_ARCHITECTURES}")
    endif()

    # cuBLAS-backed operations. Only this library and its consumers link
    # cuBLAS, so eclair_cuda and eclair-smi stay free of it. The large-M
    # W4A16 linear needs a tuned tensor-core GEMM whose compute type and
    # reduction precision it sets on its own handle for every call.
    add_library(eclair_cuda_blas STATIC src/eclair/cuda/w4a16_gemm.cpp)
    target_link_libraries(eclair_cuda_blas PUBLIC eclair_cuda PRIVATE CUDA::cublas)
    target_compile_features(eclair_cuda_blas PRIVATE cxx_std_20)
    set_target_properties(eclair_cuda_blas PROPERTIES POSITION_INDEPENDENT_CODE ON)
    eclair_target_warnings(eclair_cuda_blas)

    if(ECLAIR_BUILD_TESTS)
        # A CUDA test program: C++17 like every CUDA unit, the selected
        # architectures, and the shared helpers in tests/cuda_support.hpp.
        function(eclair_add_cuda_test name source)
            add_executable(${name} ${source})
            target_link_libraries(${name} PRIVATE eclair_cuda ${ARGN})
            set_target_properties(${name} PROPERTIES
                CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
            if(ECLAIR_CUDA_ARCHITECTURES)
                set_property(TARGET ${name} PROPERTY CUDA_ARCHITECTURES
                    "${ECLAIR_CUDA_ARCHITECTURES}")
            endif()
        endfunction()

        eclair_add_cuda_test(eclair_cuda_fp8_tests tests/cuda_fp8_tests.cu eclair_core)
        add_test(NAME eclair.cuda_fp8 COMMAND eclair_cuda_fp8_tests)
        eclair_add_cuda_test(eclair_cuda_ray_trace_tests tests/cuda_ray_trace_tests.cu eclair_backend_host)
        add_test(NAME eclair.cuda_ray_trace COMMAND eclair_cuda_ray_trace_tests)

        eclair_add_cuda_test(eclair_cuda_w4a16_linear_tests tests/cuda_w4a16_linear_tests.cu
            eclair_backend_host)
        add_test(NAME eclair.cuda_w4a16_linear COMMAND eclair_cuda_w4a16_linear_tests)

        eclair_add_cuda_test(eclair_cuda_w4a16_gemm_tests tests/cuda_w4a16_gemm_tests.cu
            eclair_cuda_blas eclair_backend_host)
        add_test(NAME eclair.cuda_w4a16_gemm COMMAND eclair_cuda_w4a16_gemm_tests)

        eclair_add_cuda_test(eclair_cuda_bf16_tests tests/cuda_bf16_tests.cu eclair_core)
        add_test(NAME eclair.cuda_bf16 COMMAND eclair_cuda_bf16_tests)

        eclair_add_cuda_test(eclair_cuda_bf16_linear_tests tests/cuda_bf16_linear_tests.cu
            eclair_backend_host CUDA::cublas)
        add_test(NAME eclair.cuda_bf16_linear COMMAND eclair_cuda_bf16_linear_tests)

        # Explicit LLM-shape performance experiment, not a default CTest load.
        eclair_add_cuda_test(eclair_cuda_skinny_linear_bench benchmarks/skinny_linear.cu CUDA::cublas)

        eclair_add_cuda_test(eclair_cuda_fp8_linear_tests tests/cuda_fp8_linear_tests.cu
            eclair_backend_host CUDA::cublas)
        add_test(NAME eclair.cuda_fp8_linear COMMAND eclair_cuda_fp8_linear_tests)

        eclair_add_cuda_test(eclair_cuda_linear_epilogue_tests tests/cuda_linear_epilogue_tests.cu
            eclair_backend_host)
        add_test(NAME eclair.cuda_linear_epilogue COMMAND eclair_cuda_linear_epilogue_tests)

        eclair_add_cuda_test(eclair_cuda_gated_delta_tests tests/cuda_gated_delta_tests.cu
            eclair_backend_host)
        add_test(NAME eclair.cuda_gated_delta COMMAND eclair_cuda_gated_delta_tests)

        eclair_add_cuda_test(eclair_cuda_gated_delta_output_tests tests/cuda_gated_delta_output_tests.cu
            eclair_backend_host)
        add_test(NAME eclair.cuda_gated_delta_output COMMAND eclair_cuda_gated_delta_output_tests)

        eclair_add_cuda_test(eclair_cuda_paged_attention_tests tests/cuda_paged_attention_tests.cu
            eclair_backend_host)
        add_test(NAME eclair.cuda_paged_attention COMMAND eclair_cuda_paged_attention_tests)
        eclair_add_cuda_test(eclair_cuda_paged_decode_tests tests/cuda_paged_decode_tests.cu
            eclair_backend_host)
        add_test(NAME eclair.cuda_paged_decode COMMAND eclair_cuda_paged_decode_tests)

        add_executable(eclair_cuda_capabilities_tests tests/cuda_capabilities_tests.cpp)
        target_link_libraries(eclair_cuda_capabilities_tests PRIVATE eclair_cuda CUDA::cudart)
        eclair_target_warnings(eclair_cuda_capabilities_tests)
        add_test(NAME eclair.cuda_capabilities COMMAND eclair_cuda_capabilities_tests)
    endif()

    # C interface for loaders such as Python's ctypes (integrations/vllm). Only the
    # eclair_* C functions are exported, and the CUDA runtime is shared with the host process.
    add_library(eclair_kernels SHARED src/eclair/c/gated_delta.cpp src/eclair/c/paged_attention.cpp
        src/eclair/c/gated_delta_output.cpp src/eclair/c/bf16_linear.cpp src/eclair/c/w4a16_linear.cpp
        src/eclair/c/w4a16_gemm.cpp)
    # cuBLAS is shared like the CUDA runtime: inside PyTorch the process's
    # loaded libcublas.so.12 serves both.
    target_link_libraries(eclair_kernels PRIVATE eclair_cuda eclair_cuda_blas CUDA::cudart)
    target_compile_features(eclair_kernels PRIVATE cxx_std_20)
    set_target_properties(eclair_kernels PROPERTIES
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON CUDA_RUNTIME_LIBRARY Shared)
    # Visibility on this target does not hide definitions pulled from static
    # CUDA/ops archives. Keep those implementation symbols out of the C ABI.
    target_link_options(eclair_kernels PRIVATE
        $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>)
    eclair_target_warnings(eclair_kernels)

    add_executable(eclair_device_info examples/device_info.cpp)
    target_link_libraries(eclair_device_info PRIVATE eclair_cuda CUDA::cudart)
    target_compile_features(eclair_device_info PRIVATE cxx_std_20)

    if(ECLAIR_BUILD_GPU_QUALIFICATION)
        # CMake 3.25 predates nvcc's C++20 support (added in CUDA 12).
        if(CMAKE_VERSION VERSION_LESS 3.26
           AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
           AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
            set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "-std=c++20")
            set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "-std=c++20")
        endif()
        add_executable(eclair_card_qualify benchmarks/card_qualify.cu)
        target_link_libraries(eclair_card_qualify PRIVATE CUDA::cudart CUDA::cublas)
        set_target_properties(eclair_card_qualify PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
        target_compile_options(eclair_card_qualify PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
    endif()
endif()

add_subdirectory(cli/eclair-smi)
add_subdirectory(cli/eclair-rt)
