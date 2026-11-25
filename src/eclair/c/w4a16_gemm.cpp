/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/c/w4a16_gemm.h"

#include "eclair/cuda/w4a16_gemm.hpp"
#include "eclair/cuda/w4a16_linear.hpp"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <new>

struct EclairW4A16Gemm {
    eclair::cuda::BlasHandle handle;
};

extern "C" __attribute__((visibility("default"))) int eclair_w4a16_gemm_create(
    int device_index, EclairW4A16Gemm** gemm) {
    if (gemm == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    *gemm = nullptr;
    auto* owner = new (std::nothrow) EclairW4A16Gemm{};
    if (owner == nullptr) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }
    const cudaError_t status = eclair::cuda::BlasHandle::create(device_index, owner->handle);
    if (status != cudaSuccess) {
        delete owner;
        return static_cast<int>(status);
    }
    *gemm = owner;
    return 0;
}

extern "C" __attribute__((visibility("default"))) void eclair_w4a16_gemm_destroy(
    EclairW4A16Gemm* gemm) {
    delete gemm;
}

extern "C" __attribute__((visibility("default"))) int eclair_w4a16_gemm(
    EclairW4A16Gemm* gemm, const EclairW4A16GemmArgs* args, void* stream) {
    if (gemm == nullptr || args == nullptr || args->abi != ECLAIR_W4A16_GEMM_ABI) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto& a = *args;
    for (const std::uint64_t value :
         {a.activation_elements, a.weight_elements, a.scale_elements, a.dequantized_elements,
          a.output_elements, a.workspace_bytes, a.m, a.n, a.k}) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    const auto size = [](std::uint64_t v) { return static_cast<std::size_t>(v); };
    const eclair::ops::W4A16GemmBuffers b{
        {static_cast<const std::uint16_t*>(a.activations), size(a.activation_elements)},
        {a.weights, size(a.weight_elements)},
        {static_cast<const std::uint16_t*>(a.scales), size(a.scale_elements)},
        {static_cast<std::uint16_t*>(a.dequantized), size(a.dequantized_elements)},
        {static_cast<std::uint16_t*>(a.output), size(a.output_elements)},
        {static_cast<std::byte*>(a.workspace), size(a.workspace_bytes)}};
    return static_cast<int>(eclair::cuda::launch_w4a16_gemm(
        gemm->handle, {size(a.m), size(a.n), size(a.k), 1}, b, static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_w4a16_gemm_image(int device_index) {
    return static_cast<int>(eclair::cuda::check_w4a16_dequantize_image(device_index));
}
