/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace eclair::cuda {

inline constexpr std::size_t kDeviceNameCapacity = 256U;

struct DeviceCapabilities {
    int device_index = -1;
    int compute_major = 0;
    int compute_minor = 0;
    std::size_t global_memory_bytes = 0U;
    int multiprocessor_count = 0;
    int warp_size = 0;
    int max_threads_per_block = 0;
    char name[kDeviceNameCapacity]{};
};

// Query one device without changing the current device or allocating memory.
// The caller owns the output object and must provide a non-null pointer.
[[nodiscard]] cudaError_t query_device_capabilities(
    int device_index, DeviceCapabilities* output) noexcept;

}  // namespace eclair::cuda
