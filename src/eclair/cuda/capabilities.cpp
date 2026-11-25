/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/capabilities.hpp"

#include <cstring>

namespace eclair::cuda {

cudaError_t query_device_capabilities(int device_index,
                                      DeviceCapabilities* output) noexcept {
    if (output == nullptr || device_index < 0) {
        return cudaErrorInvalidValue;
    }

    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, device_index);
    if (status != cudaSuccess) {
        return status;
    }

    DeviceCapabilities capabilities{};
    capabilities.device_index = device_index;
    capabilities.compute_major = properties.major;
    capabilities.compute_minor = properties.minor;
    capabilities.global_memory_bytes = properties.totalGlobalMem;
    capabilities.multiprocessor_count = properties.multiProcessorCount;
    capabilities.warp_size = properties.warpSize;
    capabilities.max_threads_per_block = properties.maxThreadsPerBlock;
    std::memcpy(capabilities.name, properties.name, kDeviceNameCapacity - 1U);
    capabilities.name[kDeviceNameCapacity - 1U] = '\0';
    *output = capabilities;
    return cudaSuccess;
}

}  // namespace eclair::cuda
