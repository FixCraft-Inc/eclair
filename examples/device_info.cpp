/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/capabilities.hpp"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(status));
        return EXIT_FAILURE;
    }
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA devices are available.\n");
        return EXIT_FAILURE;
    }

    for (int device_index = 0; device_index < device_count; ++device_index) {
        eclair::cuda::DeviceCapabilities capabilities{};
        status = eclair::cuda::query_device_capabilities(device_index, &capabilities);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "query_device_capabilities(%d): %s\n", device_index,
                         cudaGetErrorString(status));
            return EXIT_FAILURE;
        }
        std::printf("device=%d name=%s compute=%d.%d memory_bytes=%zu multiprocessors=%d "
                    "warp_size=%d max_threads_per_block=%d\n",
                    capabilities.device_index, capabilities.name,
                    capabilities.compute_major, capabilities.compute_minor,
                    capabilities.global_memory_bytes, capabilities.multiprocessor_count,
                    capabilities.warp_size, capabilities.max_threads_per_block);
    }
    return EXIT_SUCCESS;
}
