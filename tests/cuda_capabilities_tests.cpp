/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/cuda/capabilities.hpp"

#include <cuda_runtime_api.h>

int main() {
    eclair::cuda::DeviceCapabilities capabilities{};
    ECLAIR_CHECK(eclair::cuda::query_device_capabilities(-1, &capabilities) ==
                 cudaErrorInvalidValue);
    ECLAIR_CHECK(eclair::cuda::query_device_capabilities(0, nullptr) == cudaErrorInvalidValue);

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    ECLAIR_CHECK(count_status == cudaSuccess);
    ECLAIR_CHECK(device_count > 0);
    if (count_status != cudaSuccess || device_count <= 0) {
        return eclair::test::finish();
    }
    ECLAIR_CHECK(eclair::cuda::query_device_capabilities(0, &capabilities) == cudaSuccess);
    ECLAIR_CHECK(capabilities.device_index == 0);
    ECLAIR_CHECK(capabilities.compute_major >= 1);
    ECLAIR_CHECK(capabilities.compute_minor >= 0);
    ECLAIR_CHECK(capabilities.global_memory_bytes > 0U);
    ECLAIR_CHECK(capabilities.multiprocessor_count > 0);
    ECLAIR_CHECK(capabilities.warp_size > 0);
    ECLAIR_CHECK(capabilities.max_threads_per_block > 0);
    ECLAIR_CHECK(capabilities.name[0] != '\0');
    ECLAIR_CHECK(capabilities.name[eclair::cuda::kDeviceNameCapacity - 1U] == '\0');
    return eclair::test::finish();
}
