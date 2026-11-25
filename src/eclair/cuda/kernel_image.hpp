/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Internal to eclair_cuda: the kernel-image check behind capability probes.
// Not a public header.

#include <cuda_runtime_api.h>

#include <initializer_list>

namespace eclair::cuda::detail {

// Returns cudaSuccess when the build holds an image of every listed kernel
// that `device_index` can load, otherwise the first lookup error. This
// initializes the device's primary context and restores the caller's current
// device. A failed lookup is an expected answer, so its per-thread error is
// cleared before a later launch can report it.
template <typename... Kernels>
[[nodiscard]] cudaError_t check_kernel_images(int device_index, Kernels*... kernels) noexcept {
    if (device_index < 0) {
        return cudaErrorInvalidValue;
    }
    int previous_device = 0;
    cudaError_t status = cudaGetDevice(&previous_device);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        return status;
    }
    cudaError_t image_status = cudaSuccess;
    for (const void* kernel : {reinterpret_cast<const void*>(kernels)...}) {
        cudaFuncAttributes attributes{};
        image_status = cudaFuncGetAttributes(&attributes, kernel);
        if (image_status != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            break;
        }
    }
    const cudaError_t restore_status = cudaSetDevice(previous_device);
    return image_status != cudaSuccess ? image_status : restore_status;
}

}  // namespace eclair::cuda::detail
