/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/gated_delta.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

// The gated delta rule state pass of eclair/ops/gated_delta.hpp on NVIDIA
// tensor cores (WMMA, binary16 inputs, FP32 accumulation). Device pointers
// only. The caller owns buffers and the stream until the work completes.
// Nothing is allocated, copied or synchronized, and the current device does
// not change. Invalid or unsupported arguments enqueue no work and return
// cudaErrorInvalidValue. Success means enqueued.
namespace eclair::cuda {

// Beyond the contract's validation the kernel needs: key size 64 or 128, a
// value size that is a multiple of 64, binary16 arrays 16-byte aligned, the
// FP32 final states 32-byte aligned, and every count within int range.
[[nodiscard]] bool gated_delta_states_supported(
    const ops::GatedDeltaStatesShape& shape, const ops::GatedDeltaStatesBuffers& buffers) noexcept;

[[nodiscard]] cudaError_t launch_gated_delta_states(const ops::GatedDeltaStatesShape& shape,
                                                    const ops::GatedDeltaStatesBuffers& buffers,
                                                    cudaStream_t stream) noexcept;

// Kernel-image probe for every instantiation, not a numerical or performance
// test. Restores the previous current device and clears an expected error.
[[nodiscard]] cudaError_t check_gated_delta_image(int device_index) noexcept;

}  // namespace eclair::cuda
