/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/bf16_linear.hpp"

#include <cuda_runtime_api.h>

// The BF16 linear operation of eclair/ops/bf16_linear.hpp on NVIDIA GPUs.
// Device pointers only. The caller owns buffers and the stream until the work
// completes. Nothing is allocated, copied or synchronized, and the current
// device does not change. Invalid arguments enqueue no work and return
// cudaErrorInvalidValue. Success means enqueued, so callers check stream
// completion for asynchronous errors. No BF16 hardware instruction is used.
namespace eclair::cuda {

// One warp per row finds the largest finite magnitude, then converts the row.
// Results equal host::bf16_row_scaling_reference bit for bit.
[[nodiscard]] cudaError_t launch_bf16_row_scaling(const ops::Bf16RowScalingShape& shape,
                                                  const ops::Bf16RowScalingBuffers& buffers,
                                                  cudaStream_t stream) noexcept;

// M<=4 uses a CUDA-core warp per output channel, sharing weight loads across
// rows. Larger M uses FP16 WMMA. Both accumulate in FP32 in unspecified order.
// Nonempty dimensions must fit in int.
[[nodiscard]] cudaError_t launch_bf16_linear(const ops::Bf16LinearShape& shape,
                                             const ops::Bf16LinearBuffers& buffers,
                                             cudaStream_t stream) noexcept;

// Kernel-image probe for every scaling/linear variant, not a numerical or performance test.
// Restores the previous current device and clears an expected lookup error.
[[nodiscard]] cudaError_t check_bf16_linear_image(int device_index) noexcept;

}  // namespace eclair::cuda
