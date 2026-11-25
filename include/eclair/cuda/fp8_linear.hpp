/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/fp8_linear.hpp"

#include <cuda_runtime_api.h>

namespace eclair::cuda {

// Device pointers only. Caller owns buffers and stream until completion.
// No allocation, host/device copy, synchronization, device switch or implicit
// CPU fallback. Validation failures enqueue no work. Success means enqueued;
// callers must check stream completion for asynchronous errors.
// Software FP8 decode and scale -> FP16 operands -> FP32 accumulation.
// M <= 4 uses a CUDA-core GEMV; larger batches use WMMA. Zero padding handles
// arbitrary contiguous shapes in the WMMA path on tensor-core hardware.
[[nodiscard]] cudaError_t launch_fp8_linear(const ops::Fp8LinearShape& shape,
                                            const ops::Fp8LinearBuffers& buffers,
                                            cudaStream_t stream) noexcept;

// Kernel-image probe, not a numerical or performance test. Restores the
// previous current device and clears an expected image-lookup error.
[[nodiscard]] cudaError_t check_fp8_linear_image(int device_index) noexcept;

}  // namespace eclair::cuda
