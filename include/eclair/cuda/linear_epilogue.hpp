/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/linear_epilogue.hpp"

#include <cuda_runtime_api.h>

namespace eclair::cuda {

// Device pointers only. Caller owns buffers and stream until completion.
// No allocation, copy, synchronization or device switch. Validation failures
// enqueue no work. Success means enqueued. One pass reads FP32 accumulators
// and writes binary16, replacing separate multiply, add and conversion passes.
[[nodiscard]] cudaError_t launch_linear_epilogue(const ops::LinearEpilogueShape& shape,
                                                 const ops::LinearEpilogueBuffers& buffers,
                                                 cudaStream_t stream) noexcept;

}  // namespace eclair::cuda
