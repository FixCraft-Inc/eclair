/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Part of eclair_cuda_blas, the only library that links cuBLAS. C++17, so
// CUDA tests can include it. cuBLAS's own header stays out of it.

#include "eclair/ops/w4a16_linear.hpp"

#include <cuda_runtime_api.h>

struct cublasContext;  // cuBLAS's handle type (cublasHandle_t points to it).

namespace eclair::cuda {

// Owns one cuBLAS handle bound to the device it was created on. Creation
// allocates cuBLAS's own default workspace, but calls use the caller's.
// One call at a time per handle.
class BlasHandle {
public:
    BlasHandle() noexcept = default;
    ~BlasHandle();
    BlasHandle(BlasHandle&& other) noexcept;
    BlasHandle& operator=(BlasHandle&& other) noexcept;
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;

    // Replaces `handle` with a new handle on `device_index` and restores the
    // caller's current device. On failure `handle` is left empty.
    [[nodiscard]] static cudaError_t create(int device_index, BlasHandle& handle) noexcept;
    [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] int device() const noexcept { return device_; }
    [[nodiscard]] cublasContext* get() const noexcept { return handle_; }

private:
    void reset() noexcept;

    cublasContext* handle_ = nullptr;
    int device_ = -1;
};

// Large-M W4A16 linear (ops::validate_w4a16_gemm): Eclair's reconstruction
// writes every binary16 weight into `dequantized`, then cuBLAS multiplies
// with FP16 operands, FP32 compute, FP32 split reductions only and no
// atomics, all set on the handle for this call together with the stream and
// the caller's workspace. M, N and K must fit in int, the workspace must
// start on a 256-byte boundary and the current device must be the handle's.
// Rejected arguments enqueue nothing. A cuBLAS failure after the
// reconstruction leaves only `dequantized` written. The caller owns every
// buffer and the stream until completion. No allocation or synchronization.
[[nodiscard]] cudaError_t launch_w4a16_gemm(BlasHandle& handle, const ops::W4A16LinearShape& shape,
                                            const ops::W4A16GemmBuffers& buffers,
                                            cudaStream_t stream) noexcept;

}  // namespace eclair::cuda
