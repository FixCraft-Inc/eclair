/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/w4a16_gemm.hpp"

#include "eclair/cuda/w4a16_linear.hpp"

#include <cublas_v2.h>

#include <cstdint>
#include <limits>
#include <utility>

namespace eclair::cuda {
namespace {

// cuBLAS statuses at Eclair's cudaError_t seam. The enum is cuBLAS's own and
// grows between releases, so unknown values map to cudaErrorUnknown.
cudaError_t from_blas(cublasStatus_t status) noexcept {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return cudaSuccess;
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return cudaErrorInitializationError;
        case CUBLAS_STATUS_ALLOC_FAILED:
            return cudaErrorMemoryAllocation;
        case CUBLAS_STATUS_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return cudaErrorNoKernelImageForDevice;
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return cudaErrorNotSupported;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return cudaErrorLaunchFailure;
        default:
            return cudaErrorUnknown;
    }
}

}  // namespace

BlasHandle::~BlasHandle() {
    reset();
}

BlasHandle::BlasHandle(BlasHandle&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), device_(std::exchange(other.device_, -1)) {}

BlasHandle& BlasHandle::operator=(BlasHandle&& other) noexcept {
    if (this != &other) {
        reset();
        handle_ = std::exchange(other.handle_, nullptr);
        device_ = std::exchange(other.device_, -1);
    }
    return *this;
}

void BlasHandle::reset() noexcept {
    if (handle_ != nullptr) {
        // Nothing can report a destroy failure from here. cuBLAS releases the
        // handle's resources on its device regardless of the current one.
        static_cast<void>(cublasDestroy(handle_));
        handle_ = nullptr;
        device_ = -1;
    }
}

cudaError_t BlasHandle::create(int device_index, BlasHandle& handle) noexcept {
    handle.reset();
    if (device_index < 0) {
        return cudaErrorInvalidValue;
    }
    int previous = 0;
    cudaError_t status = cudaGetDevice(&previous);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        return status;
    }
    cublasHandle_t created = nullptr;
    status = from_blas(cublasCreate(&created));
    BlasHandle owner;
    owner.handle_ = created;
    owner.device_ = device_index;
    // Fixed for the handle's lifetime: host scalars and no atomics, so
    // repeated calls on one stream give identical bits.
    if (status == cudaSuccess) {
        status = from_blas(cublasSetPointerMode(created, CUBLAS_POINTER_MODE_HOST));
    }
    if (status == cudaSuccess) {
        status = from_blas(cublasSetAtomicsMode(created, CUBLAS_ATOMICS_NOT_ALLOWED));
    }
    const cudaError_t restore = cudaSetDevice(previous);
    if (status == cudaSuccess && restore == cudaSuccess) {
        handle = std::move(owner);
    }
    return status != cudaSuccess ? status : restore;
}

cudaError_t launch_w4a16_gemm(BlasHandle& handle, const ops::W4A16LinearShape& s,
                              const ops::W4A16GemmBuffers& b, cudaStream_t stream) noexcept {
    if (!handle.valid() || ops::validate_w4a16_gemm(s, b) != ops::Status::Success) {
        return cudaErrorInvalidValue;
    }
    if (s.m == 0U || s.n == 0U) {
        return cudaSuccess;
    }
    constexpr auto max = static_cast<std::size_t>(std::numeric_limits<int>::max());
    // cuBLAS requires a 256-byte aligned workspace.
    if (s.m > max || s.n > max || s.k > max ||
        reinterpret_cast<std::uintptr_t>(b.workspace.data) % 256U != 0U) {
        return cudaErrorInvalidValue;
    }
    int current = -1;
    cudaError_t status = cudaGetDevice(&current);
    if (status != cudaSuccess) {
        return status;
    }
    if (current != handle.device()) {
        return cudaErrorInvalidDevice;
    }
    status = launch_w4a16_dequantize(s.n, s.k, {b.weights, b.scales, b.dequantized}, stream);
    if (status != cudaSuccess) {
        return status;
    }
    cublasHandle_t blas = handle.get();
    // cublasSetStream resets the workspace to cuBLAS's default pool, so the
    // caller's workspace is set after it.
    status = from_blas(cublasSetStream(blas, stream));
    if (status == cudaSuccess) {
        status = from_blas(cublasSetWorkspace(blas, b.workspace.data, b.workspace.size));
    }
    if (status == cudaSuccess) {
        status = from_blas(cublasSetMathMode(
            blas, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                                            CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    }
    if (status != cudaSuccess) {
        return status;
    }
    // Row-major out[M,N] = A[M,K] * W[K,N] is column-major out^T = W^T * A^T,
    // with W^T as the [N,K] operand (leading dimension N) and A^T as [K,M].
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const int m = static_cast<int>(s.m);
    const int n = static_cast<int>(s.n);
    const int k = static_cast<int>(s.k);
    return from_blas(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                                  b.dequantized.data, CUDA_R_16F, n, b.activations.data, CUDA_R_16F,
                                  k, &beta, b.output.data, CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
}

}  // namespace eclair::cuda
