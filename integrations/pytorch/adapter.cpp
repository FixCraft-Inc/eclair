/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/capability/policy.hpp"
#include "eclair/cuda/fp8_linear.hpp"
#include "eclair/cuda/linear_epilogue.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#ifndef ECLAIR_TORCH_ATEN_OVERRIDE
#define ECLAIR_TORCH_ATEN_OVERRIDE 1
#endif

namespace eclair::pytorch {
namespace {

void check_policy() {
    capability::FeaturePolicy policy;
    if (const char* value = std::getenv("ECLAIR_DISABLE")) {
        const auto error = policy.disable_list(value);
        TORCH_CHECK(!error, "Eclair policy: ", error.value_or(""));
    }
    TORCH_CHECK(!policy.is_disabled(capability::Feature::Fp8Gemm),
                "Eclair FP8 matrix multiplication is disabled by ECLAIR_DISABLE");
}

void check_tensor(const at::Tensor& value, const at::Tensor& anchor, const char* name) {
    TORCH_CHECK(value.is_cuda() && value.device() == anchor.device(), "Eclair ", name,
                " must be on the same CUDA device");
    TORCH_CHECK(value.layout() == at::kStrided && value.is_contiguous(), "Eclair ", name,
                " must be a contiguous strided tensor");
    TORCH_CHECK(!value.is_conj() && !value.is_neg(), "Eclair ", name,
                " must not carry conjugate or negative view bits");
    TORCH_CHECK(!value.requires_grad(), "Eclair supports inference only: ", name);
}

void check_device_and_stream() {
    const auto* properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 7 && properties->minor == 0,
                "Eclair FP8 linear is validated for sm_70 only");
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const auto status = cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture);
    TORCH_CHECK(status == cudaSuccess && capture == cudaStreamCaptureStatusNone,
                "Eclair PyTorch adapter does not support CUDA graph capture");
    TORCH_CHECK(!at::globalContext().deterministicAlgorithms(),
                "Eclair PyTorch adapter does not promise deterministic algorithms");
}

#if ECLAIR_TORCH_ATEN_OVERRIDE
std::size_t workspace_budget() {
    constexpr std::size_t kDefault = 64U * 1024U * 1024U;
    const char* value = std::getenv("ECLAIR_TORCH_WORKSPACE_BYTES");
    if (value == nullptr) {
        return kDefault;
    }
    const std::string_view text(value);
    std::size_t bytes = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), bytes);
    TORCH_CHECK(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
                "ECLAIR_TORCH_WORKSPACE_BYTES must be a nonnegative byte count");
    return bytes;
}
#endif

formats::Fp8Format fp8_format(at::ScalarType dtype) {
    if (dtype == at::kFloat8_e4m3fn) {
        return formats::Fp8Format::E4M3FN;
    }
    TORCH_CHECK(dtype == at::kFloat8_e5m2, "Eclair requires E4M3FN or E5M2 weights");
    return formats::Fp8Format::E5M2;
}

at::Tensor fp8_linear(const at::Tensor& a, const at::Tensor& weights, const at::Tensor& scales,
                      std::int64_t block_n, std::int64_t block_k,
                      const std::optional<at::Tensor>& bias) {
    check_policy();
    check_tensor(a, a, "activations");
    check_tensor(weights, a, "weights");
    check_tensor(scales, a, "scales");
    TORCH_CHECK(a.dim() == 2 && weights.dim() == 2 && a.size(1) == weights.size(1),
                "Eclair requires A[M,K] and weights[N,K]");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && weights.size(0) > 0,
                "Eclair PyTorch adapter requires nonempty matrices");
    TORCH_CHECK(a.scalar_type() == at::kHalf && scales.scalar_type() == at::kFloat,
                "Eclair requires FP16 activations and FP32 scales");
    TORCH_CHECK(block_n > 0 && block_k > 0, "Eclair scale blocks must be positive");
    const auto format = fp8_format(weights.scalar_type());
    const c10::cuda::CUDAGuard guard(a.device());
    check_device_and_stream();
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                    weights.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max(),
                "Eclair dimensions exceed the CUDA operation limit");
    const ops::Fp8LinearShape shape{
        static_cast<std::size_t>(a.size(0)), static_cast<std::size_t>(weights.size(0)),
        static_cast<std::size_t>(a.size(1)), static_cast<std::size_t>(block_n),
        static_cast<std::size_t>(block_k),   format};
    ops::Fp8LinearSizes sizes;
    TORCH_CHECK(ops::fp8_linear_sizes(shape, sizes) == ops::Status::Success,
                "Eclair invalid FP8 linear shape");
    TORCH_CHECK(static_cast<std::size_t>(scales.numel()) == sizes.scales,
                "Eclair scale count does not match the block layout");
    if (bias) {
        check_tensor(*bias, a, "bias");
        TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->dim() == 1 &&
                        bias->size(0) == weights.size(0),
                    "Eclair bias must be FP32 [N]");
    }
    auto output = at::empty({a.size(0), weights.size(0)}, a.options().dtype(at::kFloat));
    const ops::Fp8LinearBuffers buffers{
        {reinterpret_cast<const std::uint16_t*>(a.const_data_ptr<at::Half>()),
         static_cast<std::size_t>(a.numel())},
        {static_cast<const std::uint8_t*>(weights.const_data_ptr()),
         static_cast<std::size_t>(weights.numel())},
        {scales.const_data_ptr<float>(), static_cast<std::size_t>(scales.numel())},
        {bias ? bias->const_data_ptr<float>() : nullptr,
         bias ? static_cast<std::size_t>(bias->numel()) : 0U},
        {output.mutable_data_ptr<float>(), static_cast<std::size_t>(output.numel())}};
    const auto status =
        cuda::launch_fp8_linear(shape, buffers, at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(status == cudaSuccess, "Eclair FP8 linear launch: ", cudaGetErrorString(status));
    return output;
}

// The framework owns this temporary and the thread-local cuBLAS handle. Do not
// change the handle's stream, pointer mode or math mode. Its workspace is
// separate from the explicit expanded-weight budget below.
at::Tensor expanded_mm(const at::Tensor& half_a, const at::Tensor& weights) {
    auto half_w = weights.to(at::kHalf);
    auto output = at::empty({half_a.size(0), weights.size(0)}, half_a.options().dtype(at::kFloat));
    const auto handle = at::cuda::getCurrentCUDABlasHandle();
    cublasPointerMode_t mode{};
    TORCH_CHECK(cublasGetPointerMode(handle, &mode) == CUBLAS_STATUS_SUCCESS &&
                    mode == CUBLAS_POINTER_MODE_HOST,
                "Eclair requires host cuBLAS pointer mode");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto status =
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(weights.size(0)),
                     static_cast<int>(half_a.size(0)), static_cast<int>(half_a.size(1)), &alpha,
                     half_w.const_data_ptr(), CUDA_R_16F, static_cast<int>(half_a.size(1)),
                     half_a.const_data_ptr(), CUDA_R_16F, static_cast<int>(half_a.size(1)), &beta,
                     output.mutable_data_ptr(), CUDA_R_32F, static_cast<int>(weights.size(0)),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "Eclair cuBLAS GEMM status: ", status);
    return output;
}

// One pass from FP32 accumulators to binary16, replacing separate multiply,
// add and conversion passes with identical results. The FP32 temporary is
// released to the allocator when the caller's last reference drops.
at::Tensor fp16_epilogue(const at::Tensor& accumulators, const at::Tensor& scale,
                         const std::optional<at::Tensor>& bias) {
    auto output = at::empty(accumulators.sizes(), accumulators.options().dtype(at::kHalf));
    const auto elements = static_cast<std::size_t>(accumulators.numel());
    const ops::LinearEpilogueShape shape{static_cast<std::size_t>(accumulators.size(0)),
                                         static_cast<std::size_t>(accumulators.size(1))};
    const ops::LinearEpilogueBuffers buffers{
        {accumulators.const_data_ptr<float>(), elements},
        {scale.const_data_ptr<float>(), 1U},
        {bias ? bias->const_data_ptr<float>() : nullptr,
         bias ? static_cast<std::size_t>(bias->numel()) : 0U},
        {reinterpret_cast<std::uint16_t*>(output.mutable_data_ptr<at::Half>()), elements}};
    const auto status =
        cuda::launch_linear_epilogue(shape, buffers, at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(status == cudaSuccess, "Eclair FP16 epilogue launch: ", cudaGetErrorString(status));
    return output;
}

// W8A16 with a scalar post-accumulation weight scale. Activations stay FP16;
// unlike scaled_mm there is no FP8 activation quantization. This is separate
// from fp8_linear's block scales, which round into FP16 before multiplication.
// The FP32 result is the default. FP16 output is the same value rounded once
// to binary16, produced without a separate conversion pass.
at::Tensor w8a16_mm(const at::Tensor& a, const at::Tensor& weights, const at::Tensor& scale,
                    const std::optional<at::Tensor>& bias, std::int64_t workspace_bytes,
                    std::optional<at::ScalarType> out_dtype) {
    check_policy();
    check_tensor(a, a, "activations");
    check_tensor(weights, a, "weights");
    check_tensor(scale, a, "scale");
    TORCH_CHECK(a.dim() == 2 && weights.dim() == 2 && a.size(1) == weights.size(1),
                "Eclair W8A16 requires A[M,K] and weights[N,K]");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "Eclair W8A16 requires FP16 activations");
    static_cast<void>(fp8_format(weights.scalar_type()));
    TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 1,
                "Eclair W8A16 requires a scalar FP32 scale");
    TORCH_CHECK(workspace_bytes >= 0, "Eclair W8A16 workspace bytes must be nonnegative");
    TORCH_CHECK(!out_dtype || *out_dtype == at::kFloat || *out_dtype == at::kHalf,
                "Eclair W8A16 out_dtype must be FP32 or FP16");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && weights.size(0) > 0 &&
                    a.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max() &&
                    weights.size(0) <= std::numeric_limits<int>::max(),
                "Eclair W8A16 requires nonempty signed-int dimensions");
    if (bias) {
        check_tensor(*bias, a, "bias");
        TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->dim() == 1 &&
                        bias->size(0) == weights.size(0),
                    "Eclair W8A16 bias must be FP32 [N]");
    }
    const c10::cuda::CUDAGuard guard(a.device());
    check_device_and_stream();
    at::Tensor output;
    if (a.size(0) > 4 && static_cast<std::size_t>(weights.numel()) <=
                             static_cast<std::size_t>(workspace_bytes) / sizeof(at::Half)) {
        output = expanded_mm(a, weights);
    } else {
        auto one = at::ones({}, a.options().dtype(at::kFloat));
        output = fp8_linear(a, weights, one, weights.size(0), weights.size(1), std::nullopt);
    }
    if (out_dtype == at::kHalf) {
        return fp16_epilogue(output, scale, bias);
    }
    output.mul_(scale.reshape({}));
    if (bias) {
        output.add_(*bias);
    }
    return output;
}

#if ECLAIR_TORCH_ATEN_OVERRIDE
// Explicit opt-in replacement for a bounded subset of torch 2.7.1's operator.
// FP8 A/B widen exactly to FP16. Scalar FP32 scales apply after accumulation,
// unlike the block-scaled W8A16 contract above. Never fold them into FP16 W.
at::Tensor scaled_mm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& scale_a,
                     const at::Tensor& scale_b, const std::optional<at::Tensor>& bias,
                     const std::optional<at::Tensor>& scale_result,
                     std::optional<at::ScalarType> out_dtype, bool use_fast_accum) {
    check_policy();
    check_tensor(a, a, "A");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "Eclair scaled_mm requires A[M,K] and B[K,N]");
    const auto weights = b.transpose(0, 1);
    check_tensor(weights, a, "B transpose");
    static_cast<void>(fp8_format(a.scalar_type()));
    static_cast<void>(fp8_format(b.scalar_type()));
    check_tensor(scale_a, a, "scale_a");
    check_tensor(scale_b, a, "scale_b");
    TORCH_CHECK(scale_a.scalar_type() == at::kFloat && scale_b.scalar_type() == at::kFloat &&
                    scale_a.numel() == 1 && scale_b.numel() == 1,
                "Eclair scaled_mm supports scalar FP32 scales only");
    TORCH_CHECK(!scale_result && !use_fast_accum,
                "Eclair scaled_mm does not support scale_result or fast accumulation");
    TORCH_CHECK(out_dtype && (*out_dtype == at::kFloat || *out_dtype == at::kHalf ||
                              *out_dtype == at::kBFloat16),
                "Eclair scaled_mm requires explicit FP32, FP16 or BF16 out_dtype");
    if (bias) {
        check_tensor(*bias, a, "bias");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == b.size(1) &&
                        (bias->scalar_type() == at::kFloat || bias->scalar_type() == *out_dtype),
                    "Eclair scaled_mm bias must be [N] with FP32 or output dtype");
    }
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0 &&
                    a.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max() &&
                    b.size(1) <= std::numeric_limits<int>::max(),
                "Eclair scaled_mm requires nonempty signed-int dimensions");
    const auto budget = workspace_budget();
    const c10::cuda::CUDAGuard guard(a.device());
    check_device_and_stream();
    // No stream or allocation is hidden from PyTorch's allocator. Temporary
    // tensors remain owned through their queued uses on the current stream.
    auto half_a = a.to(at::kHalf);
    at::Tensor output;
    if (a.size(0) > 4 && static_cast<std::size_t>(weights.numel()) <= budget / sizeof(at::Half)) {
        output = expanded_mm(half_a, weights);
    } else {
        auto one = at::ones({}, a.options().dtype(at::kFloat));
        output = fp8_linear(half_a, weights, one, b.size(1), a.size(1), std::nullopt);
    }
    output.mul_(scale_a.reshape({}));
    output.mul_(scale_b.reshape({}));
    if (bias) {
        output.add_(*bias);
    }
    return output.to(*out_dtype);
}
#endif

}  // namespace
}  // namespace eclair::pytorch

TORCH_LIBRARY(eclair, library) {
    library.def("adapter_info() -> str", []() -> std::string {
#if ECLAIR_TORCH_ATEN_OVERRIDE
        return "eclair.pytorch/3;aten-override";
#else
        return "eclair.pytorch/3;operators-only";
#endif
    });
    library.def(
        "w8a16_mm(Tensor a, Tensor weights, Tensor scale, Tensor? bias=None, "
        "int workspace_bytes=67108864, ScalarType? out_dtype=None) -> Tensor");
    library.def(
        "fp8_linear(Tensor a, Tensor weights, Tensor scales, int block_n, int block_k, "
        "Tensor? bias=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(eclair, CUDA, library) {
    library.impl("w8a16_mm", TORCH_FN(eclair::pytorch::w8a16_mm));
    library.impl("fp8_linear", TORCH_FN(eclair::pytorch::fp8_linear));
}
#if ECLAIR_TORCH_ATEN_OVERRIDE
TORCH_LIBRARY_IMPL(aten, CUDA, library) {
    library.impl("_scaled_mm", TORCH_FN(eclair::pytorch::scaled_mm));
}
#endif
