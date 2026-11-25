/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "cublas_support.hpp"
#include "cuda_support.hpp"

#include "eclair/cuda/fp8_linear.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/fp8_linear.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using eclair::formats::Fp8Format;
using eclair::ops::Fp8LinearBuffers;
using eclair::ops::Fp8LinearShape;
using eclair::test::Blas;
using eclair::test::checked;
using eclair::test::checked_blas;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;
using eclair::test::timings;

[[nodiscard]] bool close(float actual, float expected) {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return actual == expected;
    }
    return std::isfinite(actual) &&
           std::abs(actual - expected) <= 2.0e-4F + 2.0e-4F * std::abs(expected);
}

// Inserts `misalign` leading elements so the device views of A and W miss the
// kernels' vector alignment and exercise the scalar load paths.
template <typename T>
[[nodiscard]] std::vector<T> padded(const std::vector<T>& values, std::size_t misalign) {
    std::vector<T> result(misalign, T{});
    result.insert(result.end(), values.begin(), values.end());
    return result;
}

void check_case(const Fp8LinearShape& shape, const std::vector<std::uint16_t>& a,
                const std::vector<std::uint8_t>& w, const std::vector<float>& scales,
                const std::vector<float>& bias, bool compare_blas, std::size_t misalign = 0U) {
    std::vector<float> reference(shape.m * shape.n);
    const Fp8LinearBuffers host{{a.data(), a.size()},
                                {w.data(), w.size()},
                                {scales.data(), scales.size()},
                                {bias.data(), bias.size()},
                                {reference.data(), reference.size()}};
    ECLAIR_CHECK(eclair::host::fp8_linear_reference(shape, host) == eclair::ops::Status::Success);
    Stream stream;
    DeviceBuffer<std::uint16_t> da(a.size() + misalign);
    DeviceBuffer<std::uint8_t> dw(w.size() + misalign);
    DeviceBuffer<float> ds(scales.size());
    DeviceBuffer<float> db(bias.size());
    DeviceBuffer<float> result(reference.size() + 2U);
    da.upload(padded(a, misalign), stream.get());
    dw.upload(padded(w, misalign), stream.get());
    ds.upload(scales, stream.get());
    db.upload(bias, stream.get());
    std::vector<float> sentinels(result.size(), -12345.0F);
    result.upload(sentinels, stream.get());
    Fp8LinearBuffers device{{da.data() + misalign, a.size()},
                            {dw.data() + misalign, w.size()},
                            {ds.data(), ds.size()},
                            {db.data(), db.size()},
                            {result.data() + 1, reference.size()}};
    checked(eclair::cuda::launch_fp8_linear(shape, device, stream.get()), "FP8 linear launch");
    const auto output = result.download(stream.get());
    ECLAIR_CHECK(output.front() == -12345.0F && output.back() == -12345.0F);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!close(output[i + 1U], reference[i])) {
            std::fprintf(stderr, "M=%zu N=%zu K=%zu misalign=%zu index=%zu actual=%g expected=%g\n",
                         shape.m, shape.n, shape.k, misalign, i, output[i + 1U], reference[i]);
            ECLAIR_CHECK(false && "WMMA differs from host oracle");
            break;
        }
    }
    // Rejection must enqueue no kernel and preserve the successful output.
    auto invalid = device;
    invalid.weights.size = 0;
    ECLAIR_CHECK(eclair::cuda::launch_fp8_linear(shape, invalid, stream.get()) ==
                 cudaErrorInvalidValue);
    invalid = device;
    invalid.scales = {device.output.data, device.output.size};
    ECLAIR_CHECK(eclair::cuda::launch_fp8_linear(shape, invalid, stream.get()) ==
                 cudaErrorInvalidValue);
    const auto after_rejection = result.download(stream.get());
    for (std::size_t i = 0; i < output.size(); ++i) {
        ECLAIR_CHECK(std::isnan(output[i]) ? std::isnan(after_rejection[i])
                                           : output[i] == after_rejection[i]);
    }
    if (!compare_blas) {
        return;
    }
    std::vector<std::uint16_t> expanded(w.size());
    const std::size_t scale_columns = 1U + (shape.k - 1U) / shape.block_k;
    for (std::size_t n = 0; n < shape.n; ++n) {
        for (std::size_t k = 0; k < shape.k; ++k) {
            expanded[n * shape.k + k] = eclair::formats::encode_fp16(
                eclair::formats::decode_fp8(w[n * shape.k + k], shape.format) *
                scales[(n / shape.block_n) * scale_columns + k / shape.block_k]);
        }
    }
    DeviceBuffer<std::uint16_t> expanded_device(expanded.size());
    expanded_device.upload(expanded, stream.get());
    Blas blas;
    checked_blas(cublasSetStream(blas.get(), stream.get()), "cuBLAS stream");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    checked_blas(cublasGemmEx(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(shape.n),
                              static_cast<int>(shape.m), static_cast<int>(shape.k), &alpha,
                              expanded_device.data(), CUDA_R_16F, static_cast<int>(shape.k),
                              device.activations.data, CUDA_R_16F, static_cast<int>(shape.k), &beta,
                              device.output.data, CUDA_R_32F, static_cast<int>(shape.n),
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cuBLAS GEMM");
    const auto baseline = result.download(stream.get());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float value = baseline[i + 1U] + (bias.empty() ? 0.0F : bias[i % shape.n]);
        ECLAIR_CHECK(close(value, reference[i]));
    }
}

void check_finite(Fp8LinearShape shape, bool with_bias, std::size_t misalign = 0U) {
    eclair::ops::Fp8LinearSizes sizes;
    ECLAIR_CHECK(eclair::ops::fp8_linear_sizes(shape, sizes) == eclair::ops::Status::Success);
    std::vector<std::uint16_t> a(sizes.activations);
    std::vector<std::uint8_t> w(sizes.weights);
    std::vector<float> scales(sizes.scales);
    std::vector<float> bias(with_bias ? shape.n : 0U);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = eclair::formats::encode_fp16(static_cast<float>(static_cast<int>(i % 31U) - 15) /
                                            16.0F);
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = eclair::formats::encode_fp8(static_cast<float>(static_cast<int>(i % 17U) - 8) / 4.0F,
                                           shape.format);
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = std::ldexp(1.0F, static_cast<int>(i % 4U) - 2);
    }
    for (std::size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(i % 7U) / 8.0F;
    }
    check_case(shape, a, w, scales, bias, true, misalign);
}

// A NaN block scale poisons exactly the outputs whose dot product uses that
// block. Zero padding past K must not combine with it.
void check_nan_block_scale(Fp8Format format) {
    const Fp8LinearShape shape{5, 32, 40, 16, 16, format};
    std::vector<std::uint16_t> a(shape.m * shape.k);
    std::vector<std::uint8_t> w(shape.n * shape.k);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] =
            eclair::formats::encode_fp16(static_cast<float>(static_cast<int>(i % 9U) - 4) / 8.0F);
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] =
            eclair::formats::encode_fp8(static_cast<float>(static_cast<int>(i % 5U) - 2), format);
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> scales{1.0F, nan, 0.5F, 2.0F, 1.0F, 0.25F};
    check_case(shape, a, w, scales, {}, false);
    std::vector<float> reference(shape.m * shape.n);
    const Fp8LinearBuffers host{{a.data(), a.size()},
                                {w.data(), w.size()},
                                {scales.data(), scales.size()},
                                {},
                                {reference.data(), reference.size()}};
    ECLAIR_CHECK(eclair::host::fp8_linear_reference(shape, host) == eclair::ops::Status::Success);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        ECLAIR_CHECK(std::isnan(reference[i]) == (i % shape.n < 16U));
    }
}

// Every byte pattern at every position of the kernels' vector loads, with a
// unit scale (exact decode shortcut) and a non-unit scale (rounded product).
// Row m of A selects column m % K, so output[m,n] is W[n, m % K] times its
// scale. The oracle also propagates NaN from 0 * NaN in unselected columns.
void check_vector_representations(Fp8Format format) {
    // K=4 takes the CUDA-core 4-byte loads, K=16 with M=4 its 16-byte loads
    // and K=16 with M=17 the tensor-core tile loads.
    struct Size {
        std::size_t m;
        std::size_t k;
    };
    for (const Size size : {Size{4, 4}, Size{4, 16}, Size{17, 16}}) {
        const std::size_t m = size.m;
        const std::size_t k = size.k;
        std::vector<std::uint16_t> a(m * k, 0U);
        for (std::size_t row = 0; row < m; ++row) {
            a[row * k + row % k] = 0x3C00U;
        }
        std::vector<std::uint8_t> w(256U * k);
        for (std::size_t n = 0; n < 256U; ++n) {
            for (std::size_t column = 0; column < k; ++column) {
                w[n * k + column] = static_cast<std::uint8_t>((n + column * (256U / k)) % 256U);
            }
        }
        for (const float scale : {1.0F, 0.375F}) {
            check_case({m, 256, k, 256, k, format}, a, w, {scale}, {}, false);
        }
        // Channel scales mix unit and non-unit rows in one launch.
        std::vector<float> channel(256U);
        for (std::size_t n = 0; n < channel.size(); ++n) {
            const float values[] = {1.0F, 0.375F, 3.0F, std::ldexp(1.0F, -10)};
            channel[n] = values[n % 4U];
        }
        check_case({m, 256, k, 1, k, format}, a, w, channel, {}, false);
    }
}

void benchmark() {
    Blas blas;
    int version = 0;
    checked_blas(cublasGetVersion(blas.get(), &version), "cuBLAS version");
    std::printf(
        "BENCH cuBLAS=%d warmup=100ms-per-path samples=9 stream=nonblocking graphs=off "
        "timing=CUDA-events synchronization=event precision=FP16/FP32\n",
        version);
    std::puts(
        "BENCH baseline=cuBLAS-resident-FP16; excludes weight expansion and uploads; "
        "Eclair includes FP8 decode/scale; workspace=0; utilization not measured");
    struct BenchmarkShape {
        std::size_t m;
        std::size_t n;
        std::size_t k;
    };
    // 1024x1024 keeps the recorded history comparable. 4096x4096 weights are
    // 16 MiB in FP8, large enough that DRAM traffic rather than launch
    // latency dominates the decode batches.
    const BenchmarkShape shapes[] = {{1, 1024, 1024},   {16, 1024, 1024},  {128, 1024, 1024},
                                     {1, 4096, 4096},   {4, 4096, 4096},   {16, 4096, 4096},
                                     {128, 4096, 4096}, {1024, 4096, 4096}};
    for (const BenchmarkShape& size : shapes) {
        const Fp8LinearShape shape{size.m, size.n, size.k, size.n, size.k, Fp8Format::E4M3FN};
        Stream stream;
        checked_blas(cublasSetStream(blas.get(), stream.get()), "benchmark stream");
        std::vector<std::uint16_t> a(shape.m * shape.k);
        std::vector<std::uint8_t> w(shape.n * shape.k);
        std::vector<std::uint16_t> expanded(w.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = eclair::formats::encode_fp16(static_cast<float>(static_cast<int>(i % 31U) - 15) /
                                                16.0F);
        }
        for (std::size_t i = 0; i < w.size(); ++i) {
            const float value = static_cast<float>(static_cast<int>(i % 17U) - 8) / 4.0F;
            w[i] = eclair::formats::encode_fp8(value, shape.format);
            expanded[i] = eclair::formats::encode_fp16(value);
        }
        DeviceBuffer<std::uint16_t> da(a.size());
        DeviceBuffer<std::uint8_t> dw(w.size());
        DeviceBuffer<std::uint16_t> de(expanded.size());
        DeviceBuffer<float> ds(1);
        DeviceBuffer<float> result(shape.m * shape.n);
        DeviceBuffer<float> baseline(shape.m * shape.n);
        da.upload(a, stream.get());
        dw.upload(w, stream.get());
        de.upload(expanded, stream.get());
        ds.upload({1.0F}, stream.get());
        const Fp8LinearBuffers device{{da.data(), da.size()},
                                      {dw.data(), dw.size()},
                                      {ds.data(), 1},
                                      {},
                                      {result.data(), result.size()}};
        const auto fused = [&] {
            checked(eclair::cuda::launch_fp8_linear(shape, device, stream.get()),
                    "benchmark linear");
        };
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const auto resident = [&] {
            checked_blas(
                cublasGemmEx(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(shape.n),
                             static_cast<int>(shape.m), static_cast<int>(shape.k), &alpha,
                             de.data(), CUDA_R_16F, static_cast<int>(shape.k), da.data(),
                             CUDA_R_16F, static_cast<int>(shape.k), &beta, baseline.data(),
                             CUDA_R_32F, static_cast<int>(shape.n), CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "benchmark cuBLAS");
        };
        fused();
        resident();
        const auto actual = result.download(stream.get());
        const auto expected = baseline.download(stream.get());
        float max_error = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            ECLAIR_CHECK(close(actual[i], expected[i]));
            max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
        }
        if (eclair::test::failure_count() != 0) {
            throw std::runtime_error("benchmark correctness failed");
        }
        const auto fused_times = timings(stream.get(), fused);
        const auto baseline_times = timings(stream.get(), resident);
        const double operations = 2.0 * static_cast<double>(shape.m) *
                                  static_cast<double>(shape.n) * static_cast<double>(shape.k);
        std::printf(
            "BENCH M=%zu N=%zu K=%zu E4M3FN tensor-scale FP8_weight_bytes=%zu "
            "FP16_baseline_weight_bytes=%zu max_abs_error=%g fused_median_us=%.3f "
            "fused_p95_us=%.3f baseline_median_us=%.3f baseline_p95_us=%.3f "
            "fused_TFLOPS=%.5f baseline_TFLOPS=%.5f\n",
            shape.m, shape.n, shape.k, w.size(), expanded.size() * sizeof(std::uint16_t), max_error,
            fused_times[4] * 1000.0F, fused_times.back() * 1000.0F, baseline_times[4] * 1000.0F,
            baseline_times.back() * 1000.0F,
            operations / (static_cast<double>(fused_times[4]) * 1.0e9),
            operations / (static_cast<double>(baseline_times[4]) * 1.0e9));
    }
}

void run() {
    eclair::test::print_device_header();
    std::puts(
        "FP8 decode/scale -> FP16 RNE; WMMA FP32 output; nonblocking stream; stream synchronize; "
        "atol=2e-4 rtol=2e-4; baseline=cuBLAS GemmEx FP16/FP32");
    checked(eclair::cuda::check_fp8_linear_image(0), "linear kernel image");
    ECLAIR_CHECK(eclair::cuda::check_fp8_linear_image(-1) == cudaErrorInvalidValue);
    ECLAIR_CHECK(eclair::cuda::launch_fp8_linear({}, {}, nullptr) == cudaSuccess);
    for (const auto format : {Fp8Format::E4M3FN, Fp8Format::E5M2}) {
        check_finite({17, 19, 33, 7, 13, format}, true);
        check_finite({1, 257, 129, 128, 128, format}, false);
        check_finite({4, 19, 33, 7, 13, format}, true);
        check_finite({32, 48, 64, 1, 64, format}, true);
        check_finite({16, 32, 48, 32, 48, format}, false);
        // CUDA-core path with vector loads, then the same shape misaligned.
        check_finite({1, 64, 128, 64, 128, format}, true);
        check_finite({3, 70, 256, 1, 256, format}, false);
        check_finite({4, 33, 1024, 33, 1024, format}, true);
        check_finite({4, 33, 1024, 33, 1024, format}, true, 1U);
        // CUDA-core block scales: shared per 4-column group, per column with
        // block_k % 4 != 0, and on the 4-byte path.
        check_finite({2, 40, 96, 8, 32, format}, true);
        check_finite({3, 21, 64, 5, 6, format}, false);
        check_finite({4, 9, 20, 2, 3, format}, true);
        // Tensor-core tiles: first M past the CUDA-core bound, several K
        // stages with a partial last stage, partial edge tiles and every
        // scale kind, with aligned and misaligned views.
        check_finite({5, 64, 64, 64, 64, format}, true);
        check_finite({65, 130, 200, 16, 64, format}, true);
        check_finite({100, 67, 96, 67, 96, format}, false);
        check_finite({100, 67, 96, 67, 96, format}, false, 1U);
        check_finite({257, 300, 520, 1, 520, format}, true);
        check_finite({129, 64, 1024, 64, 1024, format}, true, 3U);
        check_nan_block_scale(format);
        std::vector<std::uint8_t> w(256);
        for (std::size_t i = 0; i < w.size(); ++i) {
            w[i] = static_cast<std::uint8_t>(i);
        }
        check_case({1, 256, 1, 256, 1, format}, {0x3C00U}, w, {1.0F}, {}, false);
        check_case({17, 256, 1, 256, 1, format}, std::vector<std::uint16_t>(17U, 0x3C00U), w,
                   {1.0F}, {}, false);
        check_case({1, 256, 1, 256, 1, format}, {0x3C00U}, w, {0.375F}, {}, false);
        check_case({17, 256, 1, 256, 1, format}, std::vector<std::uint16_t>(17U, 0x3C00U), w,
                   {0.375F}, {}, false);
        check_vector_representations(format);
    }
    for (const float scale :
         {std::ldexp(1.0F, -24), std::ldexp(1.0F, -25), 1.0F + std::ldexp(1.0F, -11), 256.0F,
          std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        check_case({1, 1, 1, 1, 1, Fp8Format::E4M3FN}, {0x3C00U}, {0x7EU}, {scale}, {}, false);
        check_case({5, 1, 1, 1, 1, Fp8Format::E4M3FN}, std::vector<std::uint16_t>(5U, 0x3C00U),
                   {0x7EU}, {scale}, {}, false);
    }
    for (const std::uint16_t a : {std::uint16_t{0x0001U}, std::uint16_t{0x8000U},
                                  std::uint16_t{0x7C00U}, std::uint16_t{0x7E00U}}) {
        check_case({1, 1, 1, 1, 1, Fp8Format::E4M3FN}, {a}, {0x38U}, {1.0F}, {}, false);
        check_case({5, 1, 1, 1, 1, Fp8Format::E4M3FN}, std::vector<std::uint16_t>(5U, a), {0x38U},
                   {1.0F}, {}, false);
    }
    std::puts(
        "checked: irregular, decode, channel and tensor shapes; CUDA-core and tensor-core paths; "
        "vector and misaligned loads; multi-stage partial tiles; all FP8 representations; "
        "subnormal, signed-zero, NaN, infinity, NaN block scale, overflow, rounding, guards "
        "and rejection");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--benchmark")) {
        std::fprintf(stderr, "Usage: eclair_cuda_fp8_linear_tests [--benchmark]\n");
        return 2;
    }
    return eclair::test::contain([argc] {
        run();
        if (argc == 2 && eclair::test::failure_count() == 0) {
            benchmark();
        }
    });
}
