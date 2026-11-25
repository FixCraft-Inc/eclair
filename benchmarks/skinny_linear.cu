/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "../tests/check.hpp"
#include "../tests/cublas_support.hpp"
#include "../tests/cuda_support.hpp"

#include "eclair/cuda/bf16_linear.hpp"
#include "eclair/cuda/linear_epilogue.hpp"
#include "eclair/formats/fp16.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using eclair::test::checked;
using eclair::test::checked_blas;
using eclair::test::DeviceBuffer;

// Exact binary16 values, independent of device random generators and seeds.
__global__ void fill_weights(std::uint16_t* weights, std::size_t count) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        weights[i] = __half_as_ushort(
            __float2half_rn(static_cast<float>(static_cast<int>(i % 31U) - 15) / 16.0F));
    }
}

void run() {
    eclair::test::print_device_header();
    eclair::test::Blas blas;
    eclair::test::Stream stream;
    checked_blas(cublasSetStream(blas.get(), stream.get()), "cuBLAS stream");
    int version = 0;
    checked_blas(cublasGetVersion(blas.get(), &version), "cuBLAS version");
    std::printf(
        "cuBLAS=%d rows=1..4 FP16 A/W/output; Eclair zero-shift BF16 linear + "
        "FP16 epilogue; cuBLAS GemmEx COMPUTE_32F DEFAULT_TENSOR_OP; "
        "nine CUDA-event samples, 100ms warmup per path, event synchronization; "
        "FP64 sampled oracle, exact dyadic inputs; utilization not measured\n",
        version);
    struct Shape {
        std::size_t n;
        std::size_t k;
    };
    for (const Shape shape : {Shape{5120, 6144}, Shape{248320, 5120}}) {
        const auto n = shape.n, k = shape.k;
        DeviceBuffer<std::uint16_t> weights(n * k);
        fill_weights<<<4096, 256, 0, stream.get()>>>(weights.data(), n * k);
        checked(cudaGetLastError(), "weight initialization");
        const DeviceBuffer<std::int32_t> ws(std::vector<std::int32_t>(n, 0), stream.get());
        const DeviceBuffer<std::int32_t> as(std::vector<std::int32_t>(4U, 0), stream.get());
        const DeviceBuffer<float> scale(std::vector<float>{1.0F}, stream.get());
        for (std::size_t m = 1; m <= 4U; ++m) {
            std::vector<std::uint16_t> a(m * k);
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] = eclair::formats::encode_fp16(
                    static_cast<float>(static_cast<int>(i % 17U) - 8) / 8.0F);
            }
            const DeviceBuffer<std::uint16_t> da(a, stream.get());
            DeviceBuffer<float> sums(m * n);
            DeviceBuffer<std::uint16_t> actual(m * n), baseline(m * n);
            const eclair::ops::Bf16LinearBuffers buffers{
                {da.data(), m * k},  {as.data(), m}, {weights.data(), n * k}, {ws.data(), n}, {},
                {sums.data(), m * n}};
            const eclair::ops::LinearEpilogueBuffers epilogue{
                {sums.data(), m * n}, {scale.data(), 1}, {}, {actual.data(), m * n}};
            const auto eclair = [&] {
                checked(eclair::cuda::launch_bf16_linear({m, n, k}, buffers, stream.get()),
                        "skinny linear");
                checked(eclair::cuda::launch_linear_epilogue({m, n}, epilogue, stream.get()),
                        "FP16 epilogue");
            };
            const float alpha = 1.0F, beta = 0.0F;
            const auto cublas = [&] {
                checked_blas(cublasGemmEx(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n),
                                          static_cast<int>(m), static_cast<int>(k), &alpha,
                                          weights.data(), CUDA_R_16F, static_cast<int>(k),
                                          da.data(), CUDA_R_16F, static_cast<int>(k), &beta,
                                          baseline.data(), CUDA_R_16F, static_cast<int>(n),
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                             "FP16 cuBLAS");
            };
            eclair();
            cublas();
            const auto got = actual.download(stream.get()), ref = baseline.download(stream.get());
            for (std::size_t row = 0; row < m; ++row) {
                for (const std::size_t channel : {std::size_t{0}, n / 3U, n / 2U, n - 1U}) {
                    double sum = 0;
                    for (std::size_t column = 0; column < k; ++column) {
                        const double w = static_cast<double>(
                                             static_cast<int>((channel * k + column) % 31U) - 15) /
                                         16.0;
                        sum +=
                            static_cast<double>(eclair::formats::decode_fp16(a[row * k + column])) *
                            w;
                    }
                    const auto expected = eclair::formats::encode_fp16(static_cast<float>(sum));
                    ECLAIR_CHECK(got[row * n + channel] == expected);
                    ECLAIR_CHECK(ref[row * n + channel] == expected);
                }
            }
            const auto et = eclair::test::timings(stream.get(), eclair);
            const auto bt = eclair::test::timings(stream.get(), cublas);
            const double bytes = static_cast<double>(n * k * sizeof(std::uint16_t));
            std::printf(
                "M=%zu N=%zu K=%zu eclair_ms=%.6f cublas_ms=%.6f "
                "eclair_weight_GBs=%.3f cublas_weight_GBs=%.3f max_eclair_ms=%.6f "
                "max_cublas_ms=%.6f sampled_FP64=pass\n",
                m, n, k, et[4], bt[4], bytes / (et[4] * 1e6), bytes / (bt[4] * 1e6), et.back(),
                bt.back());
        }
    }
}

}  // namespace

int main() {
    return eclair::test::contain(run);
}
