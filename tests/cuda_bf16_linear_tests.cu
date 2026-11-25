/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "cublas_support.hpp"
#include "cuda_support.hpp"

#include "eclair/cuda/bf16_linear.hpp"
#include "eclair/formats/bf16.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/bf16_linear.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using eclair::ops::Bf16LinearBuffers;
using eclair::ops::Bf16LinearShape;
using eclair::ops::Bf16RowScalingBuffers;
using eclair::ops::Bf16RowScalingShape;
using eclair::ops::Status;
using eclair::test::Blas;
using eclair::test::checked;
using eclair::test::checked_blas;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;
using eclair::test::timings;

constexpr std::uint16_t kHalfSentinel = 0xA5A5U;
constexpr std::int32_t kShiftSentinel = 0x5A5A5A5A;
constexpr float kFloatSentinel = -12345.0F;

// Row-major storage of a matrix written one row per line.
template <std::size_t Rows, std::size_t Columns>
[[nodiscard]] std::vector<std::uint16_t> flatten(
    const std::array<std::array<std::uint16_t, Columns>, Rows>& rows) {
    std::vector<std::uint16_t> values;
    for (const auto& row : rows) {
        values.insert(values.end(), row.begin(), row.end());
    }
    return values;
}

struct ScaledMatrix {
    std::vector<std::uint16_t> values;
    std::vector<std::int32_t> shifts;
};

[[nodiscard]] ScaledMatrix host_scale(const std::vector<std::uint16_t>& matrix, std::size_t rows,
                                      std::size_t columns) {
    ScaledMatrix scaled{std::vector<std::uint16_t>(matrix.size()), std::vector<std::int32_t>(rows)};
    const Bf16RowScalingBuffers buffers{{matrix.data(), matrix.size()},
                                        {scaled.values.data(), scaled.values.size()},
                                        {scaled.shifts.data(), scaled.shifts.size()}};
    ECLAIR_CHECK(eclair::host::bf16_row_scaling_reference({rows, columns}, buffers) ==
                 Status::Success);
    return scaled;
}

// `offset` leading elements, so a device view built `offset` elements in
// misses the kernels' 16-byte alignment and takes their scalar paths.
template <typename T>
[[nodiscard]] std::vector<T> padded(const std::vector<T>& values, std::size_t offset, T fill) {
    std::vector<T> result(offset, fill);
    result.insert(result.end(), values.begin(), values.end());
    return result;
}

// A row-scaled matrix on the device, `offset` elements into its buffers, with
// one sentinel after the values and around the shifts.
struct DeviceScaled {
    DeviceBuffer<std::uint16_t> values;
    DeviceBuffer<std::int32_t> shifts;
    std::size_t offset;
    std::size_t count;
    std::size_t rows;

    [[nodiscard]] std::uint16_t* value_data() const { return values.data() + offset; }
    [[nodiscard]] std::int32_t* shift_data() const { return shifts.data() + 1; }
};

// Row-scales `matrix` on the GPU and checks every value, shift and sentinel
// against the host reference bit for bit. Rejected launches must not write.
void scale_on_device(const std::vector<std::uint16_t>& matrix, std::size_t rows,
                     std::size_t columns, std::size_t offset, cudaStream_t stream,
                     DeviceScaled& target) {
    const ScaledMatrix expected = host_scale(matrix, rows, columns);
    const DeviceBuffer<std::uint16_t> input(padded(matrix, offset, std::uint16_t{0}), stream);
    const Bf16RowScalingBuffers buffers{{input.data() + offset, matrix.size()},
                                        {target.value_data(), matrix.size()},
                                        {target.shift_data(), rows}};
    checked(eclair::cuda::launch_bf16_row_scaling({rows, columns}, buffers, stream), "row scaling");
    auto invalid = buffers;
    invalid.output = {const_cast<std::uint16_t*>(invalid.input.data), invalid.input.size};
    ECLAIR_CHECK(eclair::cuda::launch_bf16_row_scaling({rows, columns}, invalid, stream) ==
                 cudaErrorInvalidValue);
    invalid = buffers;
    invalid.shifts.size = rows - 1U;
    ECLAIR_CHECK(eclair::cuda::launch_bf16_row_scaling({rows, columns}, invalid, stream) ==
                 cudaErrorInvalidValue);
    const auto values = target.values.download(stream);
    const auto shifts = target.shifts.download(stream);
    bool exact = values[offset + matrix.size()] == kHalfSentinel &&
                 shifts.front() == kShiftSentinel && shifts.back() == kShiftSentinel;
    for (std::size_t i = 0; exact && i < offset; ++i) {
        exact = values[i] == kHalfSentinel;
    }
    for (std::size_t i = 0; exact && i < matrix.size(); ++i) {
        exact = values[offset + i] == expected.values[i];
    }
    for (std::size_t row = 0; exact && row < rows; ++row) {
        exact = shifts[1U + row] == expected.shifts[row];
    }
    if (!exact) {
        std::fprintf(stderr, "rows=%zu columns=%zu offset=%zu\n", rows, columns, offset);
        ECLAIR_CHECK(false && "GPU row scaling differs from the host reference");
    }
}

[[nodiscard]] DeviceScaled scaled_buffers(std::size_t rows, std::size_t columns, std::size_t offset,
                                          cudaStream_t stream) {
    const std::size_t count = rows * columns;
    return {
        DeviceBuffer<std::uint16_t>(std::vector<std::uint16_t>(offset + count + 1U, kHalfSentinel),
                                    stream),
        DeviceBuffer<std::int32_t>(std::vector<std::int32_t>(rows + 2U, kShiftSentinel), stream),
        offset, count, rows};
}

void check_row_scaling(const std::vector<std::uint16_t>& matrix, std::size_t rows,
                       std::size_t columns, std::size_t offset) {
    Stream stream;
    DeviceScaled target = scaled_buffers(rows, columns, offset, stream.get());
    scale_on_device(matrix, rows, columns, offset, stream.get(), target);
}

// Every BF16 pattern, permuted so each row mixes magnitudes, infinities and
// NaNs, in several row lengths and on both the vector and scalar paths.
void check_row_scaling_patterns() {
    std::vector<std::uint16_t> patterns(65536U);
    for (std::size_t i = 0; i < patterns.size(); ++i) {
        patterns[i] = static_cast<std::uint16_t>((i * 40503U) & 0xFFFFU);
    }
    check_row_scaling(patterns, 256, 256, 0);
    check_row_scaling(patterns, 256, 256, 1);
    check_row_scaling(patterns, 4096, 16, 0);
    check_row_scaling(patterns, 8192, 8, 3);
    // More rows than resident warps, so the grid-stride loop wraps.
    check_row_scaling(patterns, 65536, 1, 0);
    std::vector<std::uint16_t> long_rows(40000U * 8U);
    for (std::size_t i = 0; i < long_rows.size(); ++i) {
        long_rows[i] = patterns[(i * 7U) % patterns.size()];
    }
    check_row_scaling(long_rows, 40000, 8, 0);
    for (const std::size_t columns : {1U, 7U, 8U, 9U, 15U, 16U, 17U, 255U, 256U, 257U, 1000U}) {
        const std::vector<std::uint16_t> matrix(
            patterns.begin(), patterns.begin() + static_cast<std::ptrdiff_t>(3U * columns));
        check_row_scaling(matrix, 3, columns, 0);
        check_row_scaling(matrix, 3, columns, 2);
    }
}

// Tensor cores add exact products after aligning them to the largest
// exponent, truncating what falls below roughly 23 bits. They do not normalize
// binary16 subnormal inputs first, so a subnormal operand aligns as if its
// exponent were -14 and can shrink the window for the other products. Each
// nonzero subnormal therefore weighs 2^-14 in the error bound. check_alignment
// measures both effects on the device under test.
[[nodiscard]] double alignment_magnitude(std::uint16_t bits) {
    const double value = std::fabs(static_cast<double>(eclair::formats::decode_fp16(bits)));
    return value != 0.0 && value < 0x1p-14 ? 0x1p-14 : value;
}

// FP32 summation of K products in any order with truncation errs by at most
// about (K + 1) * 2^-23 times the sum of aligned product magnitudes. Twice
// that, FP32 rounding of the result and bias, and a small absolute term for
// results in FP32's subnormal range bound the error.
[[nodiscard]] bool close(float actual, float expected, double magnitude, std::size_t k) {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return actual == expected;
    }
    const double bound = static_cast<double>(k + 4U) * std::ldexp(magnitude, -22) +
                         std::ldexp(std::fabs(static_cast<double>(expected)), -22) +
                         std::ldexp(1.0, -148);
    return std::isfinite(actual) &&
           std::fabs(static_cast<double>(actual) - static_cast<double>(expected)) <= bound;
}

// The sum of aligned product magnitudes of the row-scaled operands, rescaled
// like a result.
[[nodiscard]] double magnitude_at(const Bf16LinearShape& shape, const ScaledMatrix& a,
                                  const ScaledMatrix& w, std::size_t m, std::size_t n) {
    double sum = 0.0;
    for (std::size_t k = 0; k < shape.k; ++k) {
        sum += alignment_magnitude(a.values[m * shape.k + k]) *
               alignment_magnitude(w.values[n * shape.k + k]);
    }
    if (!std::isfinite(sum)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::ldexp(sum, -(eclair::formats::clamp_bf16_fp16_shift(a.shifts[m]) +
                             eclair::formats::clamp_bf16_fp16_shift(w.shifts[n])));
}

// Row scaling of both BF16 operands on the GPU, then the linear operation,
// compared with the host pipeline: bit for bit for the scaled operands, within
// the summation bound for the output. Rejections must enqueue nothing.
void check_linear(const Bf16LinearShape& shape, const std::vector<std::uint16_t>& a,
                  const std::vector<std::uint16_t>& w, const std::vector<float>& bias,
                  std::size_t offset = 0U) {
    const ScaledMatrix host_a = host_scale(a, shape.m, shape.k);
    const ScaledMatrix host_w = host_scale(w, shape.n, shape.k);
    std::vector<float> reference(shape.m * shape.n);
    const Bf16LinearBuffers host{{host_a.values.data(), host_a.values.size()},
                                 {host_a.shifts.data(), host_a.shifts.size()},
                                 {host_w.values.data(), host_w.values.size()},
                                 {host_w.shifts.data(), host_w.shifts.size()},
                                 {bias.data(), bias.size()},
                                 {reference.data(), reference.size()}};
    ECLAIR_CHECK(eclair::host::bf16_linear_reference(shape, host) == Status::Success);

    Stream stream;
    DeviceScaled device_a = scaled_buffers(shape.m, shape.k, offset, stream.get());
    DeviceScaled device_w = scaled_buffers(shape.n, shape.k, offset, stream.get());
    scale_on_device(a, shape.m, shape.k, offset, stream.get(), device_a);
    scale_on_device(w, shape.n, shape.k, offset, stream.get(), device_w);
    const DeviceBuffer<float> device_bias(bias, stream.get());
    DeviceBuffer<float> result(std::vector<float>(reference.size() + 2U, kFloatSentinel),
                               stream.get());
    const Bf16LinearBuffers device{
        {device_a.value_data(), device_a.count}, {device_a.shift_data(), shape.m},
        {device_w.value_data(), device_w.count}, {device_w.shift_data(), shape.n},
        {device_bias.data(), bias.size()},       {result.data() + 1, reference.size()}};
    checked(eclair::cuda::launch_bf16_linear(shape, device, stream.get()), "BF16 linear");
    const auto output = result.download(stream.get());
    ECLAIR_CHECK(output.front() == kFloatSentinel && output.back() == kFloatSentinel);
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double magnitude = magnitude_at(shape, host_a, host_w, i / shape.n, i % shape.n);
        if (!close(output[i + 1U], reference[i], magnitude, shape.k)) {
            std::fprintf(stderr, "M=%zu N=%zu K=%zu offset=%zu index=%zu actual=%a expected=%a\n",
                         shape.m, shape.n, shape.k, offset, i, static_cast<double>(output[i + 1U]),
                         static_cast<double>(reference[i]));
            ECLAIR_CHECK(false && "GPU BF16 linear differs from the host oracle");
            break;
        }
    }
    auto invalid = device;
    invalid.weight_shifts.size = 0;
    ECLAIR_CHECK(eclair::cuda::launch_bf16_linear(shape, invalid, stream.get()) ==
                 cudaErrorInvalidValue);
    invalid = device;
    invalid.bias = {invalid.output.data, shape.n};
    ECLAIR_CHECK(eclair::cuda::launch_bf16_linear(shape, invalid, stream.get()) ==
                 cudaErrorInvalidValue);
    const auto after_rejection = result.download(stream.get());
    for (std::size_t i = 0; i < output.size(); ++i) {
        ECLAIR_CHECK(std::isnan(output[i]) ? std::isnan(after_rejection[i])
                                           : output[i] == after_rejection[i]);
    }
}

// Random BF16 rows centered between 2^-center_range and 2^center_range, each
// spanning `spread` binades either side. A spread above 16 puts elements
// outside the row's exact window, so the flush rule is exercised too.
[[nodiscard]] std::vector<std::uint16_t> random_rows(std::mt19937& random, std::size_t rows,
                                                     std::size_t columns, int center_range,
                                                     int spread) {
    const auto next = [&](std::uint32_t bound) {
        return static_cast<std::uint32_t>(random() % bound);
    };
    std::vector<std::uint16_t> matrix(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        const int center =
            static_cast<int>(next(static_cast<std::uint32_t>(2 * center_range + 1))) - center_range;
        for (std::size_t column = 0; column < columns; ++column) {
            const int exponent =
                center + static_cast<int>(next(static_cast<std::uint32_t>(2 * spread + 1))) -
                spread;
            matrix[row * columns + column] = static_cast<std::uint16_t>(
                (next(2U) << 15U) | (static_cast<std::uint32_t>(exponent + 127) << 7U) |
                next(128U));
        }
    }
    return matrix;
}

void check_shapes() {
    std::mt19937 random(20261024U);
    struct Case {
        Bf16LinearShape shape;
        std::size_t offset;
        bool with_bias;
    };
    // Skinny M<=4 and tile paths: partial K stages, edge tiles, several
    // stages, vector/scalar loads and misaligned views.
    const Case cases[] = {
        {{1, 1, 1}, 0, false},    {{1, 257, 129}, 0, true},   {{4, 19, 33}, 0, true},
        {{2, 127, 520}, 0, true}, {{4, 73, 512}, 0, false},   {{2, 81, 256}, 1, true},
        {{5, 64, 64}, 0, false},  {{17, 19, 33}, 0, true},    {{32, 48, 64}, 0, true},
        {{64, 64, 8}, 0, false},  {{65, 130, 200}, 0, true},  {{100, 67, 96}, 1, false},
        {{3, 70, 256}, 2, true},  {{257, 300, 520}, 0, true}, {{129, 64, 1024}, 3, true},
    };
    for (const Case& test : cases) {
        const auto& shape = test.shape;
        std::vector<float> bias;
        if (test.with_bias) {
            bias.resize(shape.n);
            for (float& value : bias) {
                value = static_cast<float>(static_cast<int>(random() % 2001U) - 1000) / 8.0F;
            }
        }
        // Exact operands far outside binary16's range, then rows wide enough
        // that small elements round away.
        check_linear(shape, random_rows(random, shape.m, shape.k, 40, 12),
                     random_rows(random, shape.n, shape.k, 40, 12), bias, test.offset);
        check_linear(shape, random_rows(random, shape.m, shape.k, 20, 30),
                     random_rows(random, shape.n, shape.k, 20, 30), bias, test.offset);
    }
}

// Zero shifts also admit resident FP16 operands (the LLM's lm_head case).
// Exercise every binary16 representation on both skinny load paths. Each
// output is one value times 1, so expected values do not depend on a sum order.
void check_skinny_half_patterns() {
    constexpr std::size_t n = 65536U, k = 8U;
    Stream stream;
    for (const std::size_t offset : {0U, 1U}) {
        std::vector<std::uint16_t> a(offset + 4U * k, 0), w(offset + n * k, 0);
        for (std::size_t row = 0; row < 4U; ++row) {
            a[offset + row * k] = 0x3C00U;
        }
        for (std::size_t channel = 0; channel < n; ++channel) {
            w[offset + channel * k] = static_cast<std::uint16_t>(channel);
        }
        const DeviceBuffer<std::uint16_t> da(a, stream.get()), dw(w, stream.get());
        const DeviceBuffer<std::int32_t> as(std::vector<std::int32_t>(4U, 0), stream.get());
        const DeviceBuffer<std::int32_t> ws(std::vector<std::int32_t>(n, 0), stream.get());
        for (std::size_t m = 1U; m <= 4U; ++m) {
            const DeviceBuffer<float> out(std::vector<float>(m * n + 2U, kFloatSentinel),
                                          stream.get());
            const Bf16LinearBuffers b{{da.data() + offset, m * k},
                                      {as.data(), m},
                                      {dw.data() + offset, n * k},
                                      {ws.data(), n},
                                      {},
                                      {out.data() + 1U, m * n}};
            checked(eclair::cuda::launch_bf16_linear({m, n, k}, b, stream.get()), "FP16 skinny");
            const auto actual = out.download(stream.get());
            ECLAIR_CHECK(actual.front() == kFloatSentinel && actual.back() == kFloatSentinel);
            for (std::size_t i = 0; i < m * n; ++i) {
                const float expected =
                    eclair::formats::decode_fp16(static_cast<std::uint16_t>(i % n));
                ECLAIR_CHECK(std::isnan(expected) ? std::isnan(actual[i + 1U])
                                                  : actual[i + 1U] == expected);
            }
        }
    }
}

// Nonfinite inputs, results beyond FP32's range or in its subnormal range,
// and the flush below a row's exact window, through both kernels.
void check_exceptional() {
    const auto bf16 = [](float value) { return eclair::formats::encode_bf16(value); };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::uint16_t inf = 0x7F80U;
    const std::uint16_t neg_inf = 0xFF80U;
    // A 5x3 by 4x3 product. The rows of A test overflow against the large W
    // row, FP32 subnormal results, infinity times finite and zero, NaN, and a
    // 1 that rounds away beside 2^40. W adds an infinity of its own.
    const auto a = flatten(std::array<std::array<std::uint16_t, 3>, 5>{{
        {bf16(0x1p100F), bf16(1.0F), bf16(0.0F)},
        {bf16(0x1p-70F), bf16(0.0F), bf16(0.0F)},
        {inf, bf16(1.0F), bf16(2.0F)},
        {0x7FC0U, bf16(1.0F), bf16(1.0F)},
        {bf16(0x1p40F), bf16(-0x1p40F), bf16(1.0F)},
    }});
    const auto w = flatten(std::array<std::array<std::uint16_t, 3>, 4>{{
        {bf16(0x1p100F), bf16(0.0F), bf16(0.0F)},
        {bf16(0x1p-70F), bf16(0.0F), bf16(0.0F)},
        {bf16(1.0F), bf16(1.0F), bf16(1.0F)},
        {neg_inf, bf16(0.0F), bf16(0.0F)},
    }});
    check_linear({5, 4, 3}, a, w, {});
    check_linear({5, 4, 3}, a, w, {1.0F, -2.0F, nan, 0.5F});
    // The same products on the tile path with M past one tile row.
    std::vector<std::uint16_t> tall;
    for (int copy = 0; copy < 20; ++copy) {
        tall.insert(tall.end(), a.begin(), a.end());
    }
    check_linear({100, 4, 3}, tall, w, {});
}

// Runs one dot product of BF16 values through both kernels.
[[nodiscard]] float device_dot(const std::vector<float>& a_values,
                               const std::vector<float>& w_values) {
    std::vector<std::uint16_t> a;
    std::vector<std::uint16_t> w;
    for (const float value : a_values) {
        a.push_back(eclair::formats::encode_bf16(value));
    }
    for (const float value : w_values) {
        w.push_back(eclair::formats::encode_bf16(value));
    }
    const std::size_t k = a.size();
    Stream stream;
    // Five identical rows force the tensor-core path for this alignment probe.
    std::vector<std::uint16_t> repeated;
    for (int row = 0; row < 5; ++row) {
        repeated.insert(repeated.end(), a.begin(), a.end());
    }
    DeviceScaled device_a = scaled_buffers(5, k, 0, stream.get());
    DeviceScaled device_w = scaled_buffers(1, k, 0, stream.get());
    scale_on_device(repeated, 5, k, 0, stream.get(), device_a);
    scale_on_device(w, 1, k, 0, stream.get(), device_w);
    DeviceBuffer<float> result(5);
    const Bf16LinearBuffers buffers{{device_a.value_data(), 5U * k},
                                    {device_a.shift_data(), 5},
                                    {device_w.value_data(), k},
                                    {device_w.shift_data(), 1},
                                    {},
                                    {result.data(), 5}};
    checked(eclair::cuda::launch_bf16_linear({5, 1, k}, buffers, stream.get()), "dot product");
    return result.download(stream.get())[0];
}

// Measures how the device's tensor cores add products, the property behind
// the error bound. With normal operands, 1 + 1.5 * 2^-j stays exact while the
// window holds bit j + 1. With a subnormal operand in the largest product,
// the case below truncates its smaller product to a coarser grid. Both must
// stay within the bound, and both are printed as evidence.
void check_alignment() {
    int window = 0;
    for (int j = 8; j <= 30; ++j) {
        const float x = std::ldexp(1.5F, -j);
        if (device_dot({1.0F, x}, {1.0F, 1.0F}) != 1.0F + x) {
            break;
        }
        window = j + 1;
    }
    // Row-scaled, these are the products 2^-23 * 2^15 = 2^-8 (a subnormal
    // times a normal) and 0x1.dp-6 * -0x1.c8p-15 = -0x1.9d4p-20 (a normal
    // times a subnormal), as in a case found by the random tests.
    const std::vector<float> a{1.0F, 0x1p-38F, 0x1.dp-21F};
    const std::vector<float> w{0.0F, 1.0F, -0x1.c8p-30F};
    const double exact = 0x1p-38 + 0x1.dp-21 * -0x1.c8p-30;
    const double actual = static_cast<double>(device_dot(a, w));
    std::printf("ALIGNMENT normal_operands_exact_to_bit=%d subnormal_case_relative_error=%.3g\n",
                window, (actual - exact) / exact);
    ECLAIR_CHECK(window >= 22);
    // The bound weighs the subnormal 2^-23 as 2^-14: (3 + 4) * 2^-22 * 2^-14 * 2^15,
    // rescaled by 2^-30.
    ECLAIR_CHECK(std::fabs(actual - exact) <= 7.0 * 0x1p-22 * 0x1p1 * 0x1p-30);
}

void benchmark() {
    Blas blas;
    int version = 0;
    checked_blas(cublasGetVersion(blas.get(), &version), "cuBLAS version");
    std::printf(
        "BENCH cuBLAS=%d warmup=100ms-per-path samples=9 stream=nonblocking graphs=off "
        "timing=CUDA-events synchronization=event output=FP32\n",
        version);
    std::puts(
        "BENCH eclair=row scaling of A + linear, W row-scaled once beforehand (weight_prep "
        "timed alone); fp16=cuBLAS GemmEx on resident FP16 casts, no range protection; "
        "bf16=cuBLAS GemmEx CUDA_R_16BF; sgemm=cuBLAS SGEMM on FP32 copies; inputs are BF16 "
        "values within FP16's range so every baseline is finite; utilization not measured");
    struct Size {
        std::size_t m;
        std::size_t n;
        std::size_t k;
    };
    const Size sizes[] = {{1, 4096, 4096},
                          {16, 4096, 4096},
                          {128, 4096, 4096},
                          {1024, 4096, 4096},
                          {4096, 4096, 4096}};
    std::mt19937 random(4096U);
    for (const Size& size : sizes) {
        const Bf16LinearShape shape{size.m, size.n, size.k};
        Stream stream;
        checked_blas(cublasSetStream(blas.get(), stream.get()), "benchmark stream");
        const auto a = random_rows(random, shape.m, shape.k, 0, 3);
        const auto w = random_rows(random, shape.n, shape.k, 0, 3);
        std::vector<std::uint16_t> a16(a.size());
        std::vector<std::uint16_t> w16(w.size());
        std::vector<float> a32(a.size());
        std::vector<float> w32(w.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            a32[i] = eclair::formats::decode_bf16(a[i]);
            a16[i] = eclair::formats::encode_fp16(a32[i]);
        }
        for (std::size_t i = 0; i < w.size(); ++i) {
            w32[i] = eclair::formats::decode_bf16(w[i]);
            w16[i] = eclair::formats::encode_fp16(w32[i]);
        }
        const DeviceBuffer<std::uint16_t> da(a, stream.get());
        const DeviceBuffer<std::uint16_t> dw(w, stream.get());
        const DeviceBuffer<std::uint16_t> da16(a16, stream.get());
        const DeviceBuffer<std::uint16_t> dw16(w16, stream.get());
        const DeviceBuffer<float> da32(a32, stream.get());
        const DeviceBuffer<float> dw32(w32, stream.get());
        const DeviceBuffer<std::uint16_t> scaled_a(a.size());
        const DeviceBuffer<std::int32_t> a_shifts(shape.m);
        const DeviceBuffer<std::uint16_t> scaled_w(w.size());
        const DeviceBuffer<std::int32_t> w_shifts(shape.n);
        const DeviceBuffer<float> eclair_output(shape.m * shape.n);
        const DeviceBuffer<float> blas_output(shape.m * shape.n);
        const DeviceBuffer<float> sgemm_output(shape.m * shape.n);

        const auto prepare_weights = [&] {
            checked(eclair::cuda::launch_bf16_row_scaling({shape.n, shape.k},
                                                          {{dw.data(), w.size()},
                                                           {scaled_w.data(), w.size()},
                                                           {w_shifts.data(), shape.n}},
                                                          stream.get()),
                    "weight row scaling");
        };
        const auto scale_activations = [&] {
            checked(eclair::cuda::launch_bf16_row_scaling({shape.m, shape.k},
                                                          {{da.data(), a.size()},
                                                           {scaled_a.data(), a.size()},
                                                           {a_shifts.data(), shape.m}},
                                                          stream.get()),
                    "activation row scaling");
        };
        const Bf16LinearBuffers buffers{{scaled_a.data(), a.size()},
                                        {a_shifts.data(), shape.m},
                                        {scaled_w.data(), w.size()},
                                        {w_shifts.data(), shape.n},
                                        {},
                                        {eclair_output.data(), shape.m * shape.n}};
        const auto linear = [&] {
            checked(eclair::cuda::launch_bf16_linear(shape, buffers, stream.get()), "BF16 linear");
        };
        const auto eclair = [&] {
            scale_activations();
            linear();
        };
        const float alpha = 1.0F;
        const float beta = 0.0F;
        const int m = static_cast<int>(shape.m);
        const int n = static_cast<int>(shape.n);
        const int k = static_cast<int>(shape.k);
        const auto gemm_ex = [&](const void* weights, const void* activations, cudaDataType type) {
            return cublasGemmEx(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weights,
                                type, k, activations, type, k, &beta, blas_output.data(),
                                CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        };
        const auto fp16 = [&] {
            checked_blas(gemm_ex(dw16.data(), da16.data(), CUDA_R_16F), "cuBLAS FP16");
        };
        const auto bf16 = [&] {
            checked_blas(gemm_ex(dw.data(), da.data(), CUDA_R_16BF), "cuBLAS BF16");
        };
        const auto sgemm = [&] {
            checked_blas(cublasSgemm(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                                     dw32.data(), k, da32.data(), k, &beta, sgemm_output.data(), n),
                         "cuBLAS SGEMM");
        };

        prepare_weights();
        eclair();
        sgemm();
        const bool bf16_supported =
            gemm_ex(dw.data(), da.data(), CUDA_R_16BF) == CUBLAS_STATUS_SUCCESS;
        const auto actual = eclair_output.download(stream.get());
        const auto accurate = sgemm_output.download(stream.get());
        // 256 sampled outputs against the FP64 oracle within the summation
        // bound, and the largest difference from SGEMM over all outputs.
        const ScaledMatrix host_a = host_scale(a, shape.m, shape.k);
        const ScaledMatrix host_w = host_scale(w, shape.n, shape.k);
        for (std::size_t sample = 0; sample < 256U; ++sample) {
            const std::size_t mm = (sample * 2654435761U) % shape.m;
            const std::size_t nn = (sample * 40503U + 7U) % shape.n;
            float expected = 0;
            const Bf16LinearBuffers one{{host_a.values.data() + mm * shape.k, shape.k},
                                        {host_a.shifts.data() + mm, 1},
                                        {host_w.values.data() + nn * shape.k, shape.k},
                                        {host_w.shifts.data() + nn, 1},
                                        {},
                                        {&expected, 1}};
            ECLAIR_CHECK(eclair::host::bf16_linear_reference({1, 1, shape.k}, one) ==
                         Status::Success);
            ECLAIR_CHECK(close(actual[mm * shape.n + nn], expected,
                               magnitude_at(shape, host_a, host_w, mm, nn), shape.k));
        }
        double max_difference = 0;
        double max_accurate = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            max_difference =
                std::max(max_difference, std::fabs(static_cast<double>(actual[i]) - accurate[i]));
            max_accurate = std::max(max_accurate, std::fabs(static_cast<double>(accurate[i])));
        }
        if (eclair::test::failure_count() != 0) {
            throw std::runtime_error("benchmark correctness failed");
        }
        const auto prep_times = timings(stream.get(), prepare_weights);
        const auto scale_times = timings(stream.get(), scale_activations);
        const auto linear_times = timings(stream.get(), linear);
        const auto eclair_times = timings(stream.get(), eclair);
        const auto fp16_times = timings(stream.get(), fp16);
        const auto sgemm_times = timings(stream.get(), sgemm);
        const double operations = 2.0 * static_cast<double>(shape.m) *
                                  static_cast<double>(shape.n) * static_cast<double>(shape.k);
        const auto tflops = [&](float milliseconds) {
            return operations / (static_cast<double>(milliseconds) * 1.0e9);
        };
        std::printf(
            "BENCH M=%zu N=%zu K=%zu weight_prep_us=%.3f a_scaling_us=%.3f linear_us=%.3f "
            "eclair_median_us=%.3f eclair_p95_us=%.3f fp16_median_us=%.3f sgemm_median_us=%.3f "
            "eclair_TFLOPS=%.3f fp16_TFLOPS=%.3f sgemm_TFLOPS=%.3f max_abs_diff_vs_sgemm=%.3g "
            "max_abs_sgemm=%.3g\n",
            shape.m, shape.n, shape.k, prep_times[4] * 1000.0F, scale_times[4] * 1000.0F,
            linear_times[4] * 1000.0F, eclair_times[4] * 1000.0F, eclair_times.back() * 1000.0F,
            fp16_times[4] * 1000.0F, sgemm_times[4] * 1000.0F, tflops(eclair_times[4]),
            tflops(fp16_times[4]), tflops(sgemm_times[4]), max_difference, max_accurate);
        if (bf16_supported) {
            const auto bf16_times = timings(stream.get(), bf16);
            std::printf(
                "BENCH M=%zu N=%zu K=%zu bf16_median_us=%.3f bf16_p95_us=%.3f "
                "bf16_TFLOPS=%.3f\n",
                shape.m, shape.n, shape.k, bf16_times[4] * 1000.0F, bf16_times.back() * 1000.0F,
                tflops(bf16_times[4]));
        } else {
            std::printf("BENCH M=%zu N=%zu K=%zu bf16=unsupported\n", shape.m, shape.n, shape.k);
        }
    }
}

void run() {
    eclair::test::print_device_header();
    std::puts(
        "BF16 rows scaled onto binary16 by powers of two (bit-exact with the host); CUDA/WMMA "
        "FP32 accumulation; nonblocking stream; stream synchronize; tolerance "
        "(K+4)*2^-22*sum|a*w| + 2^-22*|ref| + 2^-148 against the FP64 oracle, with binary16 "
        "subnormal operands weighed as 2^-14");
    checked(eclair::cuda::check_bf16_linear_image(0), "BF16 linear kernel image");
    ECLAIR_CHECK(eclair::cuda::check_bf16_linear_image(-1) == cudaErrorInvalidValue);
    ECLAIR_CHECK(eclair::cuda::launch_bf16_linear({}, {}, nullptr) == cudaSuccess);
    ECLAIR_CHECK(eclair::cuda::launch_bf16_row_scaling({}, {}, nullptr) == cudaSuccess);
    ECLAIR_CHECK(eclair::cuda::launch_bf16_row_scaling({2, 0}, {}, nullptr) ==
                 cudaErrorInvalidValue);
    check_row_scaling_patterns();
    check_alignment();
    check_shapes();
    check_skinny_half_patterns();
    check_exceptional();
    std::puts(
        "checked: every BF16 pattern through row scaling on vector and scalar paths, grid-stride "
        "rows, irregular and multi-stage tile shapes, misaligned views, wide exponent spreads "
        "with flushed elements, overflow, FP32 subnormal results, infinity, NaN, bias, guards "
        "and rejection");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--benchmark")) {
        std::fprintf(stderr, "Usage: eclair_cuda_bf16_linear_tests [--benchmark]\n");
        return 2;
    }
    return eclair::test::contain([argc] {
        run();
        if (argc == 2 && eclair::test::failure_count() == 0) {
            benchmark();
        }
    });
}
