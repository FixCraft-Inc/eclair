/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>

// Experimental BF16 linear operation for tensor cores that take only binary16.
// Row scaling moves each row of a BF16 matrix onto binary16 by a power of two
// and records that power. The linear operation multiplies two row-scaled
// matrices with FP32 accumulation and undoes both rows' powers. Weights are
// row-scaled once, activations on every call. See docs/BF16_LINEAR.md.
namespace eclair::ops {

// For each row r of a contiguous row-major BF16 matrix:
//
//   shifts[r]   = formats::bf16_fp16_shift(largest finite magnitude in row r)
//   output[r,c] = formats::bf16_to_fp16_scaled_bits(input[r,c], shifts[r])
//
// Elements whose exponent is at most 32 below the row's largest stay exact.
// Smaller ones round to binary16 subnormals or zero. Infinity and NaN do not
// affect the shift and stay infinite or NaN.
struct Bf16RowScalingShape {
    std::size_t rows = 0;
    std::size_t columns = 0;
};

struct Bf16RowScalingBuffers {
    BufferView<const std::uint16_t> input;  // BF16 bits [rows, columns].
    BufferView<std::uint16_t> output;       // binary16 bits [rows, columns].
    BufferView<std::int32_t> shifts;        // [rows].
};

// Rejects missing, short, misaligned, wrapping or overlapping buffers and size
// overflow. Zero rows is a no-op that needs no buffers. Otherwise columns must
// be positive. Reads no values.
[[nodiscard]] Status validate_bf16_row_scaling(const Bf16RowScalingShape& shape,
                                               const Bf16RowScalingBuffers& buffers) noexcept;

// Linear operation on row-scaled A[M,K] and W[N,K], both row-major:
//
//   output[m,n] = fp32(sum_k A[m,k] * W[n,k] * 2^-(a_shift[m] + w_shift[n]))
//                 + bias[n]
//
// A and W hold binary16 bits and the shifts pass through
// formats::clamp_bf16_fp16_shift. Each product of two binary16 values is
// exact in FP32. The GPU accumulates in FP32 in an unspecified order, then
// applies the power of two, which is exact unless the result leaves FP32's
// normal range. The host oracle sums in FP64 and rounds once. The optional
// bias is added in FP32. NaN and infinity propagate as in IEEE arithmetic,
// and zero signs are not promised.
struct Bf16LinearShape {
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
};

// Inputs may alias each other. Output may not overlap any used input range.
struct Bf16LinearBuffers {
    BufferView<const std::uint16_t> activations;       // Row-scaled binary16 [M,K].
    BufferView<const std::int32_t> activation_shifts;  // [M].
    BufferView<const std::uint16_t> weights;           // Row-scaled binary16 [N,K].
    BufferView<const std::int32_t> weight_shifts;      // [N].
    BufferView<const float> bias;                      // [N], or size 0 when absent.
    BufferView<float> output;                          // [M,N].
};

// Element counts. The shift arrays hold M and N elements.
struct Bf16LinearSizes {
    std::size_t activations = 0;
    std::size_t weights = 0;
    std::size_t output = 0;
};

// Validates dimensions and byte-count arithmetic without allocating. A zero M
// or N is a no-op. Nonempty output needs K > 0.
[[nodiscard]] Status bf16_linear_sizes(const Bf16LinearShape& shape,
                                       Bf16LinearSizes& sizes) noexcept;

// Additionally rejects missing, short, misaligned or wrapping buffers and any
// overlap between output and used input ranges. Reads no values.
[[nodiscard]] Status validate_bf16_linear(const Bf16LinearShape& shape,
                                          const Bf16LinearBuffers& buffers) noexcept;

}  // namespace eclair::ops
