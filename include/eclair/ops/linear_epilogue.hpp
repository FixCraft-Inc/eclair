/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>

namespace eclair::ops {

// Experimental binary16 output stage for a linear operation whose FP32
// accumulators already exist:
//
//   output[m,n] = binary16(fp32(fp32(accumulators[m,n] * scale) + bias[n]))
//
// The product and the sum each round to nearest-even FP32 and are never fused
// into one multiply-add. The result then rounds to nearest-even binary16 with
// gradual underflow and overflow to infinity. Without bias no addition occurs,
// so a negative zero product stays negative. This is exactly the separate FP32
// multiply, FP32 add and FP16 conversion it replaces.
struct LinearEpilogueShape {
    std::size_t m = 0;
    std::size_t n = 0;
};

// Same memory-domain and ownership rules as BufferView. Inputs may
// alias each other, but output may not overlap any used input range.
struct LinearEpilogueBuffers {
    BufferView<const float> accumulators;  // [M,N], row-major.
    BufferView<const float> scale;         // One element.
    BufferView<const float> bias;          // [N], or size 0 when absent.
    BufferView<std::uint16_t> output;      // binary16 bits [M,N], row-major.
};

// Rejects short, missing, misaligned, wrapping or overlapping buffers and size
// overflow. Zero M or N is a no-op that needs no buffers. Reads no values.
[[nodiscard]] Status validate_linear_epilogue(const LinearEpilogueShape& shape,
                                              const LinearEpilogueBuffers& buffers) noexcept;

}  // namespace eclair::ops
