/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/formats/fp8.hpp"
#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>

namespace eclair::ops {

// Experimental, contiguous row-major linear operation: A[M,K] * W[N,K]^T
// plus optional FP32 bias[N], with FP16 A, FP8 W and FP32 output[M,N].
// Weight scales multiply decoded W before nearest-even conversion to FP16.
// Scales[ceil(N/block_n),ceil(K/block_k)] support tensor, channel and block
// scaling. K must be positive for nonempty output. No input is modified.
struct Fp8LinearShape {
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    std::size_t block_n = 1;
    std::size_t block_k = 128;
    formats::Fp8Format format = formats::Fp8Format::E4M3FN;
};

// All pointers belong to the execution backend's memory domain. Device
// buffers must be accessible on the current device. Extents are caller claims;
// validation cannot establish the actual allocation size from a raw pointer.
struct Fp8LinearBuffers {
    BufferView<const std::uint16_t> activations;
    BufferView<const std::uint8_t> weights;
    BufferView<const float> scales;
    BufferView<const float> bias;  // size == 0 means absent.
    BufferView<float> output;
};

struct Fp8LinearSizes {
    std::size_t activations = 0;
    std::size_t weights = 0;
    std::size_t scales = 0;
    std::size_t output = 0;
};

// Validate format, dimensions and byte-count arithmetic without allocating.
// A zero M or N is a no-op, still requiring a valid format and scale layout.
[[nodiscard]] Status fp8_linear_sizes(const Fp8LinearShape& shape, Fp8LinearSizes& sizes) noexcept;

// Additionally reject missing, short, misaligned or wrapping buffers and any
// overlap between output and used input ranges. Does not read tensor values.
[[nodiscard]] Status validate_fp8_linear(const Fp8LinearShape& shape,
                                         const Fp8LinearBuffers& buffers) noexcept;

}  // namespace eclair::ops
