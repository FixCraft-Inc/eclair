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

// Experimental symmetric group-128 W4A16: A[M,K], packed W[K/8,N],
// binary16 scales[K/128,N], binary16 output[M,N]. Each word's low-to-high
// nibbles name logical K offsets {0,2,4,6,1,3,5,7}. This is an explicit
// storage contract, including the unsigned nibble's zero point of 8.
//
// w[k,n] = binary16((nibble(k,n) - 8) * scale[k/128,n])
// out[m,n] = binary16(fp32(sum_k A[m,k] * w[k,n]))
//
// Products are exact in FP32. Accumulation is FP32 in an unspecified order
// and, as on tensor cores, may truncate terms about 23 bits below the largest
// one; a binary16 subnormal operand may align as if its exponent were -14
// (docs/W4A16_LINEAR.md). NaNs, infinities and gradual binary16 underflow
// follow IEEE arithmetic. Zero signs are not promised. There is no bias or
// implicit activation reorder.
struct W4A16LinearShape {
    std::size_t m = 0;
    std::size_t n = 0;
    std::size_t k = 0;
    std::size_t splits = 1;  // FP32 partial sums; execution tuning, not a weight format.
};

struct W4A16LinearBuffers {
    BufferView<const std::uint16_t> activations;
    BufferView<const std::uint32_t> weights;
    BufferView<const std::uint16_t> scales;
    BufferView<std::uint16_t> output;
    BufferView<float> workspace;  // [splits,M,N], caller-owned, never aliases other buffers.
};

// No value reads. Empty M/N is a no-op. Otherwise K must be a positive
// multiple of 128, splits positive, extents nonwrapping and writes disjoint
// from all other buffers. Inputs may alias. The FP64 host oracle validates
// scratch too but does not need to read or write it.
[[nodiscard]] Status validate_w4a16_linear(const W4A16LinearShape& shape,
                                           const W4A16LinearBuffers& buffers) noexcept;

// The contract's binary16 weights w[k,n], written row-major into a
// caller-owned [K,N] array. Large-M implementations reconstruct once and
// then multiply. Signed zeros follow IEEE multiplication here.
struct W4A16DequantizeBuffers {
    BufferView<const std::uint32_t> weights;
    BufferView<const std::uint16_t> scales;
    BufferView<std::uint16_t> dequantized;
};

// No value reads. Empty N is a no-op. Otherwise K must be a positive
// multiple of 128, extents nonwrapping and the output disjoint from the inputs.
[[nodiscard]] Status validate_w4a16_dequantize(std::size_t n, std::size_t k,
                                               const W4A16DequantizeBuffers& buffers) noexcept;

// Large-M implementation of the same numerical contract: reconstruct every
// weight into `dequantized` [K,N], then one GEMM. `workspace` is that GEMM's
// byte scratch and may be empty. shape.splits must be 1: the GEMM chooses
// its own split, still reducing in FP32.
struct W4A16GemmBuffers {
    BufferView<const std::uint16_t> activations;
    BufferView<const std::uint32_t> weights;
    BufferView<const std::uint16_t> scales;
    BufferView<std::uint16_t> dequantized;
    BufferView<std::uint16_t> output;
    BufferView<std::byte> workspace;
};

// No value reads. Empty M/N is a no-op. Otherwise K must be a positive
// multiple of 128, extents nonwrapping, and the three written buffers
// disjoint from each other and from the inputs. Inputs may alias.
[[nodiscard]] Status validate_w4a16_gemm(const W4A16LinearShape& shape,
                                         const W4A16GemmBuffers& buffers) noexcept;

}  // namespace eclair::ops
