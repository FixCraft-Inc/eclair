/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/paged_attention.hpp"

#include <cstddef>

namespace eclair::host {

// The reference keeps one FP64 accumulator per head element on the stack.
inline constexpr std::size_t kPagedPrefillReferenceMaxHeadSize = 1024U;

// Host pointers only, and the caller owns all storage. No allocation.
// Validates, then reads block ids and skips none: an id outside [0, blocks)
// returns InvalidArgument before any output is written. Dequantizes through
// formats::encode_fp16 as the contract states, then computes scores, the
// softmax and the weighted sum in FP64 and rounds each output once to
// binary16. Head sizes above kPagedPrefillReferenceMaxHeadSize are refused.
[[nodiscard]] ops::Status paged_prefill_reference(const ops::PagedPrefillShape& shape,
                                                  const ops::PagedInt8CacheLayout& layout,
                                                  const ops::PagedPrefillBuffers& buffers) noexcept;

// Validates, then reads and checks every sequence's metadata and block ids
// before writing any output: metadata outside the contract's rules or an id
// outside [0, blocks) returns InvalidArgument. Each chunk then goes through
// paged_prefill_reference. Rows no sequence owns and the workspace stay
// untouched, so the workspace view may be empty.
[[nodiscard]] ops::Status paged_decode_reference(const ops::PagedDecodeShape& shape,
                                                 const ops::PagedInt8CacheLayout& layout,
                                                 const ops::PagedDecodeBuffers& buffers) noexcept;

}  // namespace eclair::host
