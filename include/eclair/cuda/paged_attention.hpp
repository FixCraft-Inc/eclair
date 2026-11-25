/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/paged_attention.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

// The paged int8 prefill attention of eclair/ops/paged_attention.hpp on
// NVIDIA tensor cores (WMMA, binary16 inputs, FP32 accumulation). Device
// pointers only. The caller owns buffers and the stream until the work
// completes. Nothing is allocated, copied or synchronized, and the current
// device does not change. Invalid or unsupported arguments enqueue no work and
// return cudaErrorInvalidValue. Success means enqueued.
namespace eclair::cuda {

// Query tokens per thread block and the smallest chunk the kernel accepts.
// A chunk's last row tile is shifted back to end at its last row, so every
// tile reads 16 real query rows. Shorter chunks belong to a decode kernel.
inline constexpr std::size_t kPagedPrefillMinRows = 16U;
inline constexpr std::size_t kPagedPrefillMaxGroup = 8U;

// Beyond the contract's validation the kernel needs: head size 128 or 256,
// at most kPagedPrefillMaxGroup query heads per KV head and no more than fit
// Volta's 96 KiB of shared memory with their query rows (six at head size
// 256, eight at 128), at least
// kPagedPrefillMinRows rows, query rows and their head vectors 32-byte
// aligned (WMMA loads), key and value bases and byte strides 4-byte aligned,
// and every count and stride within int range.
// The head configurations the kernel implements, the part of
// paged_prefill_supported that does not depend on a call's buffers.
[[nodiscard]] bool paged_prefill_heads_supported(std::size_t heads, std::size_t kv_heads,
                                                 std::size_t head_size) noexcept;

[[nodiscard]] bool paged_prefill_supported(const ops::PagedPrefillShape& shape,
                                           const ops::PagedInt8CacheLayout& layout,
                                           const ops::PagedPrefillBuffers& buffers) noexcept;

[[nodiscard]] cudaError_t launch_paged_prefill(const ops::PagedPrefillShape& shape,
                                               const ops::PagedInt8CacheLayout& layout,
                                               const ops::PagedPrefillBuffers& buffers,
                                               cudaStream_t stream) noexcept;

// Kernel-image probe for every instantiation, not a numerical or performance
// test. Restores the previous current device and clears an expected error.
[[nodiscard]] cudaError_t check_paged_prefill_image(int device_index) noexcept;

// The batched paged decode of eclair/ops/paged_attention.hpp: one thread block
// per context split, KV head and sequence holds all of the sequence's query
// rows times the KV head's query heads, then a second kernel merges the
// splits. Both run on `stream` and read the metadata on the device, so CUDA
// graph capture records them without host synchronization. Keys enter the
// tensor cores as exact integers and their scales multiply the FP32 scores.
// Values are binary16(int8 * binary16(scale)), as the contract permits.
//
// Beyond the contract's validation the kernels need: head size 128 or 256,
// max_rows_per_sequence times the query heads per KV head at most
// kPagedDecodeMaxGroupRows, at most kPagedDecodeMaxSplits splits and 65535
// sequences, heads and KV heads, key and value bases and byte strides 4-byte
// aligned, and rows, blocks and block_size within int range.
inline constexpr std::size_t kPagedDecodeMaxGroupRows = 32U;
inline constexpr std::size_t kPagedDecodeMaxSplits = 1024U;

[[nodiscard]] bool paged_decode_heads_supported(std::size_t heads, std::size_t kv_heads,
                                                std::size_t head_size,
                                                std::size_t max_rows_per_sequence) noexcept;

[[nodiscard]] bool paged_decode_supported(const ops::PagedDecodeShape& shape,
                                          const ops::PagedInt8CacheLayout& layout,
                                          const ops::PagedDecodeBuffers& buffers) noexcept;

[[nodiscard]] cudaError_t launch_paged_decode(const ops::PagedDecodeShape& shape,
                                              const ops::PagedInt8CacheLayout& layout,
                                              const ops::PagedDecodeBuffers& buffers,
                                              cudaStream_t stream) noexcept;

[[nodiscard]] cudaError_t check_paged_decode_image(int device_index) noexcept;

}  // namespace eclair::cuda
