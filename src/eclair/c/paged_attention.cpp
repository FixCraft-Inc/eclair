/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/c/paged_attention.h"

#include "eclair/cuda/paged_attention.hpp"
#include "eclair/ops/paged_attention.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace {

[[nodiscard]] bool fits(std::uint64_t value) noexcept {
    return value <= std::numeric_limits<std::size_t>::max();
}

[[nodiscard]] std::size_t size(std::uint64_t value) noexcept {
    return static_cast<std::size_t>(value);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int eclair_paged_prefill_int8(
    const EclairPagedPrefillArgs* args, void* stream) {
    if (args == nullptr || args->abi != ECLAIR_PAGED_PREFILL_ABI) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const EclairPagedPrefillArgs& a = *args;
    for (const std::uint64_t value : {a.query_elements,
                                      a.query_row_stride,
                                      a.output_elements,
                                      a.output_row_stride,
                                      a.cache_bytes,
                                      a.scale_elements,
                                      a.block_table_entries,
                                      a.blocks,
                                      a.block_size,
                                      a.kv_heads,
                                      a.head_size,
                                      a.block_stride,
                                      a.slot_stride,
                                      a.head_stride,
                                      a.scale_block_stride,
                                      a.scale_slot_stride,
                                      a.scale_head_stride,
                                      a.heads,
                                      a.rows,
                                      a.context}) {
        if (!fits(value)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    const eclair::ops::PagedPrefillShape shape{size(a.heads),
                                               size(a.rows),
                                               size(a.context),
                                               size(a.query_row_stride),
                                               size(a.output_row_stride),
                                               a.scale};
    const eclair::ops::PagedInt8CacheLayout layout{
        size(a.blocks),           size(a.block_size),         size(a.kv_heads),
        size(a.head_size),        size(a.block_stride),       size(a.slot_stride),
        size(a.head_stride),      size(a.scale_block_stride), size(a.scale_slot_stride),
        size(a.scale_head_stride)};
    const eclair::ops::PagedPrefillBuffers buffers{
        {static_cast<const std::uint16_t*>(a.query), size(a.query_elements)},
        {static_cast<std::uint16_t*>(a.output), size(a.output_elements)},
        {static_cast<const std::int8_t*>(a.keys), size(a.cache_bytes)},
        {static_cast<const std::int8_t*>(a.values), size(a.cache_bytes)},
        {a.key_scales, size(a.scale_elements)},
        {a.value_scales, size(a.scale_elements)},
        {a.block_table, size(a.block_table_entries)}};
    return static_cast<int>(eclair::cuda::launch_paged_prefill(shape, layout, buffers,
                                                               static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_paged_prefill_image(int device_index) {
    return static_cast<int>(eclair::cuda::check_paged_prefill_image(device_index));
}

extern "C" __attribute__((visibility("default"))) int eclair_paged_prefill_heads_supported(
    std::uint64_t heads, std::uint64_t kv_heads, std::uint64_t head_size) {
    return fits(heads) && fits(kv_heads) && fits(head_size) &&
                   eclair::cuda::paged_prefill_heads_supported(size(heads), size(kv_heads),
                                                               size(head_size))
               ? 1
               : 0;
}

extern "C" __attribute__((visibility("default"))) int eclair_paged_decode_int8(
    const EclairPagedDecodeArgs* args, void* stream) {
    if (args == nullptr || args->abi != ECLAIR_PAGED_DECODE_ABI) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const EclairPagedDecodeArgs& a = *args;
    for (const std::uint64_t value : {a.query_elements,
                                      a.query_row_stride,
                                      a.output_elements,
                                      a.output_row_stride,
                                      a.cache_bytes,
                                      a.scale_elements,
                                      a.block_table_entries,
                                      a.block_table_stride,
                                      a.max_blocks_per_sequence,
                                      a.workspace_elements,
                                      a.blocks,
                                      a.block_size,
                                      a.kv_heads,
                                      a.head_size,
                                      a.block_stride,
                                      a.slot_stride,
                                      a.head_stride,
                                      a.scale_block_stride,
                                      a.scale_slot_stride,
                                      a.scale_head_stride,
                                      a.heads,
                                      a.sequences,
                                      a.rows,
                                      a.max_rows_per_sequence,
                                      a.splits}) {
        if (!fits(value)) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }
    if (a.sequences >= std::numeric_limits<std::size_t>::max()) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const eclair::ops::PagedDecodeShape shape{size(a.heads),
                                              size(a.sequences),
                                              size(a.rows),
                                              size(a.max_rows_per_sequence),
                                              size(a.max_blocks_per_sequence),
                                              size(a.block_table_stride),
                                              size(a.splits),
                                              size(a.query_row_stride),
                                              size(a.output_row_stride),
                                              a.scale};
    const eclair::ops::PagedInt8CacheLayout layout{
        size(a.blocks),           size(a.block_size),         size(a.kv_heads),
        size(a.head_size),        size(a.block_stride),       size(a.slot_stride),
        size(a.head_stride),      size(a.scale_block_stride), size(a.scale_slot_stride),
        size(a.scale_head_stride)};
    const eclair::ops::PagedDecodeBuffers buffers{
        {static_cast<const std::uint16_t*>(a.query), size(a.query_elements)},
        {static_cast<std::uint16_t*>(a.output), size(a.output_elements)},
        {static_cast<const std::int8_t*>(a.keys), size(a.cache_bytes)},
        {static_cast<const std::int8_t*>(a.values), size(a.cache_bytes)},
        {a.key_scales, size(a.scale_elements)},
        {a.value_scales, size(a.scale_elements)},
        {a.block_tables, size(a.block_table_entries)},
        {a.query_starts, size(a.sequences) + 1U},
        {a.seq_lens, size(a.sequences)},
        {a.workspace, size(a.workspace_elements)}};
    return static_cast<int>(eclair::cuda::launch_paged_decode(shape, layout, buffers,
                                                              static_cast<cudaStream_t>(stream)));
}

extern "C" __attribute__((visibility("default"))) int eclair_paged_decode_image(int device_index) {
    return static_cast<int>(eclair::cuda::check_paged_decode_image(device_index));
}

extern "C" __attribute__((visibility("default"))) int eclair_paged_decode_heads_supported(
    std::uint64_t heads, std::uint64_t kv_heads, std::uint64_t head_size,
    std::uint64_t max_rows_per_sequence) {
    return fits(heads) && fits(kv_heads) && fits(head_size) && fits(max_rows_per_sequence) &&
                   eclair::cuda::paged_decode_heads_supported(
                       size(heads), size(kv_heads), size(head_size), size(max_rows_per_sequence))
               ? 1
               : 0;
}

extern "C" __attribute__((visibility("default"))) std::uint64_t
eclair_paged_decode_workspace_elements(std::uint64_t rows, std::uint64_t heads,
                                       std::uint64_t head_size, std::uint64_t splits) {
    if (!fits(rows) || !fits(heads) || !fits(head_size) || !fits(splits)) {
        return 0U;
    }
    return static_cast<std::uint64_t>(eclair::ops::paged_decode_workspace_elements(
        size(rows), size(heads), size(head_size), size(splits)));
}
