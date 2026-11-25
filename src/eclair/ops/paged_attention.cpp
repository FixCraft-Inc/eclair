/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/ops/paged_attention.hpp"

#include "buffer_checks.hpp"

#include <limits>

namespace eclair::ops {
namespace {

using detail::AddressRange;
using detail::multiply;
using detail::overlaps;
using detail::range;

[[nodiscard]] bool add(std::size_t a, std::size_t b, std::size_t& result) noexcept {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        return false;
    }
    result = a + b;
    return true;
}

// Elements from the first to one past the last addressed element of an array
// indexed by three strides, each over `count` positions, plus `last` elements.
[[nodiscard]] bool extent(const std::size_t (&counts)[3], const std::size_t (&strides)[3],
                          std::size_t last, std::size_t& result) noexcept {
    std::size_t total = last;
    for (std::size_t axis = 0; axis < 3U; ++axis) {
        std::size_t span = 0;
        if (!multiply(counts[axis] - 1U, strides[axis], span) || !add(total, span, total)) {
            return false;
        }
    }
    result = total;
    return true;
}

// Heads, a nonempty layout and cell strides that never let distinct heads,
// slots or blocks share a value.
[[nodiscard]] bool consistent(std::size_t heads, const PagedInt8CacheLayout& layout) noexcept {
    return heads != 0U && layout.kv_heads != 0U && heads % layout.kv_heads == 0U &&
           layout.head_size != 0U && layout.blocks != 0U && layout.block_size != 0U &&
           layout.head_stride >= layout.head_size && layout.slot_stride >= layout.head_size &&
           layout.block_stride >= layout.head_size && layout.scale_head_stride != 0U &&
           layout.scale_slot_stride != 0U && layout.scale_block_stride != 0U;
}

// Readable bytes of each of keys and values, and elements of each scale array.
[[nodiscard]] bool cache_extents(const PagedInt8CacheLayout& layout, std::size_t& cache_bytes,
                                 std::size_t& scale_elements) noexcept {
    const std::size_t counts[3] = {layout.blocks, layout.block_size, layout.kv_heads};
    return extent(counts, {layout.block_stride, layout.slot_stride, layout.head_stride},
                  layout.head_size, cache_bytes) &&
           extent(counts,
                  {layout.scale_block_stride, layout.scale_slot_stride, layout.scale_head_stride},
                  1U, scale_elements);
}

// Elements from the first row to the end of the last row's head vectors.
[[nodiscard]] bool rows_extent(std::size_t rows, std::size_t stride, std::size_t row_elements,
                               std::size_t& result) noexcept {
    return multiply(rows - 1U, stride, result) && add(result, row_elements, result);
}

}  // namespace

Status validate_paged_prefill(const PagedPrefillShape& shape, const PagedInt8CacheLayout& layout,
                              const PagedPrefillBuffers& buffers) noexcept {
    if (shape.rows == 0U) {
        return Status::Success;
    }
    if (!consistent(shape.heads, layout)) {
        return Status::InvalidArgument;
    }
    std::size_t row_elements = 0;
    std::size_t tokens = 0;
    if (!multiply(shape.heads, layout.head_size, row_elements) ||
        !add(shape.context, shape.rows, tokens)) {
        return Status::SizeOverflow;
    }
    if (shape.query_row_stride < row_elements || shape.output_row_stride < row_elements) {
        return Status::InvalidArgument;
    }
    std::size_t query_elements = 0;
    std::size_t output_elements = 0;
    std::size_t cache_bytes = 0;
    std::size_t scale_elements = 0;
    if (!rows_extent(shape.rows, shape.query_row_stride, row_elements, query_elements) ||
        !rows_extent(shape.rows, shape.output_row_stride, row_elements, output_elements) ||
        !cache_extents(layout, cache_bytes, scale_elements)) {
        return Status::SizeOverflow;
    }
    const std::size_t table_entries =
        tokens / layout.block_size + (tokens % layout.block_size != 0U ? 1U : 0U);
    AddressRange output;
    AddressRange inputs[6];
    const Status checks[] = {
        range(buffers.output, output_elements, output),
        range(buffers.query, query_elements, inputs[0]),
        range(buffers.keys, cache_bytes, inputs[1]),
        range(buffers.values, cache_bytes, inputs[2]),
        range(buffers.key_scales, scale_elements, inputs[3]),
        range(buffers.value_scales, scale_elements, inputs[4]),
        range(buffers.block_table, table_entries, inputs[5]),
    };
    for (const Status check : checks) {
        if (check != Status::Success) {
            return check;
        }
    }
    for (const AddressRange& input : inputs) {
        if (overlaps(output, input)) {
            return Status::InvalidArgument;
        }
    }
    return Status::Success;
}

std::size_t paged_decode_workspace_elements(std::size_t rows, std::size_t heads,
                                            std::size_t head_size, std::size_t splits) noexcept {
    // A maximum, a sum and head_size accumulators per row, head and split.
    std::size_t per_split = 0;
    std::size_t result = 0;
    if (!add(head_size, 2U, per_split) || !multiply(rows, heads, result) ||
        !multiply(result, splits, result) || !multiply(result, per_split, result)) {
        return 0U;
    }
    return result;
}

Status validate_paged_decode(const PagedDecodeShape& shape, const PagedInt8CacheLayout& layout,
                             const PagedDecodeBuffers& buffers) noexcept {
    if (shape.rows == 0U || shape.sequences == 0U) {
        return Status::Success;
    }
    if (!consistent(shape.heads, layout) || shape.splits == 0U ||
        shape.max_rows_per_sequence == 0U || shape.max_blocks_per_sequence == 0U ||
        (shape.sequences > 1U && shape.block_table_stride < shape.max_blocks_per_sequence)) {
        return Status::InvalidArgument;
    }
    std::size_t row_elements = 0;
    std::size_t tokens = 0;
    if (!multiply(shape.heads, layout.head_size, row_elements) ||
        !multiply(shape.max_blocks_per_sequence, layout.block_size, tokens)) {
        return Status::SizeOverflow;
    }
    if (shape.query_row_stride < row_elements || shape.output_row_stride < row_elements) {
        return Status::InvalidArgument;
    }
    std::size_t query_elements = 0;
    std::size_t output_elements = 0;
    std::size_t cache_bytes = 0;
    std::size_t scale_elements = 0;
    std::size_t table_entries = 0;
    std::size_t starts = 0;
    const std::size_t workspace_elements =
        paged_decode_workspace_elements(shape.rows, shape.heads, layout.head_size, shape.splits);
    if (!rows_extent(shape.rows, shape.query_row_stride, row_elements, query_elements) ||
        !rows_extent(shape.rows, shape.output_row_stride, row_elements, output_elements) ||
        !cache_extents(layout, cache_bytes, scale_elements) ||
        !rows_extent(shape.sequences, shape.block_table_stride, shape.max_blocks_per_sequence,
                     table_entries) ||
        !add(shape.sequences, 1U, starts) || workspace_elements == 0U) {
        return Status::SizeOverflow;
    }
    AddressRange written[2];
    AddressRange inputs[9];
    const Status checks[] = {
        range(buffers.output, output_elements, written[0]),
        range(buffers.workspace, workspace_elements, written[1]),
        range(buffers.query, query_elements, inputs[0]),
        range(buffers.keys, cache_bytes, inputs[1]),
        range(buffers.values, cache_bytes, inputs[2]),
        range(buffers.key_scales, scale_elements, inputs[3]),
        range(buffers.value_scales, scale_elements, inputs[4]),
        range(buffers.block_tables, table_entries, inputs[5]),
        range(buffers.query_starts, starts, inputs[6]),
        range(buffers.seq_lens, shape.sequences, inputs[7]),
    };
    for (const Status check : checks) {
        if (check != Status::Success) {
            return check;
        }
    }
    inputs[8] = written[1];
    for (std::size_t i = 0; i < 9U; ++i) {
        if (overlaps(written[0], inputs[i]) || (i < 8U && overlaps(written[1], inputs[i]))) {
            return Status::InvalidArgument;
        }
    }
    return Status::Success;
}

}  // namespace eclair::ops
