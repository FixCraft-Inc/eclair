/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/paged_attention.hpp"

#include "eclair/formats/fp16.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace eclair::host {
namespace {

// Byte offset of a value cell and element offset of its scale. Validation
// bounded every in-range offset, so these products cannot overflow.
struct Cell {
    std::size_t value = 0;
    std::size_t scale = 0;
};

[[nodiscard]] Cell locate(const ops::PagedInt8CacheLayout& layout, const std::int32_t* table,
                          std::size_t position, std::size_t kv_head) noexcept {
    const auto block = static_cast<std::size_t>(table[position / layout.block_size]);
    const std::size_t slot = position % layout.block_size;
    return {block * layout.block_stride + slot * layout.slot_stride + kv_head * layout.head_stride,
            block * layout.scale_block_stride + slot * layout.scale_slot_stride +
                kv_head * layout.scale_head_stride};
}

// The contract's binary16 rounding of the FP32 product.
[[nodiscard]] double dequantize(std::int8_t value, float scale) noexcept {
    return static_cast<double>(
        formats::decode_fp16(formats::encode_fp16(static_cast<float>(value) * scale)));
}

}  // namespace

ops::Status paged_prefill_reference(const ops::PagedPrefillShape& shape,
                                    const ops::PagedInt8CacheLayout& layout,
                                    const ops::PagedPrefillBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_paged_prefill(shape, layout, buffers);
    if (status != ops::Status::Success || shape.rows == 0U) {
        return status;
    }
    if (layout.head_size > kPagedPrefillReferenceMaxHeadSize) {
        return ops::Status::InvalidArgument;
    }
    const std::size_t tokens = shape.context + shape.rows;
    const std::size_t entries = (tokens + layout.block_size - 1U) / layout.block_size;
    for (std::size_t entry = 0; entry < entries; ++entry) {
        const std::int32_t id = buffers.block_table.data[entry];
        if (id < 0 || static_cast<std::size_t>(id) >= layout.blocks) {
            return ops::Status::InvalidArgument;
        }
    }
    const std::size_t size = layout.head_size;
    const std::size_t group = shape.heads / layout.kv_heads;
    const double scale = static_cast<double>(shape.scale);
    std::array<double, kPagedPrefillReferenceMaxHeadSize> sums{};
    for (std::size_t row = 0; row < shape.rows; ++row) {
        const std::size_t keys = shape.context + row + 1U;
        for (std::size_t head = 0; head < shape.heads; ++head) {
            const std::size_t kv_head = head / group;
            const std::uint16_t* query =
                buffers.query.data + row * shape.query_row_stride + head * size;
            const auto score = [&](std::size_t position) noexcept {
                const Cell cell = locate(layout, buffers.block_table.data, position, kv_head);
                const float key_scale = buffers.key_scales.data[cell.scale];
                double dot = 0.0;
                for (std::size_t d = 0; d < size; ++d) {
                    dot += static_cast<double>(formats::decode_fp16(query[d])) *
                           dequantize(buffers.keys.data[cell.value + d], key_scale);
                }
                return scale * dot;
            };
            // std::max keeps the running maximum when a score is NaN. The NaN
            // still reaches the sum below, so the row becomes NaN.
            double largest = -std::numeric_limits<double>::infinity();
            for (std::size_t position = 0; position < keys; ++position) {
                largest = std::max(largest, score(position));
            }
            double total = 0.0;
            std::fill_n(sums.begin(), size, 0.0);
            for (std::size_t position = 0; position < keys; ++position) {
                const double weight = std::exp(score(position) - largest);
                const Cell cell = locate(layout, buffers.block_table.data, position, kv_head);
                const float value_scale = buffers.value_scales.data[cell.scale];
                total += weight;
                for (std::size_t d = 0; d < size; ++d) {
                    sums[d] +=
                        weight * dequantize(buffers.values.data[cell.value + d], value_scale);
                }
            }
            std::uint16_t* output =
                buffers.output.data + row * shape.output_row_stride + head * size;
            for (std::size_t d = 0; d < size; ++d) {
                output[d] = formats::encode_fp16(static_cast<float>(sums[d] / total));
            }
        }
    }
    return ops::Status::Success;
}

namespace {

// One sequence's chunk as a prefill call. Returns false for metadata outside
// the decode contract's rules.
[[nodiscard]] bool chunk(const ops::PagedDecodeShape& shape,
                         const ops::PagedInt8CacheLayout& layout,
                         const ops::PagedDecodeBuffers& buffers, std::size_t sequence,
                         ops::PagedPrefillShape& prefill,
                         ops::PagedPrefillBuffers& views) noexcept {
    const std::int32_t first = buffers.query_starts.data[sequence];
    const std::int32_t last = buffers.query_starts.data[sequence + 1U];
    const std::int32_t length = buffers.seq_lens.data[sequence];
    if (first < 0 || last < first || static_cast<std::size_t>(last) > shape.rows || length < 0) {
        return false;
    }
    const auto rows = static_cast<std::size_t>(last - first);
    const auto tokens = static_cast<std::size_t>(length);
    // Validation bounded max_blocks_per_sequence * block_size.
    if (rows > shape.max_rows_per_sequence || rows > tokens ||
        tokens > shape.max_blocks_per_sequence * layout.block_size) {
        return false;
    }
    const auto row = static_cast<std::size_t>(first);
    prefill = {shape.heads, rows, tokens - rows, shape.query_row_stride, shape.output_row_stride,
               shape.scale};
    const std::size_t entries = (tokens + layout.block_size - 1U) / layout.block_size;
    // A row of zero chunk rows touches no query or output element.
    views = {{rows == 0U ? nullptr : buffers.query.data + row * shape.query_row_stride,
              rows == 0U ? 0U : buffers.query.size - row * shape.query_row_stride},
             {rows == 0U ? nullptr : buffers.output.data + row * shape.output_row_stride,
              rows == 0U ? 0U : buffers.output.size - row * shape.output_row_stride},
             buffers.keys,
             buffers.values,
             buffers.key_scales,
             buffers.value_scales,
             {buffers.block_tables.data + sequence * shape.block_table_stride, entries}};
    return true;
}

}  // namespace

ops::Status paged_decode_reference(const ops::PagedDecodeShape& shape,
                                   const ops::PagedInt8CacheLayout& layout,
                                   const ops::PagedDecodeBuffers& buffers) noexcept {
    const ops::Status status = ops::validate_paged_decode(shape, layout, buffers);
    if (status != ops::Status::Success || shape.rows == 0U || shape.sequences == 0U) {
        return status;
    }
    if (layout.head_size > kPagedPrefillReferenceMaxHeadSize) {
        return ops::Status::InvalidArgument;
    }
    for (std::size_t sequence = 0; sequence < shape.sequences; ++sequence) {
        ops::PagedPrefillShape prefill;
        ops::PagedPrefillBuffers views;
        if (!chunk(shape, layout, buffers, sequence, prefill, views)) {
            return ops::Status::InvalidArgument;
        }
        const std::size_t tokens = prefill.context + prefill.rows;
        for (std::size_t entry = 0; entry * layout.block_size < tokens; ++entry) {
            const std::int32_t id = views.block_table.data[entry];
            if (id < 0 || static_cast<std::size_t>(id) >= layout.blocks) {
                return ops::Status::InvalidArgument;
            }
        }
    }
    for (std::size_t sequence = 0; sequence < shape.sequences; ++sequence) {
        ops::PagedPrefillShape prefill;
        ops::PagedPrefillBuffers views;
        static_cast<void>(chunk(shape, layout, buffers, sequence, prefill, views));
        const ops::Status result = paged_prefill_reference(prefill, layout, views);
        if (result != ops::Status::Success) {
            return result;
        }
    }
    return ops::Status::Success;
}

}  // namespace eclair::host
