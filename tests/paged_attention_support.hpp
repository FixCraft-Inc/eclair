/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Host-side paged int8 caches for the paged prefill tests. C++17, for CUDA
// units too. Cells are laid out like vLLM's HND cache, [block][KV head][slot]
// with eight spare bytes per cell, and the block table visits blocks in
// reverse with one spare block, so strides and indirection both matter.

#include "eclair/formats/fp16.hpp"
#include "eclair/ops/paged_attention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace eclair::test {

struct PagedCase {
    ops::PagedPrefillShape shape;
    ops::PagedInt8CacheLayout layout;
    std::vector<std::uint16_t> query;
    std::vector<std::int8_t> keys;
    std::vector<std::int8_t> values;
    std::vector<float> key_scales;
    std::vector<float> value_scales;
    std::vector<std::int32_t> table;

    [[nodiscard]] std::size_t output_elements() const {
        return shape.rows * shape.output_row_stride;
    }
    // Byte offset of a value cell and element offset of its scale.
    [[nodiscard]] std::size_t cell(std::size_t position, std::size_t kv_head) const {
        const auto block = static_cast<std::size_t>(table[position / layout.block_size]);
        return block * layout.block_stride + (position % layout.block_size) * layout.slot_stride +
               kv_head * layout.head_stride;
    }
    [[nodiscard]] std::size_t scale_index(std::size_t position, std::size_t kv_head) const {
        const auto block = static_cast<std::size_t>(table[position / layout.block_size]);
        return block * layout.scale_block_stride +
               (position % layout.block_size) * layout.scale_slot_stride +
               kv_head * layout.scale_head_stride;
    }
};

// Zero-filled storage for every slot of `context + rows` tokens, rounded up
// to whole blocks plus one spare block, with query rows `row_padding` values
// longer than heads * head_size.
[[nodiscard]] inline PagedCase make_paged_case(std::size_t heads, std::size_t kv_heads,
                                               std::size_t head_size, std::size_t block_size,
                                               std::size_t context, std::size_t rows,
                                               std::size_t row_padding = 0U) {
    PagedCase result;
    const std::size_t tokens = context + rows;
    const std::size_t used = (tokens + block_size - 1U) / block_size;
    const std::size_t cell = head_size + 8U;
    result.layout = {used + 1U,
                     block_size,
                     kv_heads,
                     head_size,
                     kv_heads * block_size * cell,
                     cell,
                     block_size * cell,
                     kv_heads * block_size,
                     1U,
                     block_size};
    const std::size_t row_elements = heads * head_size + row_padding;
    result.shape = {heads, rows, context, row_elements, row_elements, 1.0F};
    result.query.assign(rows * row_elements, 0U);
    result.keys.assign((used + 1U) * result.layout.block_stride, 0);
    result.values.assign(result.keys.size(), 0);
    result.key_scales.assign((used + 1U) * kv_heads * block_size, 0.0F);
    result.value_scales.assign(result.key_scales.size(), 0.0F);
    for (std::size_t entry = 0; entry < used; ++entry) {
        result.table.push_back(static_cast<std::int32_t>(used - entry));
    }
    return result;
}

// Standard-normal binary16 queries, uniform int8 cells, scales in [0.005,
// 0.03] and the usual 1 / sqrt(head size) score scale, so scores spread over a
// few units and the softmax is neither flat nor one-hot. Spare bytes and
// slots past the tokens stay zero.
inline void fill_random(PagedCase& paged, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> byte(-127, 127);
    std::uniform_real_distribution<float> scale(0.005F, 0.03F);
    const std::size_t heads = paged.shape.heads;
    const std::size_t size = paged.layout.head_size;
    for (std::size_t row = 0; row < paged.shape.rows; ++row) {
        for (std::size_t element = 0; element < heads * size; ++element) {
            paged.query[row * paged.shape.query_row_stride + element] =
                formats::encode_fp16(normal(generator));
        }
    }
    const std::size_t tokens = paged.shape.context + paged.shape.rows;
    for (std::size_t position = 0; position < tokens; ++position) {
        for (std::size_t kv_head = 0; kv_head < paged.layout.kv_heads; ++kv_head) {
            const std::size_t cell = paged.cell(position, kv_head);
            for (std::size_t d = 0; d < size; ++d) {
                paged.keys[cell + d] = static_cast<std::int8_t>(byte(generator));
                paged.values[cell + d] = static_cast<std::int8_t>(byte(generator));
            }
            const std::size_t index = paged.scale_index(position, kv_head);
            paged.key_scales[index] = scale(generator);
            paged.value_scales[index] = scale(generator);
        }
    }
    paged.shape.scale = 1.0F / std::sqrt(static_cast<float>(size));
}

// A batched decode over one shared cache. Sequence n has rows[n] query rows
// after contexts[n] cached tokens and its own blocks, numbered from the end of
// the cache so the tables are not in storage order. Block 0 is spare. Table
// rows hold one extra entry, set to an id past the cache, which no backend
// may read, and `unowned` query rows follow the last sequence.
struct DecodeCase {
    ops::PagedDecodeShape shape;
    ops::PagedInt8CacheLayout layout;
    std::vector<std::uint16_t> query;
    std::vector<std::int8_t> keys;
    std::vector<std::int8_t> values;
    std::vector<float> key_scales;
    std::vector<float> value_scales;
    std::vector<std::int32_t> tables;
    std::vector<std::int32_t> starts;
    std::vector<std::int32_t> lengths;

    [[nodiscard]] std::size_t output_elements() const {
        return shape.rows * shape.output_row_stride;
    }
    [[nodiscard]] std::size_t workspace_elements() const {
        return ops::paged_decode_workspace_elements(shape.rows, shape.heads, layout.head_size,
                                                    shape.splits);
    }
    [[nodiscard]] std::size_t block(std::size_t sequence, std::size_t position) const {
        return static_cast<std::size_t>(
            tables[sequence * shape.block_table_stride + position / layout.block_size]);
    }
    [[nodiscard]] std::size_t cell(std::size_t sequence, std::size_t position,
                                   std::size_t kv_head) const {
        return block(sequence, position) * layout.block_stride +
               (position % layout.block_size) * layout.slot_stride + kv_head * layout.head_stride;
    }
    [[nodiscard]] std::size_t scale_index(std::size_t sequence, std::size_t position,
                                          std::size_t kv_head) const {
        return block(sequence, position) * layout.scale_block_stride +
               (position % layout.block_size) * layout.scale_slot_stride +
               kv_head * layout.scale_head_stride;
    }
};

[[nodiscard]] inline DecodeCase make_decode_case(std::size_t heads, std::size_t kv_heads,
                                                 std::size_t head_size, std::size_t block_size,
                                                 const std::vector<std::size_t>& rows,
                                                 const std::vector<std::size_t>& contexts,
                                                 std::size_t splits, std::size_t unowned = 0U) {
    DecodeCase result;
    std::size_t max_blocks = 0;
    std::size_t total_blocks = 1;  // The spare block.
    std::size_t total_rows = unowned;
    std::size_t max_rows = 1;
    for (std::size_t n = 0; n < rows.size(); ++n) {
        const std::size_t used = (contexts[n] + rows[n] + block_size - 1U) / block_size;
        max_blocks = used > max_blocks ? used : max_blocks;
        total_blocks += used;
        total_rows += rows[n];
        max_rows = rows[n] > max_rows ? rows[n] : max_rows;
    }
    const std::size_t cell = head_size + 8U;
    result.layout = {total_blocks,
                     block_size,
                     kv_heads,
                     head_size,
                     kv_heads * block_size * cell,
                     cell,
                     block_size * cell,
                     kv_heads * block_size,
                     1U,
                     block_size};
    const std::size_t row_elements = heads * head_size;
    const std::size_t stride = max_blocks + 1U;
    result.shape = {heads,  rows.size(), total_rows,   max_rows,     max_blocks,
                    stride, splits,      row_elements, row_elements, 1.0F};
    result.query.assign(total_rows * row_elements, 0U);
    result.keys.assign(total_blocks * result.layout.block_stride, 0);
    result.values.assign(result.keys.size(), 0);
    result.key_scales.assign(total_blocks * kv_heads * block_size, 0.0F);
    result.value_scales.assign(result.key_scales.size(), 0.0F);
    result.tables.assign(rows.size() * stride, static_cast<std::int32_t>(total_blocks));
    std::size_t next_block = total_blocks - 1U;
    std::size_t row = 0;
    for (std::size_t n = 0; n < rows.size(); ++n) {
        const std::size_t used = (contexts[n] + rows[n] + block_size - 1U) / block_size;
        for (std::size_t entry = 0; entry < used; ++entry) {
            result.tables[n * stride + entry] = static_cast<std::int32_t>(next_block--);
        }
        result.starts.push_back(static_cast<std::int32_t>(row));
        result.lengths.push_back(static_cast<std::int32_t>(contexts[n] + rows[n]));
        row += rows[n];
    }
    result.starts.push_back(static_cast<std::int32_t>(row));
    return result;
}

// fill_random's distributions for every written token of every sequence.
inline void fill_random(DecodeCase& decode, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> byte(-127, 127);
    std::uniform_real_distribution<float> scale(0.005F, 0.03F);
    for (std::uint16_t& value : decode.query) {
        value = formats::encode_fp16(normal(generator));
    }
    const std::size_t size = decode.layout.head_size;
    for (std::size_t n = 0; n < decode.shape.sequences; ++n) {
        const auto tokens = static_cast<std::size_t>(decode.lengths[n]);
        for (std::size_t position = 0; position < tokens; ++position) {
            for (std::size_t kv_head = 0; kv_head < decode.layout.kv_heads; ++kv_head) {
                const std::size_t cell = decode.cell(n, position, kv_head);
                for (std::size_t d = 0; d < size; ++d) {
                    decode.keys[cell + d] = static_cast<std::int8_t>(byte(generator));
                    decode.values[cell + d] = static_cast<std::int8_t>(byte(generator));
                }
                const std::size_t index = decode.scale_index(n, position, kv_head);
                decode.key_scales[index] = scale(generator);
                decode.value_scales[index] = scale(generator);
            }
        }
    }
    decode.shape.scale = 1.0F / std::sqrt(static_cast<float>(size));
}

}  // namespace eclair::test
