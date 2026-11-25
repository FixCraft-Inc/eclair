/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/ops/buffer.hpp"

#include <cstddef>
#include <cstdint>

// Experimental causal prefill attention over a paged int8 KV cache with one
// FP32 scale per token and KV head, the layout of vLLM's int8_per_token_head
// cache. See docs/PAGED_ATTENTION.md.
namespace eclair::ops {

// Keys and values live in separate arrays of int8 cells that share strides.
// The cell of KV head g for token slot s of block b starts at byte
// b * block_stride + s * slot_stride + g * head_stride and holds head_size
// values. Its scale is the FP32 element at
// b * scale_block_stride + s * scale_slot_stride + g * scale_head_stride.
struct PagedInt8CacheLayout {
    std::size_t blocks = 0;      // Blocks addressable from the base pointers.
    std::size_t block_size = 0;  // Token slots per block.
    std::size_t kv_heads = 0;
    std::size_t head_size = 0;
    std::size_t block_stride = 0;        // Bytes.
    std::size_t slot_stride = 0;         // Bytes.
    std::size_t head_stride = 0;         // Bytes.
    std::size_t scale_block_stride = 0;  // FP32 elements.
    std::size_t scale_slot_stride = 0;   // FP32 elements.
    std::size_t scale_head_stride = 0;   // FP32 elements.
};

// One chunk of `rows` query tokens at positions context .. context + rows - 1
// of a sequence whose first context + rows tokens are already in the cache.
// Query head h reads KV head h / (heads / kv_heads). For query row i and key
// position j <= context + i:
//
//   k_j    = binary16(fp32(int8 key * key scale))
//   v_j    = binary16(fp32(int8 value * value scale))
//   s_ij   = scale * sum_d q_i[d] * k_j[d]
//   out_i  = binary16(sum_j softmax_j(s_ij) * v_j)
//
// Later keys are masked. Slots at or past context + rows are never read, so
// unwritten slots cannot reach the output. A backend may multiply a masked
// key's value by a zero weight, so a NaN value of a key inside the chunk can
// also reach earlier rows of that chunk. Query and output rows hold
// `heads` contiguous vectors of head_size binary16 values, row r starting at
// r * row stride. Backends accumulate in FP32 or better in an unspecified
// order and may round probabilities to binary16. The host reference uses
// FP64. A NaN query, key or scale of an attended key makes that row NaN.
struct PagedPrefillShape {
    std::size_t heads = 0;
    std::size_t rows = 0;
    std::size_t context = 0;
    std::size_t query_row_stride = 0;   // Elements.
    std::size_t output_row_stride = 0;  // Elements.
    float scale = 0.0F;
};

struct PagedPrefillBuffers {
    BufferView<const std::uint16_t> query;  // binary16 bits.
    BufferView<std::uint16_t> output;       // binary16 bits.
    BufferView<const std::int8_t> keys;
    BufferView<const std::int8_t> values;
    BufferView<const float> key_scales;
    BufferView<const float> value_scales;
    // The sequence's block ids in order, at least ceil((context + rows) /
    // block_size) of them. Device backends cannot read ids during validation.
    // They skip, rather than read, a block id outside [0, blocks).
    BufferView<const std::int32_t> block_table;
};

// Rejects inconsistent heads, zero sizes, strides that make rows or cells
// overlap, short or misaligned buffers, an output that overlaps an input, a
// short block table and size overflow. Zero rows is a no-op that needs no
// buffers. Reads no values.
[[nodiscard]] Status validate_paged_prefill(const PagedPrefillShape& shape,
                                            const PagedInt8CacheLayout& layout,
                                            const PagedPrefillBuffers& buffers) noexcept;

// A batch of short chunks, as an engine's decode and speculative verification
// steps produce them. Sequence n owns query rows query_starts[n] ..
// query_starts[n + 1] - 1, which are the last of its seq_lens[n] cached
// tokens, and its block ids are row n of `block_tables`, block_table_stride
// entries after row n - 1. Each chunk follows the prefill semantics above with
// rows = query_starts[n + 1] - query_starts[n] and context = seq_lens[n] - rows.
//
// The metadata lives beside the device buffers and may change between replays
// of a captured CUDA graph, so validation reads none of it. Device backends
// skip a sequence that breaks
//
//   0 <= query_starts[n] <= query_starts[n + 1] <= rows
//   query_starts[n + 1] - query_starts[n] <= max_rows_per_sequence
//   query_starts[n + 1] - query_starts[n] <= seq_lens[n]
//   seq_lens[n] <= max_blocks_per_sequence * block_size
//
// and leave its output rows unwritten, as they do for rows no sequence owns.
// The host reference refuses such metadata. Device backends split each
// context into `splits` slices and merge their partial softmax sums through a
// caller-owned FP32 workspace of paged_decode_workspace_elements values, whose
// contents need no initialization and are unspecified afterwards. The other
// layout and NaN rules are those of PagedPrefillShape, and so are the
// roundings, except that a backend may keep keys exact and multiply the FP32
// score by the key scale, and may round the value scale to binary16 before
// the product, giving binary16(int8 * binary16(value scale)).
struct PagedDecodeShape {
    std::size_t heads = 0;
    std::size_t sequences = 0;
    std::size_t rows = 0;  // Query and output rows of all sequences.
    std::size_t max_rows_per_sequence = 0;
    std::size_t max_blocks_per_sequence = 0;  // Readable entries of each table row.
    std::size_t block_table_stride = 0;       // Entries.
    std::size_t splits = 0;
    std::size_t query_row_stride = 0;   // Elements.
    std::size_t output_row_stride = 0;  // Elements.
    float scale = 0.0F;
};

struct PagedDecodeBuffers {
    BufferView<const std::uint16_t> query;  // binary16 bits.
    BufferView<std::uint16_t> output;       // binary16 bits.
    BufferView<const std::int8_t> keys;
    BufferView<const std::int8_t> values;
    BufferView<const float> key_scales;
    BufferView<const float> value_scales;
    BufferView<const std::int32_t> block_tables;
    BufferView<const std::int32_t> query_starts;  // sequences + 1 entries.
    BufferView<const std::int32_t> seq_lens;      // sequences entries.
    BufferView<float> workspace;
};

// rows * heads * splits * (head_size + 2), or 0 when that overflows.
[[nodiscard]] std::size_t paged_decode_workspace_elements(std::size_t rows, std::size_t heads,
                                                          std::size_t head_size,
                                                          std::size_t splits) noexcept;

// The prefill checks for heads, cell strides and buffers, plus a zero split
// or row limit, a table stride shorter than a row, short metadata or workspace,
// and a workspace or output that overlaps another buffer. Zero sequences or
// rows is a no-op that needs no buffers. Reads no values.
[[nodiscard]] Status validate_paged_decode(const PagedDecodeShape& shape,
                                           const PagedInt8CacheLayout& layout,
                                           const PagedDecodeBuffers& buffers) noexcept;

}  // namespace eclair::ops
