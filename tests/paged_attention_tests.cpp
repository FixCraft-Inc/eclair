/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "paged_attention_support.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/host/paged_attention.hpp"
#include "eclair/ops/paged_attention.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::host::paged_decode_reference;
using eclair::host::paged_prefill_reference;
using eclair::ops::PagedDecodeBuffers;
using eclair::ops::PagedPrefillBuffers;
using eclair::ops::Status;
using eclair::ops::validate_paged_decode;
using eclair::ops::validate_paged_prefill;
using eclair::test::DecodeCase;
using eclair::test::make_decode_case;
using eclair::test::make_paged_case;
using eclair::test::PagedCase;

[[nodiscard]] PagedPrefillBuffers views(const PagedCase& paged, std::vector<std::uint16_t>& out) {
    return {{paged.query.data(), paged.query.size()},
            {out.data(), out.size()},
            {paged.keys.data(), paged.keys.size()},
            {paged.values.data(), paged.values.size()},
            {paged.key_scales.data(), paged.key_scales.size()},
            {paged.value_scales.data(), paged.value_scales.size()},
            {paged.table.data(), paged.table.size()}};
}

[[nodiscard]] Status run(const PagedCase& paged, std::vector<std::uint16_t>& out) {
    out.assign(paged.output_elements(), 0U);
    return paged_prefill_reference(paged.shape, paged.layout, views(paged, out));
}

// Stores one token's key and value cell for a KV head, all elements equal.
void set_token(PagedCase& paged, std::size_t position, std::size_t kv_head, std::int8_t key,
               float key_scale, std::int8_t value, float value_scale) {
    const std::size_t cell = paged.cell(position, kv_head);
    for (std::size_t d = 0; d < paged.layout.head_size; ++d) {
        paged.keys[cell + d] = key;
        paged.values[cell + d] = value;
    }
    paged.key_scales[paged.scale_index(position, kv_head)] = key_scale;
    paged.value_scales[paged.scale_index(position, kv_head)] = value_scale;
}

void set_query(PagedCase& paged, std::size_t row, std::size_t head, float value) {
    for (std::size_t d = 0; d < paged.layout.head_size; ++d) {
        paged.query[row * paged.shape.query_row_stride + head * paged.layout.head_size + d] =
            encode_fp16(value);
    }
}

[[nodiscard]] float output(const PagedCase& paged, const std::vector<std::uint16_t>& out,
                           std::size_t row, std::size_t head, std::size_t d = 0U) {
    return decode_fp16(
        out[row * paged.shape.output_row_stride + head * paged.layout.head_size + d]);
}

void test_validation() {
    PagedCase paged = make_paged_case(4, 2, 8, 4, 5, 3, 2);
    std::vector<std::uint16_t> out(paged.output_elements());
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, paged.layout, views(paged, out)) ==
                 Status::Success);

    auto zero = paged.shape;
    zero.rows = 0;
    ECLAIR_CHECK(validate_paged_prefill(zero, paged.layout, {}) == Status::Success);

    auto heads = paged.shape;
    heads.heads = 3;  // Not a multiple of two KV heads.
    ECLAIR_CHECK(validate_paged_prefill(heads, paged.layout, views(paged, out)) ==
                 Status::InvalidArgument);

    auto stride = paged.shape;
    stride.query_row_stride = paged.shape.heads * paged.layout.head_size - 1U;
    ECLAIR_CHECK(validate_paged_prefill(stride, paged.layout, views(paged, out)) ==
                 Status::InvalidArgument);

    auto cells = paged.layout;
    cells.slot_stride = cells.head_size - 1U;  // Neighbouring slots would share a value.
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, cells, views(paged, out)) ==
                 Status::InvalidArgument);

    PagedPrefillBuffers short_table = views(paged, out);
    short_table.block_table.size = 1;  // Eight tokens of block size four need two entries.
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, paged.layout, short_table) ==
                 Status::BufferTooSmall);

    PagedPrefillBuffers short_query = views(paged, out);
    short_query.query.size -= 2U;  // The last row's padding may be missing, not its values.
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, paged.layout, short_query) == Status::Success);
    short_query.query.size -= 1U;
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, paged.layout, short_query) ==
                 Status::BufferTooSmall);

    PagedPrefillBuffers short_scales = views(paged, out);
    short_scales.value_scales.size -= paged.layout.block_size;
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, paged.layout, short_scales) ==
                 Status::BufferTooSmall);

    std::vector<std::uint16_t> overlap(paged.query.size() + paged.output_elements());
    PagedPrefillBuffers aliased = views(paged, out);
    aliased.query = {overlap.data(), paged.query.size()};
    aliased.output = {overlap.data() + 1, paged.output_elements()};
    ECLAIR_CHECK(validate_paged_prefill(paged.shape, paged.layout, aliased) ==
                 Status::InvalidArgument);

    auto huge = paged.shape;
    huge.context = std::numeric_limits<std::size_t>::max();
    ECLAIR_CHECK(validate_paged_prefill(huge, paged.layout, views(paged, out)) ==
                 Status::SizeOverflow);
}

// One visible key: the softmax weight is 1, so the output is that key's value
// rounded twice by the contract, binary16(fp32(int8 * scale)) then binary16
// again, which keeps it. 3 * (1/3) rounds in FP32 to 1, and 5 * 0.1 to 0.5.
void test_single_key_returns_its_value() {
    PagedCase paged = make_paged_case(2, 1, 4, 4, 0, 1);
    set_token(paged, 0, 0, 1, 1.0F, 3, 1.0F / 3.0F);
    set_query(paged, 0, 0, 0.5F);
    set_query(paged, 0, 1, -2.0F);
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(output(paged, out, 0, 0) == 1.0F);
    ECLAIR_CHECK(output(paged, out, 0, 1) == 1.0F);

    set_token(paged, 0, 0, 1, 1.0F, 5, 0.1F);
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(output(paged, out, 0, 0) == 0.5F);
}

// Equal scores weight keys equally, and row i sees keys 0 .. context + i only:
// row 0 (position 1) averages keys 0 and 1 and never sees the huge key 2.
void test_causal_mask_and_equal_weights() {
    PagedCase paged = make_paged_case(1, 1, 4, 2, 1, 2);
    set_token(paged, 0, 0, 0, 1.0F, 2, 1.0F);
    set_token(paged, 1, 0, 0, 1.0F, 6, 1.0F);
    set_token(paged, 2, 0, 0, 1.0F, 100, 1.0F);
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(output(paged, out, 0, 0) == 4.0F);   // Keys 0 and 1 (context 1, row 0).
    ECLAIR_CHECK(output(paged, out, 1, 0) == 36.0F);  // (2 + 6 + 100) / 3.
}

// A larger score dominates by exp(difference): scores 0 and ln 3 weigh 1:3.
void test_softmax_weights() {
    PagedCase paged = make_paged_case(1, 1, 4, 4, 1, 1);
    paged.shape.scale = std::log(3.0F) / 4.0F;  // Query 1 times key 1 summed over 4 elements.
    set_query(paged, 0, 0, 1.0F);
    set_token(paged, 0, 0, 0, 1.0F, 4, 1.0F);
    set_token(paged, 1, 0, 1, 1.0F, 8, 1.0F);
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(std::fabs(output(paged, out, 0, 0) - 7.0F) < 1e-2F);  // (4 + 3 * 8) / 4.
}

// Query heads 0 and 1 read KV head 0, heads 2 and 3 read KV head 1.
void test_grouped_heads_and_block_indirection() {
    PagedCase paged = make_paged_case(4, 2, 4, 2, 3, 1);
    for (std::size_t position = 0; position < 4; ++position) {
        set_token(paged, position, 0, 0, 1.0F, 10, 1.0F);
        set_token(paged, position, 1, 0, 1.0F, -20, 1.0F);
    }
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(output(paged, out, 0, 0) == 10.0F && output(paged, out, 0, 1) == 10.0F);
    ECLAIR_CHECK(output(paged, out, 0, 2) == -20.0F && output(paged, out, 0, 3) == -20.0F);

    paged.table[0] = static_cast<std::int32_t>(paged.layout.blocks);  // Past the last block.
    ECLAIR_CHECK(run(paged, out) == Status::InvalidArgument);
    paged.table[0] = -1;
    ECLAIR_CHECK(run(paged, out) == Status::InvalidArgument);
}

// A NaN scale reaches exactly the rows that attend its key. Slots past the
// tokens are never read.
void test_nan_reaches_only_attending_rows() {
    PagedCase paged = make_paged_case(1, 1, 4, 4, 0, 3);
    for (std::size_t position = 0; position < 3; ++position) {
        set_token(paged, position, 0, 1, 1.0F, 1, 1.0F);
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    paged.value_scales[paged.scale_index(1, 0)] = nan;
    paged.key_scales[paged.scale_index(3, 0)] = nan;  // Unwritten slot of the last block.
    paged.value_scales[paged.scale_index(3, 0)] = nan;
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(output(paged, out, 0, 0) == 1.0F);
    ECLAIR_CHECK(std::isnan(output(paged, out, 1, 0)));
    ECLAIR_CHECK(std::isnan(output(paged, out, 2, 0)));

    paged.value_scales[paged.scale_index(1, 0)] = 1.0F;
    paged.key_scales[paged.scale_index(2, 0)] = nan;
    ECLAIR_CHECK(run(paged, out) == Status::Success);
    ECLAIR_CHECK(output(paged, out, 1, 0) == 1.0F);
    ECLAIR_CHECK(std::isnan(output(paged, out, 2, 0)));
}

void test_head_size_limit() {
    PagedCase paged =
        make_paged_case(1, 1, eclair::host::kPagedPrefillReferenceMaxHeadSize + 8U, 4, 0, 1);
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(paged, out) == Status::InvalidArgument);
}

[[nodiscard]] PagedDecodeBuffers views(const DecodeCase& decode, std::vector<std::uint16_t>& out,
                                       std::vector<float>& workspace) {
    return {{decode.query.data(), decode.query.size()},
            {out.data(), out.size()},
            {decode.keys.data(), decode.keys.size()},
            {decode.values.data(), decode.values.size()},
            {decode.key_scales.data(), decode.key_scales.size()},
            {decode.value_scales.data(), decode.value_scales.size()},
            {decode.tables.data(), decode.tables.size()},
            {decode.starts.data(), decode.starts.size()},
            {decode.lengths.data(), decode.lengths.size()},
            {workspace.data(), workspace.size()}};
}

// The output starts as a 0x7E7E sentinel, so rows the reference must not
// write stay recognizable.
[[nodiscard]] Status run(const DecodeCase& decode, std::vector<std::uint16_t>& out) {
    out.assign(decode.output_elements(), 0x7E7EU);
    std::vector<float> workspace(decode.workspace_elements());
    return paged_decode_reference(decode.shape, decode.layout, views(decode, out, workspace));
}

void set_token(DecodeCase& decode, std::size_t sequence, std::size_t position, std::size_t kv_head,
               std::int8_t key, std::int8_t value) {
    const std::size_t cell = decode.cell(sequence, position, kv_head);
    for (std::size_t d = 0; d < decode.layout.head_size; ++d) {
        decode.keys[cell + d] = key;
        decode.values[cell + d] = value;
    }
    decode.key_scales[decode.scale_index(sequence, position, kv_head)] = 1.0F;
    decode.value_scales[decode.scale_index(sequence, position, kv_head)] = 1.0F;
}

[[nodiscard]] float output(const DecodeCase& decode, const std::vector<std::uint16_t>& out,
                           std::size_t row, std::size_t head) {
    return decode_fp16(out[row * decode.shape.output_row_stride + head * decode.layout.head_size]);
}

void test_decode_validation() {
    const DecodeCase decode = make_decode_case(4, 2, 8, 4, {2, 1}, {5, 9}, 3, 1);
    std::vector<std::uint16_t> out(decode.output_elements());
    std::vector<float> workspace(decode.workspace_elements());
    ECLAIR_CHECK(decode.workspace_elements() == 4U * 4U * 3U * 10U);
    ECLAIR_CHECK(validate_paged_decode(decode.shape, decode.layout,
                                       views(decode, out, workspace)) == Status::Success);

    auto none = decode.shape;
    none.sequences = 0;
    ECLAIR_CHECK(validate_paged_decode(none, decode.layout, {}) == Status::Success);

    for (const auto change :
         {+[](eclair::ops::PagedDecodeShape& s) { s.splits = 0; },
          +[](eclair::ops::PagedDecodeShape& s) { s.max_rows_per_sequence = 0; },
          +[](eclair::ops::PagedDecodeShape& s) { s.max_blocks_per_sequence = 0; },
          +[](eclair::ops::PagedDecodeShape& s) {
              s.block_table_stride = s.max_blocks_per_sequence - 1U;
          },
          +[](eclair::ops::PagedDecodeShape& s) { s.heads = 3; }}) {
        auto shape = decode.shape;
        change(shape);
        ECLAIR_CHECK(validate_paged_decode(shape, decode.layout, views(decode, out, workspace)) ==
                     Status::InvalidArgument);
    }

    PagedDecodeBuffers short_workspace = views(decode, out, workspace);
    short_workspace.workspace.size -= 1U;
    ECLAIR_CHECK(validate_paged_decode(decode.shape, decode.layout, short_workspace) ==
                 Status::BufferTooSmall);
    PagedDecodeBuffers short_starts = views(decode, out, workspace);
    short_starts.query_starts.size = decode.shape.sequences;  // Needs sequences + 1.
    ECLAIR_CHECK(validate_paged_decode(decode.shape, decode.layout, short_starts) ==
                 Status::BufferTooSmall);
    PagedDecodeBuffers short_table = views(decode, out, workspace);
    short_table.block_tables.size =
        decode.shape.block_table_stride + decode.shape.max_blocks_per_sequence - 1U;
    ECLAIR_CHECK(validate_paged_decode(decode.shape, decode.layout, short_table) ==
                 Status::BufferTooSmall);

    // The workspace is written, so it may not overlap the query or the output.
    std::vector<float> shared(decode.query.size() + workspace.size());
    PagedDecodeBuffers aliased = views(decode, out, workspace);
    aliased.query = {reinterpret_cast<const std::uint16_t*>(shared.data()), decode.query.size()};
    aliased.workspace = {shared.data() + 1, workspace.size()};
    ECLAIR_CHECK(validate_paged_decode(decode.shape, decode.layout, aliased) ==
                 Status::InvalidArgument);
    std::vector<std::uint16_t> joint(decode.output_elements() + 2U * workspace.size());
    PagedDecodeBuffers written = views(decode, out, workspace);
    written.output = {joint.data(), decode.output_elements()};
    written.workspace = {reinterpret_cast<float*>(joint.data() + 2), workspace.size()};
    ECLAIR_CHECK(validate_paged_decode(decode.shape, decode.layout, written) ==
                 Status::InvalidArgument);

    const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2U;
    ECLAIR_CHECK(eclair::ops::paged_decode_workspace_elements(huge, 2, 8, 1) == 0U);
    auto overflow = decode.shape;
    overflow.splits = huge;
    ECLAIR_CHECK(validate_paged_decode(overflow, decode.layout, views(decode, out, workspace)) ==
                 Status::SizeOverflow);
}

// Two sequences in one cache, each with its own blocks and values: sequence 0
// has context 3 and two rows, sequence 1 context 5 and one row. Equal scores
// average the visible values, so each row's output is the mean of its causal
// prefix, and KV head 1 serves query heads 2 and 3. The unowned last row
// keeps its sentinel.
void test_decode_sequences_and_causal_rows() {
    DecodeCase decode = make_decode_case(4, 2, 4, 2, {2, 1}, {3, 5}, 1, 1);
    const std::int8_t first[5] = {2, 4, 6, 8, 100};
    for (std::size_t position = 0; position < 5; ++position) {
        set_token(decode, 0, position, 0, 0, first[position]);
        set_token(decode, 0, position, 1, 0, static_cast<std::int8_t>(-first[position]));
    }
    for (std::size_t position = 0; position < 6; ++position) {
        set_token(decode, 1, position, 0, 0, 12);
        set_token(decode, 1, position, 1, 0, -30);
    }
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(decode, out) == Status::Success);
    ECLAIR_CHECK(output(decode, out, 0, 0) == 5.0F);   // (2 + 4 + 6 + 8) / 4, not key 4.
    ECLAIR_CHECK(output(decode, out, 1, 1) == 24.0F);  // (2 + 4 + 6 + 8 + 100) / 5.
    ECLAIR_CHECK(output(decode, out, 0, 3) == -5.0F);
    ECLAIR_CHECK(output(decode, out, 2, 0) == 12.0F);
    ECLAIR_CHECK(output(decode, out, 2, 2) == -30.0F);
    for (std::size_t element = 0; element < decode.shape.output_row_stride; ++element) {
        ECLAIR_CHECK(out[3U * decode.shape.output_row_stride + element] == 0x7E7EU);
    }
}

// Metadata outside the contract's rules and block ids outside the cache are
// refused before any output is written. Entries past a sequence's length are
// never read, even when they hold invalid ids.
void test_decode_refuses_bad_metadata() {
    DecodeCase decode = make_decode_case(2, 1, 4, 2, {2, 1}, {3, 5}, 1);
    eclair::test::fill_random(decode, 4);
    std::vector<std::uint16_t> out;
    ECLAIR_CHECK(run(decode, out) == Status::Success);
    ECLAIR_CHECK(decode.tables[3] == static_cast<std::int32_t>(decode.layout.blocks));

    const auto refused = [&](DecodeCase changed) {
        std::vector<std::uint16_t> result;
        const bool rejected = run(changed, result) == Status::InvalidArgument;
        bool untouched = true;
        for (const std::uint16_t bits : result) {
            untouched = untouched && bits == 0x7E7EU;
        }
        return rejected && untouched;
    };
    DecodeCase changed = decode;
    changed.shape.max_rows_per_sequence = 1;  // Sequence 0 has two rows.
    ECLAIR_CHECK(refused(changed));
    changed = decode;
    changed.starts[1] = 3;  // Past the next start.
    ECLAIR_CHECK(refused(changed));
    changed = decode;
    changed.starts[2] = 4;  // Past the last row.
    ECLAIR_CHECK(refused(changed));
    changed = decode;
    changed.lengths[0] = 1;  // Fewer tokens than rows.
    ECLAIR_CHECK(refused(changed));
    changed = decode;
    changed.lengths[1] = static_cast<std::int32_t>(2U * decode.shape.max_blocks_per_sequence + 1U);
    ECLAIR_CHECK(refused(changed));
    changed = decode;
    changed.tables[decode.shape.block_table_stride] = -1;
    ECLAIR_CHECK(refused(changed));
}

}  // namespace

int main() {
    test_validation();
    test_single_key_returns_its_value();
    test_causal_mask_and_equal_weights();
    test_softmax_weights();
    test_grouped_heads_and_block_indirection();
    test_nan_reaches_only_attending_rows();
    test_head_size_limit();
    test_decode_validation();
    test_decode_sequences_and_causal_rows();
    test_decode_refuses_bad_metadata();
    return eclair::test::finish();
}
