/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

// Batched paged int8 decode attention on the GPU against the FP64 host
// reference, the split merge's NaN rules, skipped metadata, replay of a
// captured CUDA graph with changed metadata, rejection of unsupported
// arguments, and decode timings at the shapes of Qwen3.8 27B's attention
// layers (24 query heads, 4 KV heads, head size 256, 1552-token cache
// blocks). The tolerance covers FP32 accumulation order, binary16
// probabilities and the binary16 output rounding, as for the prefill kernel.

#include "cuda_support.hpp"
#include "paged_attention_support.hpp"

#include "eclair/cuda/paged_attention.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/host/paged_attention.hpp"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using eclair::cuda::launch_paged_decode;
using eclair::formats::decode_fp16;
using eclair::ops::PagedDecodeBuffers;
using eclair::ops::Status;
using eclair::test::checked;
using eclair::test::DecodeCase;
using eclair::test::DeviceBuffer;
using eclair::test::make_decode_case;
using eclair::test::Stream;

constexpr float kTolerance = 8e-3F;
constexpr std::uint16_t kSentinel = 0x7E7EU;

struct DeviceCase {
    DeviceBuffer<std::uint16_t> query;
    DeviceBuffer<std::uint16_t> output;
    DeviceBuffer<std::int8_t> keys;
    DeviceBuffer<std::int8_t> values;
    DeviceBuffer<float> key_scales;
    DeviceBuffer<float> value_scales;
    DeviceBuffer<std::int32_t> tables;
    DeviceBuffer<std::int32_t> starts;
    DeviceBuffer<std::int32_t> lengths;
    DeviceBuffer<float> workspace;

    DeviceCase(const DecodeCase& decode, cudaStream_t stream)
        : query(decode.query, stream),
          output(decode.output_elements()),
          keys(decode.keys, stream),
          values(decode.values, stream),
          key_scales(decode.key_scales, stream),
          value_scales(decode.value_scales, stream),
          tables(decode.tables, stream),
          starts(decode.starts, stream),
          lengths(decode.lengths, stream),
          workspace(decode.workspace_elements()) {
        checked(cudaMemsetAsync(output.data(), 0x7E, output.size() * 2U, stream),
                "output sentinel");
        // The workspace needs no initialization. NaN bytes check that.
        checked(cudaMemsetAsync(workspace.data(), 0xFF, workspace.size() * 4U, stream),
                "workspace fill");
    }

    [[nodiscard]] PagedDecodeBuffers buffers() const {
        return {{query.data(), query.size()},
                {output.data(), output.size()},
                {keys.data(), keys.size()},
                {values.data(), values.size()},
                {key_scales.data(), key_scales.size()},
                {value_scales.data(), value_scales.size()},
                {tables.data(), tables.size()},
                {starts.data(), starts.size()},
                {lengths.data(), lengths.size()},
                {workspace.data(), workspace.size()}};
    }
};

// The reference output with the sentinel in rows it does not write.
[[nodiscard]] std::vector<std::uint16_t> reference(const DecodeCase& decode) {
    std::vector<std::uint16_t> expected(decode.output_elements(), kSentinel);
    std::vector<float> workspace(decode.workspace_elements());
    const PagedDecodeBuffers host{{decode.query.data(), decode.query.size()},
                                  {expected.data(), expected.size()},
                                  {decode.keys.data(), decode.keys.size()},
                                  {decode.values.data(), decode.values.size()},
                                  {decode.key_scales.data(), decode.key_scales.size()},
                                  {decode.value_scales.data(), decode.value_scales.size()},
                                  {decode.tables.data(), decode.tables.size()},
                                  {decode.starts.data(), decode.starts.size()},
                                  {decode.lengths.data(), decode.lengths.size()},
                                  {workspace.data(), workspace.size()}};
    if (eclair::host::paged_decode_reference(decode.shape, decode.layout, host) !=
        Status::Success) {
        throw std::runtime_error("reference rejected a test case");
    }
    return expected;
}

[[nodiscard]] std::vector<std::uint16_t> run_gpu(const DecodeCase& decode) {
    Stream stream;
    DeviceCase device(decode, stream.get());
    checked(launch_paged_decode(decode.shape, decode.layout, device.buffers(), stream.get()),
            "paged decode");
    return device.output.download(stream.get());
}

// Compares every element of `actual` with `expected`. NaN must match NaN and
// the sentinel must match the sentinel. Returns the largest difference.
float compare(const DecodeCase& decode, const std::vector<std::uint16_t>& actual,
              const std::vector<std::uint16_t>& expected, const char* name) {
    float largest = 0.0F;
    std::size_t mismatches = 0;
    const std::size_t width = decode.shape.heads * decode.layout.head_size;
    for (std::size_t row = 0; row < decode.shape.rows; ++row) {
        for (std::size_t element = 0; element < width; ++element) {
            const std::size_t index = row * decode.shape.output_row_stride + element;
            if (expected[index] == kSentinel || actual[index] == kSentinel) {
                mismatches += expected[index] != actual[index] ? 1U : 0U;
                continue;
            }
            const float a = decode_fp16(actual[index]);
            const float e = decode_fp16(expected[index]);
            if (std::isnan(e) || std::isnan(a)) {
                mismatches += std::isnan(e) != std::isnan(a) ? 1U : 0U;
                continue;
            }
            const float difference = std::fabs(a - e);
            largest = std::fmax(largest, difference);
            mismatches += difference > kTolerance ? 1U : 0U;
        }
    }
    std::printf(
        "%s: heads=%zu kv_heads=%zu head_size=%zu block=%zu sequences=%zu rows=%zu "
        "max_rows=%zu splits=%zu max_abs=%.3g mismatches=%zu\n",
        name, decode.shape.heads, decode.layout.kv_heads, decode.layout.head_size,
        decode.layout.block_size, decode.shape.sequences, decode.shape.rows,
        decode.shape.max_rows_per_sequence, decode.shape.splits, largest, mismatches);
    ECLAIR_CHECK(mismatches == 0U);
    return largest;
}

float compare(const DecodeCase& decode, const char* name) {
    return compare(decode, run_gpu(decode), reference(decode), name);
}

// Each row capacity (8, 16 and 32 rows of query heads), both head sizes,
// several sequences, splits that are empty, single and short, block sizes
// that are and are not powers of two, and an unowned row.
void test_against_reference() {
    struct Shape {
        std::size_t heads, kv_heads, head_size, block, splits;
        std::vector<std::size_t> rows, contexts;
        std::size_t unowned;
    };
    const Shape shapes[] = {
        {24, 4, 256, 16, 8, {1}, {0}, 0},        // One token: seven splits are empty.
        {24, 4, 256, 16, 16, {1}, {1000}, 0},    // Decode, 6 rows of the 8-row kernel.
        {24, 4, 256, 16, 16, {2}, {1000}, 0},    // One MTP draft, 16-row kernel.
        {24, 4, 256, 1552, 64, {3}, {5000}, 1},  // Two drafts, 32-row kernel.
        {24, 4, 256, 48, 5, {1, 3, 2}, {50, 700, 33}, 2},
        {24, 4, 256, 48, 7, {4}, {257}, 0},        // Full 24-row register tile, group of six.
        {32, 4, 256, 48, 7, {3}, {257}, 0},        // Full tile with groups of eight.
        {24, 4, 256, 16, 1, {3}, {300}, 0},        // One split holds the whole context.
        {8, 2, 128, 16, 7, {1, 2}, {200, 64}, 0},  // Head size 128, 8 rows: split keys.
        {16, 2, 128, 16, 9, {4}, {500}, 0},        // Groups of eight, 32 rows.
        {4, 4, 256, 16, 4, {5, 1}, {90, 31}, 0},   // One query head per KV head.
    };
    std::uint32_t seed = 11;
    for (const Shape& s : shapes) {
        DecodeCase decode = make_decode_case(s.heads, s.kv_heads, s.head_size, s.block, s.rows,
                                             s.contexts, s.splits, s.unowned);
        eclair::test::fill_random(decode, seed++);
        static_cast<void>(compare(decode, "random"));
    }
}

// Pin every signed byte and all four positions of a packed word on the
// register-fragment path and the larger-row WMMA fallback.
void test_packed_bytes() {
    for (const std::size_t rows : {3U, 5U}) {
        DecodeCase decode = make_decode_case(24, 4, 256, 48, {rows}, {257}, 7);
        for (std::size_t i = 0; i < decode.query.size(); ++i) {
            decode.query[i] = eclair::formats::encode_fp16(i % 2U == 0U ? 0.0625F : -0.0625F);
        }
        for (std::size_t t = 0; t < 257U + rows; ++t) {
            for (std::size_t h = 0; h < 4U; ++h) {
                decode.key_scales[decode.scale_index(0, t, h)] = 0.0078125F;
                decode.value_scales[decode.scale_index(0, t, h)] = 0.0078125F;
                const std::size_t cell = decode.cell(0, t, h);
                for (std::size_t d = 0; d < 256U; ++d) {
                    decode.keys[cell + d] =
                        static_cast<std::int8_t>(static_cast<int>((d + t + h) % 256U) - 128);
                    decode.values[cell + d] = static_cast<std::int8_t>(
                        static_cast<int>((d + 7U * t + 3U * h) % 256U) - 128);
                }
            }
        }
        static_cast<void>(compare(decode, "packed signed bytes"));
    }
}

// The packed K path requires eight-byte alignment, while the public contract
// allows four. V stays only four-byte aligned on both paths, as in vLLM.
void test_alignment_windows() {
    DecodeCase decode = make_decode_case(24, 4, 256, 48, {3}, {257}, 7);
    eclair::test::fill_random(decode, 23);
    const auto expected = reference(decode);
    for (const std::size_t key_offset : {0U, 4U}) {
        Stream stream;
        DeviceCase device(decode, stream.get());
        std::vector<std::int8_t> key_bytes(key_offset, 0);
        key_bytes.insert(key_bytes.end(), decode.keys.begin(), decode.keys.end());
        std::vector<std::int8_t> value_bytes(4U, 0);
        value_bytes.insert(value_bytes.end(), decode.values.begin(), decode.values.end());
        DeviceBuffer<std::int8_t> keys(key_bytes, stream.get());
        DeviceBuffer<std::int8_t> values(value_bytes, stream.get());
        auto buffers = device.buffers();
        buffers.keys = {keys.data() + key_offset, decode.keys.size()};
        buffers.values = {values.data() + 4U, decode.values.size()};
        checked(launch_paged_decode(decode.shape, decode.layout, buffers, stream.get()),
                "aligned decode");
        static_cast<void>(compare(decode, device.output.download(stream.get()), expected,
                                  "four-byte cache bases"));
    }
}

// Unwritten slots, the spare block and table entries past a sequence's length
// point at NaN scales and extreme bytes, and none reach the output. A NaN
// value scale reaches the rows of its KV head that attend the key. A split
// whose visible keys all have NaN key scales still makes its row NaN, which a
// maximum that skipped NaN would have hidden as an empty split.
void test_unwritten_and_nan_slots() {
    DecodeCase decode = make_decode_case(12, 2, 256, 16, {1, 3}, {60, 37}, 4);
    eclair::test::fill_random(decode, 3);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t n = 0; n < decode.shape.sequences; ++n) {
        const auto tokens = static_cast<std::size_t>(decode.lengths[n]);
        const std::size_t padded = (tokens + 15U) / 16U * 16U;
        for (std::size_t position = tokens; position < padded; ++position) {
            for (std::size_t kv_head = 0; kv_head < 2U; ++kv_head) {
                decode.key_scales[decode.scale_index(n, position, kv_head)] = nan;
                decode.value_scales[decode.scale_index(n, position, kv_head)] = nan;
                const std::size_t cell = decode.cell(n, position, kv_head);
                for (std::size_t d = 0; d < 256U; ++d) {
                    decode.keys[cell + d] = 127;
                    decode.values[cell + d] = -127;
                }
            }
        }
        // The spare entry names block 0, whose scales are NaN below.
        const std::size_t used = padded / 16U;
        decode.tables[n * decode.shape.block_table_stride + used] = 0;
    }
    for (std::size_t i = 0; i < decode.layout.kv_heads * 16U; ++i) {
        decode.key_scales[i] = nan;
        decode.value_scales[i] = nan;
    }
    static_cast<void>(compare(decode, "unwritten slots"));

    // Sequence 1 owns rows 1-3 at positions 37-39. Rows 2 and 3 attend key 38,
    // so their heads 6-11 (KV head 1) are NaN. The contract lets the NaN also
    // reach row 1 through a zero weight, but never KV head 0's heads.
    decode.value_scales[decode.scale_index(1, 38, 1)] = nan;
    const std::vector<std::uint16_t> actual = run_gpu(decode);
    const std::vector<std::uint16_t> expected = reference(decode);
    std::size_t wrong = 0;
    for (std::size_t row = 0; row < decode.shape.rows; ++row) {
        for (std::size_t head = 0; head < decode.shape.heads; ++head) {
            for (std::size_t d = 0; d < 256U; ++d) {
                const std::size_t index = row * decode.shape.output_row_stride + head * 256U + d;
                const float a = decode_fp16(actual[index]);
                const float e = decode_fp16(expected[index]);
                if (head >= 6U && row >= 2U) {
                    wrong += std::isnan(a) && std::isnan(e) ? 0U : 1U;
                } else if (!(head >= 6U && row == 1U && std::isnan(a))) {
                    wrong += !std::isnan(e) && std::fabs(a - e) <= kTolerance ? 0U : 1U;
                }
            }
        }
    }
    std::printf("attended NaN: wrong=%zu\n", wrong);
    ECLAIR_CHECK(wrong == 0U);

    DecodeCase split = make_decode_case(6, 1, 256, 16, {1}, {63}, 2);
    eclair::test::fill_random(split, 8);
    for (std::size_t position = 32; position < 64; ++position) {
        split.key_scales[split.scale_index(0, position, 0)] = nan;
    }
    const std::vector<std::uint16_t> split_actual = run_gpu(split);
    static_cast<void>(compare(split, split_actual, reference(split), "NaN split"));
    for (const std::uint16_t bits : split_actual) {
        ECLAIR_CHECK(std::isnan(decode_fp16(bits)));
    }
}

// A sequence whose metadata breaks the contract's rules leaves its rows
// unwritten. The other sequences are unaffected.
void test_skipped_metadata() {
    DecodeCase base = make_decode_case(24, 4, 256, 16, {2, 1, 3}, {40, 70, 20}, 6);
    eclair::test::fill_random(base, 21);
    const std::vector<std::uint16_t> expected = reference(base);
    struct Change {
        const char* name;
        std::size_t sequence;
        void (*apply)(DecodeCase&);
    };
    const Change changes[] = {
        {"rows over the limit", 2, [](DecodeCase& d) { d.shape.max_rows_per_sequence = 2; }},
        {"fewer tokens than rows", 0, [](DecodeCase& d) { d.lengths[0] = 1; }},
        {"tokens past the table", 1,
         [](DecodeCase& d) {
             d.lengths[1] = static_cast<std::int32_t>(d.shape.max_blocks_per_sequence * 16U + 1U);
         }},
        {"start past the end", 2, [](DecodeCase& d) { d.starts[3] = 7; }},
    };
    for (const Change& change : changes) {
        DecodeCase decode = base;
        change.apply(decode);
        const std::vector<std::uint16_t> actual = run_gpu(decode);
        std::size_t wrong = 0;
        for (std::size_t n = 0; n < base.shape.sequences; ++n) {
            for (auto row = static_cast<std::size_t>(base.starts[n]);
                 row < static_cast<std::size_t>(base.starts[n + 1U]); ++row) {
                for (std::size_t e = 0; e < base.shape.output_row_stride; ++e) {
                    const std::size_t index = row * base.shape.output_row_stride + e;
                    if (n == change.sequence) {
                        wrong += actual[index] == kSentinel ? 0U : 1U;
                    } else {
                        const float difference =
                            std::fabs(decode_fp16(actual[index]) - decode_fp16(expected[index]));
                        wrong += difference <= kTolerance ? 0U : 1U;
                    }
                }
            }
        }
        std::printf("skipped metadata (%s): wrong=%zu\n", change.name, wrong);
        ECLAIR_CHECK(wrong == 0U);
    }
}

// A graph captured once replays with the metadata it reads at replay time,
// as an engine's decode graph does when each step adds a token.
void test_graph_replay() {
    DecodeCase decode = make_decode_case(24, 4, 256, 16, {3}, {500}, 32);
    eclair::test::fill_random(decode, 31);
    Stream stream;
    DeviceCase device(decode, stream.get());
    cudaGraph_t graph = nullptr;
    checked(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal), "begin capture");
    const cudaError_t launched =
        launch_paged_decode(decode.shape, decode.layout, device.buffers(), stream.get());
    checked(cudaStreamEndCapture(stream.get(), &graph), "end capture");
    checked(launched, "captured decode");
    cudaGraphExec_t executable = nullptr;
    checked(cudaGraphInstantiate(&executable, graph, 0ULL), "instantiate");
    for (const std::int32_t length : {503, 495, 512}) {
        decode.lengths[0] = length;
        device.lengths.upload(decode.lengths, stream.get());
        checked(cudaGraphLaunch(executable, stream.get()), "graph launch");
        const std::vector<std::uint16_t> actual = device.output.download(stream.get());
        static_cast<void>(compare(decode, actual, reference(decode), "graph replay"));
    }
    checked(cudaGraphExecDestroy(executable), "destroy executable");
    checked(cudaGraphDestroy(graph), "destroy graph");
}

// Unsupported arguments enqueue nothing, so the sentinel output survives.
void test_rejections() {
    Stream stream;
    DecodeCase decode = make_decode_case(24, 4, 256, 16, {3}, {10}, 4);
    eclair::test::fill_random(decode, 5);
    DeviceCase device(decode, stream.get());

    auto too_many_rows = decode.shape;
    too_many_rows.max_rows_per_sequence = 6;  // 36 query rows per KV head.
    ECLAIR_CHECK(launch_paged_decode(too_many_rows, decode.layout, device.buffers(),
                                     stream.get()) == cudaErrorInvalidValue);
    auto too_many_splits = decode.shape;
    too_many_splits.splits = eclair::cuda::kPagedDecodeMaxSplits + 1U;
    DeviceBuffer<float> large(eclair::ops::paged_decode_workspace_elements(
        decode.shape.rows, decode.shape.heads, 256, too_many_splits.splits));
    PagedDecodeBuffers large_workspace = device.buffers();
    large_workspace.workspace = {large.data(), large.size()};
    ECLAIR_CHECK(launch_paged_decode(too_many_splits, decode.layout, large_workspace,
                                     stream.get()) == cudaErrorInvalidValue);
    PagedDecodeBuffers misaligned = device.buffers();
    misaligned.values.data += 2;  // Value cells need 4-byte alignment.
    misaligned.values.size -= 2U;
    ECLAIR_CHECK(launch_paged_decode(decode.shape, decode.layout, misaligned, stream.get()) ==
                 cudaErrorInvalidValue);
    PagedDecodeBuffers short_workspace = device.buffers();
    short_workspace.workspace.size -= 1U;
    ECLAIR_CHECK(launch_paged_decode(decode.shape, decode.layout, short_workspace, stream.get()) ==
                 cudaErrorInvalidValue);
    DecodeCase small_head = make_decode_case(4, 1, 64, 16, {1}, {10}, 2);
    DeviceCase small(small_head, stream.get());
    ECLAIR_CHECK(launch_paged_decode(small_head.shape, small_head.layout, small.buffers(),
                                     stream.get()) == cudaErrorInvalidValue);

    ECLAIR_CHECK(eclair::cuda::paged_decode_heads_supported(24, 4, 256, 5));
    ECLAIR_CHECK(!eclair::cuda::paged_decode_heads_supported(24, 4, 256, 6));
    ECLAIR_CHECK(eclair::cuda::paged_decode_heads_supported(16, 2, 128, 4));
    ECLAIR_CHECK(!eclair::cuda::paged_decode_heads_supported(24, 4, 64, 1));
    ECLAIR_CHECK(!eclair::cuda::paged_decode_heads_supported(24, 4, 256, 0));
    ECLAIR_CHECK(!eclair::cuda::paged_decode_heads_supported(3, 2, 128, 1));

    ECLAIR_CHECK(cudaGetLastError() == cudaSuccess);
    for (const std::uint16_t bits : device.output.download(stream.get())) {
        ECLAIR_CHECK(bits == kSentinel);
    }
    ECLAIR_CHECK(eclair::cuda::check_paged_decode_image(0) == cudaSuccess);
}

[[nodiscard]] __device__ std::uint32_t mix(std::uint64_t value) {
    value ^= value >> 33U;
    value *= 0xFF51AFD7ED558CCDULL;
    value ^= value >> 33U;
    return static_cast<std::uint32_t>(value);
}

__global__ void fill_cache(std::int8_t* bytes, float* scales, std::size_t byte_count,
                           std::size_t scale_count, std::uint64_t seed) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         i < byte_count; i += stride) {
        bytes[i] = static_cast<std::int8_t>(static_cast<int>(mix(seed + i) % 255U) - 127);
    }
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         i < scale_count; i += stride) {
        scales[i] = 0.005F + 0.025F * static_cast<float>(mix(seed ^ (i * 7U)) % 1000U) / 1000.0F;
    }
}

__global__ void fill_query(std::uint16_t* query, std::size_t count, std::uint64_t seed) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += stride) {
        const float value = static_cast<float>(mix(seed + i) % 4001U) / 1000.0F - 2.0F;
        query[i] = __half_as_ushort(__float2half_rn(value));
    }
}

// Median CUDA-event time of one sequence's decode at Qwen3.8 27B's shapes,
// for one row (decode) and two or three rows (MTP verification). Bandwidth
// counts the K and V bytes and scales the kernel reads once per KV head.
void benchmark() {
    constexpr std::size_t kHeads = 24;
    constexpr std::size_t kKvHeads = 4;
    constexpr std::size_t kHeadSize = 256;
    constexpr std::size_t kBlock = 1552;
    constexpr std::size_t kMaxTokens = 211003;
    const std::size_t blocks = (kMaxTokens + kBlock - 1U) / kBlock + 1U;
    DecodeCase layout_case = make_decode_case(kHeads, kKvHeads, kHeadSize, kBlock, {1}, {0}, 1);
    eclair::ops::PagedInt8CacheLayout layout = layout_case.layout;
    layout.blocks = blocks;
    Stream stream;
    DeviceBuffer<std::int8_t> keys(blocks * layout.block_stride);
    DeviceBuffer<std::int8_t> values(keys.size());
    DeviceBuffer<float> key_scales(blocks * kKvHeads * kBlock);
    DeviceBuffer<float> value_scales(key_scales.size());
    std::vector<std::int32_t> table;
    for (std::size_t entry = 0; entry + 1U < blocks; ++entry) {
        table.push_back(static_cast<std::int32_t>(entry + 1U));
    }
    DeviceBuffer<std::int32_t> tables(table, stream.get());
    constexpr std::size_t kMaxRows = 3;
    DeviceBuffer<std::uint16_t> query(kMaxRows * kHeads * kHeadSize);
    DeviceBuffer<std::uint16_t> output(query.size());
    fill_cache<<<1024, 256, 0, stream.get()>>>(keys.data(), key_scales.data(), keys.size(),
                                               key_scales.size(), 11U);
    fill_cache<<<1024, 256, 0, stream.get()>>>(values.data(), value_scales.data(), values.size(),
                                               value_scales.size(), 23U);
    fill_query<<<1024, 256, 0, stream.get()>>>(query.data(), query.size(), 37U);
    checked(cudaGetLastError(), "fill");
    for (const std::size_t splits : {64U, 128U}) {
        for (const std::size_t context : {2000U, 33000U, 100000U, 211000U}) {
            for (std::size_t rows = 1; rows <= kMaxRows; ++rows) {
                const eclair::ops::PagedDecodeShape shape{kHeads,
                                                          1U,
                                                          rows,
                                                          rows,
                                                          table.size(),
                                                          table.size(),
                                                          splits,
                                                          kHeads * kHeadSize,
                                                          kHeads * kHeadSize,
                                                          0.0625F};
                const std::vector<std::int32_t> starts{0, static_cast<std::int32_t>(rows)};
                const std::vector<std::int32_t> lengths{static_cast<std::int32_t>(context + rows)};
                DeviceBuffer<std::int32_t> device_starts(starts, stream.get());
                DeviceBuffer<std::int32_t> device_lengths(lengths, stream.get());
                DeviceBuffer<float> workspace(
                    eclair::ops::paged_decode_workspace_elements(rows, kHeads, kHeadSize, splits));
                const PagedDecodeBuffers buffers{{query.data(), rows * kHeads * kHeadSize},
                                                 {output.data(), rows * kHeads * kHeadSize},
                                                 {keys.data(), keys.size()},
                                                 {values.data(), values.size()},
                                                 {key_scales.data(), key_scales.size()},
                                                 {value_scales.data(), value_scales.size()},
                                                 {tables.data(), tables.size()},
                                                 {device_starts.data(), device_starts.size()},
                                                 {device_lengths.data(), device_lengths.size()},
                                                 {workspace.data(), workspace.size()}};
                const auto samples = eclair::test::timings(stream.get(), [&] {
                    checked(launch_paged_decode(shape, layout, buffers, stream.get()),
                            "benchmark launch");
                });
                bool finite = true;
                for (const std::uint16_t bits : output.download(stream.get())) {
                    finite = finite && std::isfinite(decode_fp16(bits));
                }
                ECLAIR_CHECK(finite);
                const double bytes =
                    static_cast<double>(context + rows) * kKvHeads * (2.0 * kHeadSize + 8.0);
                std::printf(
                    "benchmark: splits=%zu context=%zu rows=%zu median_ms=%.3f max_ms=%.3f "
                    "kv_gb_per_s=%.0f finite=%d\n",
                    splits, context, rows, samples[4], samples[8],
                    bytes / (static_cast<double>(samples[4]) * 1e-3) / 1e9, finite ? 1 : 0);
            }
        }
    }
}

}  // namespace

int main() {
    return eclair::test::contain([] {
        eclair::test::print_device_header();
        test_against_reference();
        test_packed_bytes();
        test_alignment_windows();
        test_unwritten_and_nan_slots();
        test_skipped_metadata();
        test_graph_replay();
        test_rejections();
        benchmark();
    });
}
