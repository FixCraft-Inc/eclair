/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

// Paged int8 prefill attention on the GPU against the FP64 host reference,
// rejection of unsupported arguments, and prefill timings at the shapes of
// Qwen3.8 27B's attention layers (24 query heads, 4 KV heads, head size 256,
// 1552-token cache blocks). Every comparison uses the reference's own
// definition, so the tolerance only covers FP32 accumulation order, binary16
// probabilities and the binary16 output rounding.

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
#include <vector>

namespace {

using eclair::cuda::launch_paged_prefill;
using eclair::formats::decode_fp16;
using eclair::ops::PagedPrefillBuffers;
using eclair::ops::Status;
using eclair::test::checked;
using eclair::test::DeviceBuffer;
using eclair::test::make_paged_case;
using eclair::test::PagedCase;
using eclair::test::Stream;

// Output magnitudes stay below 4. Binary16 probabilities and output rounding
// contribute about 2e-3 in the worst observed element.
constexpr float kTolerance = 8e-3F;

struct DeviceCase {
    DeviceBuffer<std::uint16_t> query;
    DeviceBuffer<std::uint16_t> output;
    DeviceBuffer<std::int8_t> keys;
    DeviceBuffer<std::int8_t> values;
    DeviceBuffer<float> key_scales;
    DeviceBuffer<float> value_scales;
    DeviceBuffer<std::int32_t> table;

    DeviceCase(const PagedCase& paged, cudaStream_t stream)
        : query(paged.query, stream),
          output(paged.output_elements()),
          keys(paged.keys, stream),
          values(paged.values, stream),
          key_scales(paged.key_scales, stream),
          value_scales(paged.value_scales, stream),
          table(paged.table, stream) {}

    [[nodiscard]] PagedPrefillBuffers buffers() const {
        return {{query.data(), query.size()},
                {output.data(), output.size()},
                {keys.data(), keys.size()},
                {values.data(), values.size()},
                {key_scales.data(), key_scales.size()},
                {value_scales.data(), value_scales.size()},
                {table.data(), table.size()}};
    }
};

[[nodiscard]] std::vector<std::uint16_t> reference(const PagedCase& paged) {
    std::vector<std::uint16_t> expected(paged.output_elements(), 0U);
    const PagedPrefillBuffers host{{paged.query.data(), paged.query.size()},
                                   {expected.data(), expected.size()},
                                   {paged.keys.data(), paged.keys.size()},
                                   {paged.values.data(), paged.values.size()},
                                   {paged.key_scales.data(), paged.key_scales.size()},
                                   {paged.value_scales.data(), paged.value_scales.size()},
                                   {paged.table.data(), paged.table.size()}};
    if (eclair::host::paged_prefill_reference(paged.shape, paged.layout, host) != Status::Success) {
        throw std::runtime_error("reference rejected a test case");
    }
    return expected;
}

[[nodiscard]] std::vector<std::uint16_t> run_gpu(const PagedCase& paged) {
    Stream stream;
    DeviceCase device(paged, stream.get());
    checked(cudaMemsetAsync(device.output.data(), 0xFF, device.output.size() * 2U, stream.get()),
            "output sentinel");
    checked(launch_paged_prefill(paged.shape, paged.layout, device.buffers(), stream.get()),
            "paged prefill");
    return device.output.download(stream.get());
}

// Compares every row's head vectors with the reference. NaN must match NaN,
// and infinities must match signs. Returns the largest finite difference.
float compare(const PagedCase& paged, const std::vector<std::uint16_t>& actual, const char* name) {
    const std::vector<std::uint16_t> expected = reference(paged);
    float largest = 0.0F;
    std::size_t mismatches = 0;
    const std::size_t width = paged.shape.heads * paged.layout.head_size;
    for (std::size_t row = 0; row < paged.shape.rows; ++row) {
        for (std::size_t element = 0; element < width; ++element) {
            const std::size_t index = row * paged.shape.output_row_stride + element;
            const float a = decode_fp16(actual[index]);
            const float e = decode_fp16(expected[index]);
            if (std::isnan(e) || std::isnan(a)) {
                mismatches += std::isnan(e) != std::isnan(a) ? 1U : 0U;
                continue;
            }
            if (std::isinf(e) || std::isinf(a)) {
                mismatches += a != e ? 1U : 0U;
                continue;
            }
            const float difference = std::fabs(a - e);
            largest = std::fmax(largest, difference);
            mismatches += difference > kTolerance ? 1U : 0U;
        }
    }
    std::printf(
        "%s: heads=%zu kv_heads=%zu head_size=%zu block=%zu context=%zu rows=%zu "
        "max_abs=%.3g mismatches=%zu\n",
        name, paged.shape.heads, paged.layout.kv_heads, paged.layout.head_size,
        paged.layout.block_size, paged.shape.context, paged.shape.rows, largest, mismatches);
    ECLAIR_CHECK(mismatches == 0U);
    return largest;
}

float compare(const PagedCase& paged, const char* name) {
    return compare(paged, run_gpu(paged), name);
}

void test_against_reference() {
    struct Shape {
        std::size_t heads, kv_heads, head_size, block, context, rows;
    };
    const Shape shapes[] = {
        {24, 4, 256, 16, 0, 16},     // One row tile, one key tile, Qwen3.8's grouping.
        {24, 4, 256, 16, 37, 33},    // Shifted last row tile, partial key tiles.
        {12, 2, 256, 48, 1000, 48},  // Block size that is not a power of two.
        {8, 2, 128, 16, 300, 64},    // Head size 128, groups of four.
        {16, 2, 128, 16, 100, 16},   // Groups of eight, the maximum at head size 128.
        {4, 4, 256, 16, 70, 40},     // One query head per KV head.
        {2, 1, 256, 16, 67, 19},     // Every compile-time loader group.
        {3, 1, 256, 16, 67, 19},    {4, 1, 256, 16, 67, 19}, {5, 1, 256, 16, 67, 19},
        {1, 1, 128, 16, 67, 19},    {2, 1, 128, 16, 67, 19}, {3, 1, 128, 16, 67, 19},
        {5, 1, 128, 16, 67, 19},    {6, 1, 128, 16, 67, 19}, {7, 1, 128, 16, 67, 19},
    };
    std::uint32_t seed = 7;
    for (const Shape& s : shapes) {
        PagedCase paged =
            make_paged_case(s.heads, s.kv_heads, s.head_size, s.block, s.context, s.rows);
        eclair::test::fill_random(paged, seed++);
        static_cast<void>(compare(paged, "random"));
    }
    // Query rows padded beyond their head vectors, as a strided view would be.
    PagedCase padded = make_paged_case(12, 2, 256, 16, 20, 20, 16);
    eclair::test::fill_random(padded, 99);
    static_cast<void>(compare(padded, "padded rows"));
}

// Row zero sees one key, so its output directly checks all 256 int8 values
// and the contract's FP32 scale product before the binary16 rounding. Tiny
// scales and a large scale times zero distinguish it from a binary16 scale.
void test_dequantization_rounding() {
    PagedCase paged = make_paged_case(6, 1, 256, 16, 0, 16);
    for (const float scale :
         {0.0F, -0.0F, 1.0F / 7.0F, 0.0001234567F, 0x1p-26F, 0x1p-126F, 0x1p-149F, 70000.0F}) {
        for (std::size_t position = 0; position < 16U; ++position) {
            const std::size_t cell = paged.cell(position, 0);
            for (std::size_t d = 0; d < 256U; ++d) {
                paged.values[cell + d] =
                    scale > 65504.0F ? 0 : static_cast<std::int8_t>(static_cast<int>(d) - 128);
            }
            paged.key_scales[paged.scale_index(position, 0)] = 1.0F;
            paged.value_scales[paged.scale_index(position, 0)] = scale;
        }
        const auto actual = run_gpu(paged);
        const auto expected = reference(paged);
        for (std::size_t element = 0; element < 6U * 256U; ++element) {
            // FP32 accumulation need not preserve the sign of a zero sum.
            ECLAIR_CHECK(decode_fp16(actual[element]) == decode_fp16(expected[element]));
        }
    }
    std::puts("dequantization: all int8 values, FP32 scales and binary16 roundings passed");
}

// Zero keys isolate query representations: finite queries give uniform
// attention, while infinity times zero and NaN must poison their own row.
void test_query_representations_and_nonfinite_scales() {
    PagedCase paged = make_paged_case(6, 1, 256, 16, 0, 16);
    for (std::size_t position = 0; position < 16U; ++position) {
        const std::size_t cell = paged.cell(position, 0);
        for (std::size_t d = 0; d < 256U; ++d) {
            paged.values[cell + d] = static_cast<std::int8_t>(static_cast<int>(d) - 128);
        }
        paged.key_scales[paged.scale_index(position, 0)] = 1.0F;
        paged.value_scales[paged.scale_index(position, 0)] = 1.0F / 7.0F;
    }
    const std::uint16_t patterns[] = {0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x03FFU, 0x83FFU, 0x0400U,
                                      0x8400U, 0x7BFFU, 0xFBFFU, 0x7C00U, 0xFC00U, 0x7E01U};
    for (std::size_t row = 0; row < paged.shape.rows; ++row) {
        for (std::size_t d = 0; d < paged.shape.query_row_stride; ++d) {
            paged.query[row * paged.shape.query_row_stride + d] = patterns[row % 13U];
        }
    }
    static_cast<void>(compare(paged, "query representations"));
    for (std::uint16_t& value : paged.query) {
        value = 0U;
    }
    const float infinity = std::numeric_limits<float>::infinity();
    for (const float scale : {infinity, -infinity, std::numeric_limits<float>::quiet_NaN()}) {
        paged.key_scales[paged.scale_index(0, 0)] = scale;
        static_cast<void>(compare(paged, "nonfinite attended key scale"));
        paged.key_scales[paged.scale_index(0, 0)] = 1.0F;
        paged.value_scales[paged.scale_index(0, 0)] = scale;
        static_cast<void>(compare(paged, "nonfinite attended value scale"));
        paged.value_scales[paged.scale_index(0, 0)] = 1.0F / 7.0F;
    }
    // A finite FP32 key product may overflow binary16 before QK.
    paged.key_scales[paged.scale_index(0, 0)] = 70000.0F;
    paged.keys[paged.cell(0, 0)] = 1;
    static_cast<void>(compare(paged, "binary16 key overflow"));
}

// Unwritten slots and a spare block holding NaN scales and extreme bytes
// never reach the output. A NaN value scale of a key inside the chunk reaches
// every row that attends it. The contract lets it also reach earlier rows of
// the chunk through a zero weight, but never the other KV head's rows.
void test_unwritten_and_nan_slots() {
    PagedCase paged = make_paged_case(12, 2, 256, 16, 21, 20);
    eclair::test::fill_random(paged, 3);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::size_t tokens = 41;
    const std::size_t padded_tokens = 48;
    for (std::size_t position = tokens; position < padded_tokens; ++position) {
        for (std::size_t kv_head = 0; kv_head < 2; ++kv_head) {
            paged.key_scales[paged.scale_index(position, kv_head)] = nan;
            paged.value_scales[paged.scale_index(position, kv_head)] = nan;
            const std::size_t cell = paged.cell(position, kv_head);
            for (std::size_t d = 0; d < 256; ++d) {
                paged.keys[cell + d] = 127;
                paged.values[cell + d] = -127;
            }
        }
    }
    for (float& scale : paged.value_scales) {
        if (scale == 0.0F) {
            scale = nan;  // The spare block 0 and padding cells.
        }
    }
    static_cast<void>(compare(paged, "unwritten slots"));

    paged.value_scales[paged.scale_index(30, 1)] = nan;
    const std::vector<std::uint16_t> actual = run_gpu(paged);
    const std::vector<std::uint16_t> expected = reference(paged);
    // Row 9 (position 30) is the first to see key 30. KV head 1 serves query heads 6-11.
    std::size_t wrong = 0;
    for (std::size_t row = 0; row < paged.shape.rows; ++row) {
        for (std::size_t head = 0; head < paged.shape.heads; ++head) {
            for (std::size_t d = 0; d < 256; ++d) {
                const std::size_t index = row * paged.shape.output_row_stride + head * 256 + d;
                const float a = decode_fp16(actual[index]);
                const float e = decode_fp16(expected[index]);
                const bool attends = head >= 6 && row >= 9;
                const bool may_be_nan = head >= 6 && row < 9;
                if (attends) {
                    wrong += std::isnan(a) && std::isnan(e) ? 0U : 1U;
                } else if (!(may_be_nan && std::isnan(a))) {
                    wrong += !std::isnan(e) && std::fabs(a - e) <= kTolerance ? 0U : 1U;
                }
            }
        }
    }
    std::printf("attended NaN: wrong=%zu\n", wrong);
    ECLAIR_CHECK(wrong == 0U);
}

// Both engine block sizes, shifted row tiles and four-byte K/V bases must
// replay with new query/cache contents, rather than retain captured fragments.
void test_graph_replay_and_alignment() {
    struct Graph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;

        ~Graph() {
            if (executable != nullptr) {
                ECLAIR_CHECK(cudaGraphExecDestroy(executable) == cudaSuccess);
            }
            if (graph != nullptr) {
                ECLAIR_CHECK(cudaGraphDestroy(graph) == cudaSuccess);
            }
        }
    };
    for (const std::size_t block : {1552U, 1568U}) {
        PagedCase paged = make_paged_case(6, 1, 256, block, block + 37U, 19, 16);
        eclair::test::fill_random(paged, 30U);
        Stream stream;
        DeviceCase device(paged, stream.get());
        DeviceBuffer<std::int8_t> keys(paged.keys.size() + 4U);
        DeviceBuffer<std::int8_t> values(paged.values.size() + 4U);
        auto buffers = device.buffers();
        buffers.keys = {keys.data() + 4, paged.keys.size()};
        buffers.values = {values.data() + 4, paged.values.size()};
        Graph captured;
        checked(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal), "begin capture");
        const cudaError_t launched =
            launch_paged_prefill(paged.shape, paged.layout, buffers, stream.get());
        checked(cudaStreamEndCapture(stream.get(), &captured.graph), "end capture");
        checked(launched, "captured prefill");
        checked(cudaGraphInstantiate(&captured.executable, captured.graph, 0ULL), "instantiate");
        for (const std::uint32_t seed : {31U, 32U}) {
            eclair::test::fill_random(paged, seed);
            device.query.upload(paged.query, stream.get());
            device.key_scales.upload(paged.key_scales, stream.get());
            device.value_scales.upload(paged.value_scales, stream.get());
            checked(cudaMemcpyAsync(keys.data() + 4, paged.keys.data(), paged.keys.size(),
                                    cudaMemcpyHostToDevice, stream.get()),
                    "upload shifted keys");
            checked(cudaMemcpyAsync(values.data() + 4, paged.values.data(), paged.values.size(),
                                    cudaMemcpyHostToDevice, stream.get()),
                    "upload shifted values");
            checked(cudaMemsetAsync(device.output.data(), 0x7E, device.output.size() * 2U,
                                    stream.get()),
                    "output sentinel");
            checked(cudaGraphLaunch(captured.executable, stream.get()), "graph launch");
            const auto actual = device.output.download(stream.get());
            static_cast<void>(compare(paged, actual, "graph replay, four-byte K/V"));
            for (std::size_t row = 0; row < paged.shape.rows; ++row) {
                for (std::size_t d = 6U * 256U; d < paged.shape.output_row_stride; ++d) {
                    ECLAIR_CHECK(actual[row * paged.shape.output_row_stride + d] == 0x7E7EU);
                }
            }
        }
    }
}

// Unsupported arguments enqueue nothing, so the sentinel output survives.
void test_rejections() {
    Stream stream;
    PagedCase paged = make_paged_case(24, 4, 256, 16, 10, 16);
    eclair::test::fill_random(paged, 5);
    DeviceCase device(paged, stream.get());
    checked(cudaMemsetAsync(device.output.data(), 0x7E, device.output.size() * 2U, stream.get()),
            "output sentinel");

    auto short_chunk = paged.shape;
    short_chunk.rows = eclair::cuda::kPagedPrefillMinRows - 1U;
    ECLAIR_CHECK(launch_paged_prefill(short_chunk, paged.layout, device.buffers(), stream.get()) ==
                 cudaErrorInvalidValue);

    PagedCase small_head = make_paged_case(4, 1, 64, 16, 0, 16);
    DeviceCase small(small_head, stream.get());
    ECLAIR_CHECK(launch_paged_prefill(small_head.shape, small_head.layout, small.buffers(),
                                      stream.get()) == cudaErrorInvalidValue);

    PagedPrefillBuffers misaligned = device.buffers();
    misaligned.query.data += 8;  // 16 bytes: binary16-aligned but not a 32-byte WMMA tile.
    misaligned.query.size -= 8U;
    ECLAIR_CHECK(launch_paged_prefill(paged.shape, paged.layout, misaligned, stream.get()) ==
                 cudaErrorInvalidValue);

    auto wide_group = paged.shape;
    wide_group.heads = 4U * (eclair::cuda::kPagedPrefillMaxGroup + 1U);
    ECLAIR_CHECK(launch_paged_prefill(wide_group, paged.layout, device.buffers(), stream.get()) ==
                 cudaErrorInvalidValue);
    // Eight query heads of size 256 do not fit their query rows in shared memory.
    PagedCase full_group = make_paged_case(16, 2, 256, 16, 0, 16);
    DeviceCase full(full_group, stream.get());
    ECLAIR_CHECK(launch_paged_prefill(full_group.shape, full_group.layout, full.buffers(),
                                      stream.get()) == cudaErrorInvalidValue);

    ECLAIR_CHECK(eclair::cuda::paged_prefill_heads_supported(24, 4, 256));
    ECLAIR_CHECK(eclair::cuda::paged_prefill_heads_supported(16, 2, 128));
    ECLAIR_CHECK(!eclair::cuda::paged_prefill_heads_supported(16, 2, 256));
    ECLAIR_CHECK(!eclair::cuda::paged_prefill_heads_supported(24, 4, 64));
    ECLAIR_CHECK(!eclair::cuda::paged_prefill_heads_supported(3, 2, 128));
    ECLAIR_CHECK(!eclair::cuda::paged_prefill_heads_supported(4, 0, 128));

    ECLAIR_CHECK(cudaGetLastError() == cudaSuccess);
    for (const std::uint16_t bits : device.output.download(stream.get())) {
        ECLAIR_CHECK(bits == 0x7E7EU);
    }
    ECLAIR_CHECK(eclair::cuda::check_paged_prefill_image(0) == cudaSuccess);
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

// Median CUDA-event time of the paged prefill at Qwen3.8 27B's attention
// shapes. FLOPs count the causal half of the chunk's own keys.
void benchmark() {
    constexpr std::size_t kKvHeads = 4;
    constexpr std::size_t kHeadSize = 256;
    constexpr std::size_t kBlock = 1552;
    struct Point {
        std::size_t heads, context, rows;
    };
    // Qwen3.8 27B's 24 query heads, then groups of four at one context.
    for (const Point point :
         {Point{24, 0, 2048}, Point{24, 8192, 4096}, Point{24, 32768, 4096},
          Point{24, 196608, 4096}, Point{24, 196608, 8192}, Point{16, 32768, 4096}}) {
        const std::size_t heads = point.heads;
        Stream stream;
        PagedCase paged = make_paged_case(heads, kKvHeads, kHeadSize, kBlock, 0, 1);
        const std::size_t tokens = point.context + point.rows;
        const std::size_t blocks = (tokens + kBlock - 1U) / kBlock + 1U;
        paged.shape.rows = point.rows;
        paged.shape.context = point.context;
        paged.shape.scale = 1.0F / 16.0F;
        paged.layout.blocks = blocks;
        paged.table.clear();
        for (std::size_t entry = 0; entry + 1U < blocks; ++entry) {
            paged.table.push_back(static_cast<std::int32_t>(entry + 1U));
        }
        DeviceBuffer<std::uint16_t> query(point.rows * paged.shape.query_row_stride);
        DeviceBuffer<std::uint16_t> output(point.rows * paged.shape.output_row_stride);
        DeviceBuffer<std::int8_t> keys(blocks * paged.layout.block_stride);
        DeviceBuffer<std::int8_t> values(keys.size());
        DeviceBuffer<float> key_scales(blocks * kKvHeads * kBlock);
        DeviceBuffer<float> value_scales(key_scales.size());
        DeviceBuffer<std::int32_t> table(paged.table, stream.get());
        fill_cache<<<1024, 256, 0, stream.get()>>>(keys.data(), key_scales.data(), keys.size(),
                                                   key_scales.size(), 11U);
        fill_cache<<<1024, 256, 0, stream.get()>>>(values.data(), value_scales.data(),
                                                   values.size(), value_scales.size(), 23U);
        fill_query<<<1024, 256, 0, stream.get()>>>(query.data(), query.size(), 37U);
        checked(cudaGetLastError(), "fill");
        const PagedPrefillBuffers buffers{{query.data(), query.size()},
                                          {output.data(), output.size()},
                                          {keys.data(), keys.size()},
                                          {values.data(), values.size()},
                                          {key_scales.data(), key_scales.size()},
                                          {value_scales.data(), value_scales.size()},
                                          {table.data(), table.size()}};
        const auto samples = eclair::test::timings(stream.get(), [&] {
            checked(launch_paged_prefill(paged.shape, paged.layout, buffers, stream.get()),
                    "benchmark launch");
        });
        bool finite = true;
        for (const std::uint16_t bits : output.download(stream.get())) {
            finite = finite && std::isfinite(decode_fp16(bits));
        }
        ECLAIR_CHECK(finite);
        const double flops =
            4.0 * kHeadSize * heads * static_cast<double>(point.rows) *
            (static_cast<double>(point.context) + (static_cast<double>(point.rows) + 1.0) / 2.0);
        std::printf(
            "benchmark: heads=%zu context=%zu rows=%zu median_ms=%.3f max_ms=%.3f "
            "tflops=%.1f finite=%d\n",
            heads, point.context, point.rows, samples[4], samples[8],
            flops / (static_cast<double>(samples[4]) * 1e-3) / 1e12, finite ? 1 : 0);
    }
}

}  // namespace

int main() {
    return eclair::test::contain([] {
        eclair::test::print_device_header();
        test_against_reference();
        test_dequantization_rounding();
        test_query_representations_and_nonfinite_scales();
        test_unwritten_and_nan_slots();
        test_graph_replay_and_alignment();
        test_rejections();
        benchmark();
    });
}
