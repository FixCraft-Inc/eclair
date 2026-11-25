/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "cuda_support.hpp"

#include "eclair/cuda/bf16_convert.hpp"
#include "eclair/formats/bf16.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using eclair::test::checked;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;

[[nodiscard]] float from_bits(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Offsets 0 and 4 take the vector path (with tails from the trimmed counts),
// 1 and 3 the scalar path. Sentinels around the view catch stray stores.
template <typename In, typename Out, typename Launch>
void check_conversion(const std::vector<In>& input, const std::vector<Out>& expected, Out sentinel,
                      Launch launch) {
    const struct {
        std::size_t offset;
        std::size_t trim;
    } placements[] = {{0, 0}, {0, 1}, {0, 3}, {4, 2}, {1, 0}, {3, 1}};
    for (const auto placement : placements) {
        const std::size_t count = input.size() - placement.trim;
        Stream stream;
        std::vector<In> shifted(placement.offset, In{});
        shifted.insert(shifted.end(), input.begin(),
                       input.begin() + static_cast<std::ptrdiff_t>(count));
        const DeviceBuffer<In> device_input(shifted, stream.get());
        const DeviceBuffer<Out> device_output(
            std::vector<Out>(placement.offset + count + 1U, sentinel), stream.get());
        checked(launch(device_input.data() + placement.offset,
                       device_output.data() + placement.offset, count, stream.get()),
                "conversion launch");
        const std::vector<Out> output = device_output.download(stream.get());
        std::size_t mismatches = 0;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const bool inside = index >= placement.offset && index < placement.offset + count;
            const Out want = inside ? expected[index - placement.offset] : sentinel;
            if (std::memcmp(&output[index], &want, sizeof(Out)) != 0 && mismatches++ < 4U) {
                std::fprintf(stderr, "offset=%zu index=%zu differs\n", placement.offset, index);
            }
        }
        ECLAIR_CHECK(mismatches == 0U);
    }
}

// Every bfloat16 pattern widens exactly. NaN patterns keep their bits too,
// since widening only appends zeros.
void check_widening() {
    std::vector<std::uint16_t> input(65536U);
    std::vector<float> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint16_t>(index);
        expected[index] = from_bits(static_cast<std::uint32_t>(index) << 16U);
    }
    check_conversion(
        input, expected, from_bits(0x5A5A5A5AU),
        [](const std::uint16_t* in, float* out, std::size_t count, cudaStream_t stream) {
            return eclair::cuda::launch_bf16_to_fp32(in, out, count, stream);
        });
}

// Every exponent and kept fraction with the dropped halves at and next to the
// rounding decisions, plus random patterns, against the host encoder.
void check_narrowing() {
    std::vector<float> input;
    for (std::uint32_t high = 0; high < 65536U; ++high) {
        for (const std::uint32_t low : {0x0000U, 0x7FFFU, 0x8000U, 0x8001U, 0xFFFFU}) {
            input.push_back(from_bits((high << 16U) | low));
        }
    }
    std::mt19937 random(16U);
    for (int index = 0; index < (1 << 16); ++index) {
        input.push_back(from_bits(static_cast<std::uint32_t>(random())));
    }
    std::vector<std::uint16_t> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] = eclair::formats::encode_bf16(input[index]);
    }
    check_conversion(
        input, expected, std::uint16_t{0x5A5AU},
        [](const float* in, std::uint16_t* out, std::size_t count, cudaStream_t stream) {
            return eclair::cuda::launch_fp32_to_bf16(in, out, count, stream);
        });
}

void check_rejections() {
    ECLAIR_CHECK(eclair::cuda::launch_bf16_to_fp32(nullptr, nullptr, 0U, nullptr) == cudaSuccess);
    ECLAIR_CHECK(eclair::cuda::launch_fp32_to_bf16(nullptr, nullptr, 1U, nullptr) ==
                 cudaErrorInvalidValue);
    Stream stream;
    const DeviceBuffer<float> buffer(std::vector<float>(16U, 1.0F), stream.get());
    // Narrowing in place would let one thread's store overwrite another's input.
    ECLAIR_CHECK(eclair::cuda::launch_fp32_to_bf16(buffer.data(),
                                                   reinterpret_cast<std::uint16_t*>(buffer.data()),
                                                   16U, stream.get()) == cudaErrorInvalidValue);
    for (const float value : buffer.download(stream.get())) {
        ECLAIR_CHECK(value == 1.0F);
    }
}

void run() {
    eclair::test::print_device_header();
    std::puts(
        "BF16 to and from FP32 without BF16 instructions; bit-exact against the host "
        "functions; nonblocking stream; stream synchronize");
    checked(eclair::cuda::check_bf16_convert_image(0), "BF16 kernel image");
    ECLAIR_CHECK(eclair::cuda::check_bf16_convert_image(-1) == cudaErrorInvalidValue);
    check_rejections();
    check_widening();
    check_narrowing();
    std::puts(
        "checked: all BF16 patterns, every rounding decision, ties, overflow, subnormals, "
        "NaN, vector and scalar paths, tails, stray stores, null and in-place rejection");
}

}  // namespace

int main() {
    return eclair::test::contain(run);
}
