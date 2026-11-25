/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"
#include "cuda_support.hpp"

#include "eclair/cuda/fp8_convert.hpp"
#include "eclair/formats/fp16.hpp"
#include "eclair/formats/fp8.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace {

using eclair::formats::Fp8Format;
using eclair::test::checked;
using eclair::test::DeviceBuffer;
using eclair::test::Stream;

constexpr std::initializer_list<Fp8Format> kFormats = {Fp8Format::E4M3FN, Fp8Format::E5M2};

template <typename T>
[[nodiscard]] bool same_bits(const T& left, const T& right) {
    return std::memcmp(&left, &right, sizeof(T)) == 0;
}

// Launches a conversion of `input` into a device view that starts `offset`
// elements into its buffers, so offsets of 1-3 miss the vector alignment and
// take the scalar path. Sentinels around the output view catch stray stores.
template <typename In, typename Out, typename Launch>
[[nodiscard]] std::vector<Out> convert_on_device(const std::vector<In>& input, std::size_t offset,
                                                 Out sentinel, Launch launch) {
    Stream stream;
    std::vector<In> shifted(offset, In{});
    shifted.insert(shifted.end(), input.begin(), input.end());
    const DeviceBuffer<In> device_input(shifted, stream.get());
    const DeviceBuffer<Out> device_output(std::vector<Out>(offset + input.size() + 1U, sentinel),
                                          stream.get());
    checked(launch(device_input.data() + offset, device_output.data() + offset, input.size(),
                   stream.get()),
            "conversion launch");
    const std::vector<Out> output = device_output.download(stream.get());
    for (std::size_t index = 0; index < offset; ++index) {
        ECLAIR_CHECK(same_bits(output[index], sentinel));
    }
    ECLAIR_CHECK(same_bits(output.back(), sentinel));
    return {output.begin() + static_cast<std::ptrdiff_t>(offset), output.end() - 1};
}

// Views that take the vector path with no tail, tails of one to three
// elements, and the scalar path at several misalignments.
struct Placement {
    std::size_t offset;
    std::size_t trim;
};
constexpr Placement kPlacements[] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {4, 0}, {1, 0}, {3, 2}};

template <typename In>
[[nodiscard]] std::vector<In> trimmed(const std::vector<In>& values, std::size_t trim) {
    return {values.begin(), values.end() - static_cast<std::ptrdiff_t>(trim)};
}

// Every byte pattern, against the float oracle rounded to binary16.
void check_decode(Fp8Format format) {
    std::vector<std::uint8_t> input(1024U);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>(index * 7U);
    }
    for (const Placement placement : kPlacements) {
        const auto values = trimmed(input, placement.trim);
        const auto output = convert_on_device(values, placement.offset, std::uint16_t{0x5A5AU},
                                              [format](const std::uint8_t* in, std::uint16_t* out,
                                                       std::size_t count, cudaStream_t stream) {
                                                  return eclair::cuda::launch_fp8_to_fp16(
                                                      in, out, count, format, stream);
                                              });
        for (std::size_t index = 0; index < values.size(); ++index) {
            const float decoded = eclair::formats::decode_fp8(values[index], format);
            if (std::isnan(decoded)) {
                ECLAIR_CHECK(std::isnan(eclair::formats::decode_fp16(output[index])));
            } else {
                ECLAIR_CHECK(output[index] == eclair::formats::encode_fp16(decoded));
            }
        }
    }
}

// Every binary16 pattern, compared bit for bit with the reference encoder.
void check_encode_from_fp16(Fp8Format format) {
    std::vector<std::uint16_t> input(65536U);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint16_t>(index);
    }
    for (const Placement placement : kPlacements) {
        const auto values = trimmed(input, placement.trim);
        const auto output = convert_on_device(values, placement.offset, std::uint8_t{0xA5U},
                                              [format](const std::uint16_t* in, std::uint8_t* out,
                                                       std::size_t count, cudaStream_t stream) {
                                                  return eclair::cuda::launch_fp16_to_fp8(
                                                      in, out, count, format, stream);
                                              });
        std::size_t mismatches = 0;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const std::uint8_t expected =
                eclair::formats::encode_fp8(eclair::formats::decode_fp16(values[index]), format);
            if (output[index] != expected && mismatches++ < 4U) {
                std::fprintf(stderr, "FP16 0x%04x: actual 0x%02x expected 0x%02x\n", values[index],
                             output[index], expected);
            }
        }
        ECLAIR_CHECK(mismatches == 0U);
    }
}

// binary32 inputs: every binary16 value, each FP8 value's neighborhood and
// rounding midpoints, specials and random bit patterns of both signs.
[[nodiscard]] std::vector<float> fp32_inputs(Fp8Format format) {
    std::vector<float> inputs = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()};
    for (std::uint32_t bits = 0; bits < 65536U; ++bits) {
        inputs.push_back(eclair::formats::decode_fp16(static_cast<std::uint16_t>(bits)));
    }
    for (unsigned int bits = 0; bits < 0x7FU; ++bits) {
        const float value = eclair::formats::decode_fp8(static_cast<std::uint8_t>(bits), format);
        const float next =
            eclair::formats::decode_fp8(static_cast<std::uint8_t>(bits + 1U), format);
        if (!std::isfinite(value) || !std::isfinite(next)) {
            continue;
        }
        const float midpoint = value + (next - value) / 2.0F;
        for (const float input : {midpoint, std::nextafter(midpoint, 0.0F),
                                  std::nextafter(midpoint, std::numeric_limits<float>::max())}) {
            inputs.push_back(input);
        }
    }
    std::mt19937 random(70U);
    for (int index = 0; index < (1 << 18); ++index) {
        const auto bits = static_cast<std::uint32_t>(random());
        float value = 0.0F;
        std::memcpy(&value, &bits, sizeof(value));
        inputs.push_back(value);
    }
    const std::size_t positive = inputs.size();
    for (std::size_t index = 0; index < positive; ++index) {
        inputs.push_back(-inputs[index]);
    }
    return inputs;
}

void check_encode_from_fp32(Fp8Format format) {
    const std::vector<float> input = fp32_inputs(format);
    std::vector<std::uint8_t> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] = eclair::formats::encode_fp8(input[index], format);
    }
    for (const Placement placement : kPlacements) {
        const auto values = trimmed(input, placement.trim);
        const auto output = convert_on_device(
            values, placement.offset, std::uint8_t{0xA5U},
            [format](const float* in, std::uint8_t* out, std::size_t count, cudaStream_t stream) {
                return eclair::cuda::launch_fp32_to_fp8(in, out, count, format, stream);
            });
        std::size_t mismatches = 0;
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (output[index] != expected[index] && mismatches++ < 4U) {
                std::fprintf(stderr, "FP32 %a: actual 0x%02x expected 0x%02x\n",
                             static_cast<double>(values[index]), output[index], expected[index]);
            }
        }
        ECLAIR_CHECK(mismatches == 0U);
    }
}

// Rejections enqueue nothing, which the untouched output proves.
void check_rejections() {
    using eclair::cuda::launch_fp16_to_fp8;
    using eclair::cuda::launch_fp32_to_fp8;
    using eclair::cuda::launch_fp8_to_fp16;
    ECLAIR_CHECK(launch_fp8_to_fp16(nullptr, nullptr, 0U, Fp8Format::E4M3FN, nullptr) ==
                 cudaSuccess);
    ECLAIR_CHECK(launch_fp8_to_fp16(nullptr, nullptr, 1U, Fp8Format::E4M3FN, nullptr) ==
                 cudaErrorInvalidValue);
    ECLAIR_CHECK(launch_fp32_to_fp8(nullptr, nullptr, 1U, Fp8Format::E5M2, nullptr) ==
                 cudaErrorInvalidValue);

    Stream stream;
    const DeviceBuffer<std::uint16_t> buffer(std::vector<std::uint16_t>(64U, 0x3C00U),
                                             stream.get());
    auto* bytes = reinterpret_cast<std::uint8_t*>(buffer.data());
    const auto unknown = static_cast<Fp8Format>(7);
    ECLAIR_CHECK(launch_fp16_to_fp8(buffer.data(), bytes + 64, 32U, unknown, stream.get()) ==
                 cudaErrorInvalidValue);
    // The output's first byte is the input's last: in-place work would race.
    ECLAIR_CHECK(launch_fp16_to_fp8(buffer.data(), bytes + 63, 32U, Fp8Format::E4M3FN,
                                    stream.get()) == cudaErrorInvalidValue);
    ECLAIR_CHECK(launch_fp8_to_fp16(bytes + 1, buffer.data(), 8U, Fp8Format::E5M2, stream.get()) ==
                 cudaErrorInvalidValue);
    ECLAIR_CHECK(launch_fp8_to_fp16(bytes, buffer.data() + 1,
                                    std::numeric_limits<std::size_t>::max(), Fp8Format::E5M2,
                                    stream.get()) == cudaErrorInvalidValue);
    for (const std::uint16_t value : buffer.download(stream.get())) {
        ECLAIR_CHECK(value == 0x3C00U);
    }
}

// Effective bandwidth of each conversion on 2^25 elements, against a
// device-to-device copy that moves the same number of bytes.
void benchmark() {
    constexpr std::size_t kCount = std::size_t{1} << 25U;
    std::puts(
        "BENCH elements=33554432 warmup=100ms-per-path samples=9 stream=nonblocking "
        "timing=CUDA-events synchronization=event; GB/s counts bytes read plus written; "
        "baseline=cudaMemcpyAsync device-to-device of the same total bytes");
    Stream stream;
    std::vector<std::uint8_t> fp8(kCount);
    std::vector<std::uint16_t> fp16(kCount);
    for (std::size_t index = 0; index < kCount; ++index) {
        fp8[index] = static_cast<std::uint8_t>(index * 7U);
        fp16[index] = static_cast<std::uint16_t>(index * 2654435761U >> 16U);
    }
    const DeviceBuffer<std::uint8_t> device_fp8(fp8, stream.get());
    const DeviceBuffer<std::uint16_t> device_fp16(fp16, stream.get());
    const DeviceBuffer<float> device_fp32(kCount);
    const DeviceBuffer<std::uint16_t> half_output(kCount);
    const DeviceBuffer<std::uint8_t> byte_output(kCount);
    const DeviceBuffer<std::uint8_t> copy_source(3U * kCount);
    const DeviceBuffer<std::uint8_t> copy_target(3U * kCount);
    checked(cudaMemsetAsync(device_fp32.data(), 0x3F, kCount * sizeof(float), stream.get()),
            "FP32 fill");

    const auto report = [&](const char* name, std::size_t bytes, auto launch) {
        const auto samples = eclair::test::timings(stream.get(), launch);
        std::printf("BENCH %s bytes=%zu median_us=%.3f max_us=%.3f GBps=%.1f\n", name, bytes,
                    samples[4] * 1000.0F, samples.back() * 1000.0F,
                    static_cast<double>(bytes) / (static_cast<double>(samples[4]) * 1.0e6));
    };
    for (const Fp8Format format : kFormats) {
        const char* suffix = format == Fp8Format::E4M3FN ? "E4M3FN" : "E5M2";
        std::printf("BENCH format=%s\n", suffix);
        report("fp8_to_fp16", 3U * kCount, [&] {
            checked(eclair::cuda::launch_fp8_to_fp16(device_fp8.data(), half_output.data(), kCount,
                                                     format, stream.get()),
                    "decode");
        });
        report("fp16_to_fp8", 3U * kCount, [&] {
            checked(eclair::cuda::launch_fp16_to_fp8(device_fp16.data(), byte_output.data(), kCount,
                                                     format, stream.get()),
                    "encode FP16");
        });
        report("fp32_to_fp8", 5U * kCount, [&] {
            checked(eclair::cuda::launch_fp32_to_fp8(device_fp32.data(), byte_output.data(), kCount,
                                                     format, stream.get()),
                    "encode FP32");
        });
    }
    report("copy_d2d", 6U * kCount, [&] {
        checked(cudaMemcpyAsync(copy_target.data(), copy_source.data(), 3U * kCount,
                                cudaMemcpyDeviceToDevice, stream.get()),
                "copy");
    });
}

void run() {
    eclair::test::print_device_header();
    std::puts(
        "FP8 decode to binary16 and encode from binary16/FP32; bit-exact against the host "
        "reference (decoded NaNs by class); nonblocking stream; stream synchronize");
    checked(eclair::cuda::check_fp8_to_fp16_image(0), "decode kernel image");
    checked(eclair::cuda::check_fp8_encode_image(0), "encode kernel image");
    ECLAIR_CHECK(eclair::cuda::check_fp8_encode_image(-1) == cudaErrorInvalidValue);
    check_rejections();
    for (const Fp8Format format : kFormats) {
        check_decode(format);
        check_encode_from_fp16(format);
        check_encode_from_fp32(format);
    }
    std::puts(
        "checked: all FP8 and binary16 patterns, FP8 rounding midpoints, random binary32, "
        "saturation, NaN, signed zero, vector and scalar paths, tails, stray stores, "
        "null, overlapping, wrapping and unknown-format rejection");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--benchmark")) {
        std::fprintf(stderr, "Usage: eclair_cuda_fp8_tests [--benchmark]\n");
        return 2;
    }
    return eclair::test::contain([argc] {
        run();
        if (argc == 2 && eclair::test::failure_count() == 0) {
            benchmark();
        }
    });
}
