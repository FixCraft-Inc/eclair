/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/host/w4a16_linear.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::ops::Status;

struct Problem {
    eclair::ops::W4A16LinearShape shape{1, 1, 128, 2};
    std::vector<std::uint16_t> a = std::vector<std::uint16_t>(128, 0);
    std::vector<std::uint32_t> w = std::vector<std::uint32_t>(16, 0x88888888U);
    std::vector<std::uint16_t> scales{0x3C00}, out{0x7BFF};
    std::vector<float> scratch = std::vector<float>(2, 123);
    auto buffers() {
        return eclair::ops::W4A16LinearBuffers{{a.data(), a.size()},
                                               {w.data(), w.size()},
                                               {scales.data(), scales.size()},
                                               {out.data(), out.size()},
                                               {scratch.data(), scratch.size()}};
    }
    auto run() { return eclair::host::w4a16_linear_reference(shape, buffers()); }
};

void test_packing_and_rounding() {
    Problem p;
    // Fixed storage word: logical values 0..7, independent of extraction code.
    p.w[0] = 0x75316420U;
    for (std::size_t k = 0; k < 8; ++k) {
        p.a.assign(128, 0);
        p.a[k] = 0x3C00;
        ECLAIR_CHECK(p.run() == Status::Success);
        ECLAIR_CHECK(decode_fp16(p.out[0]) == static_cast<float>(k) - 8.0F);
        ECLAIR_CHECK(p.scratch[0] == 123);
    }
    // Every unsigned nibble, with a scale whose product needs half rounding.
    p.a.assign(128, 0);
    p.a[0] = 0x3C00;
    p.scales[0] = 0x3C01;
    for (unsigned int q = 0; q < 16; ++q) {
        p.w[0] = 0x88888880U | q;
        ECLAIR_CHECK(p.run() == Status::Success);
        const float exact = (static_cast<float>(q) - 8.0F) * (1.0F + 1.0F / 1024.0F);
        ECLAIR_CHECK(p.out[0] == encode_fp16(exact));
    }
    // 3 * (1 + 2^-10) rounds upward before multiplying the activation.
    // Add its exact negative: omitting the weight rounding returns -2^-10.
    // Use a second scale group to cancel the rounded first weight exactly.
    p.shape.k = 256;
    p.a.assign(256, 0);
    p.a[0] = 0x3C00;
    p.a[128] = 0xC202;
    p.w.assign(32, 0x88888888U);
    p.w[0] = 0x8888888BU;
    p.w[16] = 0x88888889U;
    p.scales = {0x3C01, 0x3C00};
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(decode_fp16(p.out[0]) == 0);
}

void test_special_values() {
    Problem p;
    p.w[0] = 0x88888889U;
    p.a[0] = 0x3C00;
    for (const auto bits : {0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF}) {
        p.scales[0] = static_cast<std::uint16_t>(bits);
        ECLAIR_CHECK(p.run() == Status::Success);
        ECLAIR_CHECK(p.out[0] == bits);
    }
    for (const auto bits : {0x0000, 0x8000}) {
        p.scales[0] = static_cast<std::uint16_t>(bits);
        ECLAIR_CHECK(p.run() == Status::Success);
        ECLAIR_CHECK(decode_fp16(p.out[0]) == 0);
    }
    p.scales[0] = 0x7C00;  // Zero quantized weights times infinity are NaN too.
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isnan(decode_fp16(p.out[0])));
    p.scales[0] = 0x7E00;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isnan(decode_fp16(p.out[0])));
    p.scales[0] = 0x3C00;
    p.a[0] = 0x7C00;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isinf(decode_fp16(p.out[0])));
    p.scales[0] = 0x7BFF;
    p.a[0] = 0x4000;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isinf(decode_fp16(p.out[0])));
}

void test_validation() {
    Problem p;
    auto b = p.buffers();
    b.output.data = p.a.data();
    ECLAIR_CHECK(eclair::ops::validate_w4a16_linear(p.shape, b) == Status::InvalidArgument);
    b = p.buffers();
    b.workspace.data = reinterpret_cast<float*>(p.w.data());
    ECLAIR_CHECK(eclair::ops::validate_w4a16_linear(p.shape, b) == Status::InvalidArgument);
    b = p.buffers();
    b.scales.size = 0;
    ECLAIR_CHECK(eclair::ops::validate_w4a16_linear(p.shape, b) == Status::BufferTooSmall);
    p.shape.k = 127;
    ECLAIR_CHECK(p.run() == Status::InvalidArgument);
    p.shape.k = 128;
    p.shape.splits = 0;
    ECLAIR_CHECK(p.run() == Status::InvalidArgument);
    p.shape.splits = 2;
    p.shape.n = std::numeric_limits<std::size_t>::max();
    ECLAIR_CHECK(p.run() == Status::SizeOverflow);
    ECLAIR_CHECK(p.out[0] == 0x7BFF);
    p.shape.m = 0;
    ECLAIR_CHECK(eclair::ops::validate_w4a16_linear(p.shape, {}) == Status::Success);
}

void test_dequantize_validation() {
    using eclair::ops::validate_w4a16_dequantize;
    std::vector<std::uint32_t> w(32, 0x88888888U);
    std::vector<std::uint16_t> scales(2, 0x3C00), out(256, 0x7BFF);
    const eclair::ops::W4A16DequantizeBuffers b{
        {w.data(), w.size()}, {scales.data(), scales.size()}, {out.data(), out.size()}};
    ECLAIR_CHECK(validate_w4a16_dequantize(1, 256, b) == Status::Success);
    ECLAIR_CHECK(validate_w4a16_dequantize(1, 384, b) == Status::BufferTooSmall);
    ECLAIR_CHECK(validate_w4a16_dequantize(1, 200, b) == Status::InvalidArgument);
    ECLAIR_CHECK(validate_w4a16_dequantize(1, 0, b) == Status::InvalidArgument);
    ECLAIR_CHECK(validate_w4a16_dequantize(0, 0, {}) == Status::Success);
    ECLAIR_CHECK(validate_w4a16_dequantize(std::numeric_limits<std::size_t>::max(), 128, b) ==
                 Status::SizeOverflow);
    auto overlap = b;
    overlap.dequantized.data = reinterpret_cast<std::uint16_t*>(w.data()) + 1;
    overlap.dequantized.size = 128;
    ECLAIR_CHECK(validate_w4a16_dequantize(1, 128, overlap) == Status::InvalidArgument);
    overlap = b;
    overlap.dequantized.data = scales.data();
    overlap.dequantized.size = 128;
    ECLAIR_CHECK(validate_w4a16_dequantize(1, 128, overlap) == Status::InvalidArgument);
}

void test_gemm_validation() {
    using eclair::ops::validate_w4a16_gemm;
    std::vector<std::uint16_t> a(2 * 128, 0), scales(1, 0x3C00), weights(128, 0), out(2, 0);
    std::vector<std::uint32_t> w(16, 0x88888888U);
    std::vector<std::byte> workspace(512);
    const eclair::ops::W4A16LinearShape shape{2, 1, 128, 1};
    const eclair::ops::W4A16GemmBuffers b{
        {a.data(), a.size()},           {w.data(), w.size()},
        {scales.data(), scales.size()}, {weights.data(), weights.size()},
        {out.data(), out.size()},       {workspace.data(), workspace.size()}};
    ECLAIR_CHECK(validate_w4a16_gemm(shape, b) == Status::Success);
    auto empty_workspace = b;
    empty_workspace.workspace = {};
    ECLAIR_CHECK(validate_w4a16_gemm(shape, empty_workspace) == Status::Success);
    ECLAIR_CHECK(validate_w4a16_gemm({2, 1, 128, 2}, b) == Status::InvalidArgument);
    ECLAIR_CHECK(validate_w4a16_gemm({2, 1, 64, 1}, b) == Status::InvalidArgument);
    ECLAIR_CHECK(validate_w4a16_gemm({3, 1, 128, 1}, b) == Status::BufferTooSmall);
    ECLAIR_CHECK(validate_w4a16_gemm({0, 1, 128, 2}, {}) == Status::Success);
    ECLAIR_CHECK(validate_w4a16_gemm({std::numeric_limits<std::size_t>::max(), 2, 128, 1}, b) ==
                 Status::SizeOverflow);
    auto invalid = b;
    invalid.dequantized.size = 127;
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::BufferTooSmall);
    // Each written buffer against an input and against the other written ones.
    invalid = b;
    invalid.output.data = a.data();
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::InvalidArgument);
    invalid = b;
    invalid.dequantized.data = a.data() + 1;
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::InvalidArgument);
    invalid = b;
    invalid.workspace.data = reinterpret_cast<std::byte*>(scales.data());
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::InvalidArgument);
    invalid = b;
    invalid.output.data = weights.data() + 127;
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::InvalidArgument);
    invalid = b;
    invalid.workspace.data = reinterpret_cast<std::byte*>(out.data());
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::InvalidArgument);
    invalid = b;
    invalid.workspace.data = reinterpret_cast<std::byte*>(weights.data());
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::InvalidArgument);
    // Inputs may alias each other.
    invalid = b;
    invalid.scales.data = a.data();
    ECLAIR_CHECK(validate_w4a16_gemm(shape, invalid) == Status::Success);
}

}  // namespace

int main() {
    test_packing_and_rounding();
    test_special_values();
    test_validation();
    test_dequantize_validation();
    test_gemm_validation();
    return eclair::test::finish();
}
