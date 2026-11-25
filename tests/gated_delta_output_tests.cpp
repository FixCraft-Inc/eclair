/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/formats/fp16.hpp"
#include "eclair/host/gated_delta_output.hpp"

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace {

using eclair::formats::decode_fp16;
using eclair::formats::encode_fp16;
using eclair::ops::Status;

struct Problem {
    eclair::ops::GatedDeltaOutputShape shape{2, 1, 1, 1, 1, 1, 1, 0.5F};
    std::vector<std::uint16_t> q{encode_fp16(1), encode_fp16(2)};
    std::vector<std::uint16_t> k{encode_fp16(2), encode_fp16(3)};
    std::vector<std::uint16_t> v{encode_fp16(3), encode_fp16(5)};
    std::vector<std::uint16_t> state{encode_fp16(4)}, out{0x7BFF, 0x7BFF};
    std::vector<float> gates{0, 0};
    std::vector<std::int32_t> starts{0, 2}, indices{0, 0};
    auto buffers() {
        return eclair::ops::GatedDeltaOutputBuffers{{q.data(), q.size()},
                                                    {k.data(), k.size()},
                                                    {v.data(), v.size()},
                                                    {state.data(), state.size()},
                                                    {gates.data(), gates.size()},
                                                    {starts.data(), starts.size()},
                                                    {indices.data(), indices.size()},
                                                    {out.data(), out.size()}};
    }
    auto run() { return eclair::host::gated_delta_output_reference(shape, buffers()); }
};

void test_known_answers() {
    Problem p;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(decode_fp16(p.out[0]) == 5.0F);   // (1*4 + 1*2*3)/2.
    ECLAIR_CHECK(decode_fp16(p.out[1]) == 25.0F);  // (2*4 + 2*2*3 + 2*3*5)/2.
    p.gates = {std::log(0.5F), std::log(0.25F)};
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(decode_fp16(p.out[0]) == 4.0F);
    ECLAIR_CHECK(decode_fp16(p.out[1]) == 19.0F);
    // (1 + 2^-10)^2 rounds down by 2^-20 as a binary16 correlation.
    // Its exact negative in the prior exposes that rounding before summation.
    p.q = {0x3C01, 0};
    p.k = {0x3C01, 0};
    p.v = {0x3C00, 0};
    p.state = {0xBC01};
    p.gates = {0, 0};
    p.shape.scale = 1;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(p.out[0] == 0x8010);  // -2^-20; omitting the rounding gives zero.
    p.q = {0x3C00, 0};
    p.k = {0x3C00, 0};
    p.v = {0xC000, 0};
    p.state = {0x4000};
    p.shape.scale = std::numeric_limits<float>::max();
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(p.out[0] == 0);  // Scale after cancellation: no infinity - infinity.
}

void test_validation() {
    Problem p;
    auto b = p.buffers();
    b.output.data = p.q.data();
    ECLAIR_CHECK(eclair::ops::validate_gated_delta_output(p.shape, b) == Status::InvalidArgument);
    b = p.buffers();
    b.values.size = 1;
    ECLAIR_CHECK(eclair::ops::validate_gated_delta_output(p.shape, b) == Status::BufferTooSmall);
    p.shape.scale = std::numeric_limits<float>::infinity();
    ECLAIR_CHECK(p.run() == Status::InvalidArgument);
    p.shape.scale = 1;
    p.shape.tokens = std::numeric_limits<std::size_t>::max();
    p.shape.key_size = 2;
    ECLAIR_CHECK(p.run() == Status::SizeOverflow);
    p.shape.tokens = 0;
    ECLAIR_CHECK(eclair::ops::validate_gated_delta_output(p.shape, {}) == Status::Success);
    Problem invalid;
    for (const auto& pair : {std::pair{0, 1}, std::pair{1, 0}, std::pair{-1, 0}}) {
        invalid.indices = {pair.first, pair.second};
        ECLAIR_CHECK(invalid.run() == Status::InvalidArgument);
        ECLAIR_CHECK(invalid.out[0] == 0x7BFF);
    }
}

void test_nonfinite() {
    Problem p;
    p.v[1] = 0x7E00;  // Masked for row zero; reference skips it.
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isfinite(decode_fp16(p.out[0])));
    ECLAIR_CHECK(std::isnan(decode_fp16(p.out[1])));
    p.v = {0x0001, 0x8000};  // Smallest subnormal and negative zero.
    p.q = {encode_fp16(1), encode_fp16(1)};
    p.k = {encode_fp16(1), encode_fp16(1)};
    p.state[0] = 0;
    p.shape.scale = 1;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(p.out[0] == 0x0001 && p.out[1] == 0x0001);
    p.q[0] = 0x7E00;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isnan(decode_fp16(p.out[0])));
    p.q[0] = encode_fp16(1);
    p.k[0] = 0x7C00;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(std::isinf(decode_fp16(p.out[0])));
    p.k[0] = encode_fp16(1);
    p.gates[0] = -1000;
    ECLAIR_CHECK(p.run() == Status::Success);
    ECLAIR_CHECK(p.out[0] == 0x0001);  // Zero state decay, self-correlation still one.
}

}  // namespace

int main() {
    test_known_answers();
    test_validation();
    test_nonfinite();
    return eclair::test::finish();
}
