/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/bf16_convert.hpp"

#include "convert_kernel.hpp"
#include "kernel_image.hpp"

#include "eclair/formats/bf16.hpp"

#include <cstdint>

namespace eclair::cuda {
namespace {

using detail::convert_kernel;
using detail::launch_conversion;

struct Bf16ToFp32 {
    __device__ float operator()(std::uint16_t bits) const {
        return __uint_as_float(formats::bf16_to_fp32_bits(bits));
    }
};

struct Fp32ToBf16 {
    __device__ std::uint16_t operator()(float value) const {
        return formats::fp32_bits_to_bf16(__float_as_uint(value));
    }
};

}  // namespace

cudaError_t launch_bf16_to_fp32(const std::uint16_t* input, float* output, std::size_t count,
                                cudaStream_t stream) noexcept {
    return launch_conversion(input, output, count, Bf16ToFp32{}, stream);
}

cudaError_t launch_fp32_to_bf16(const float* input, std::uint16_t* output, std::size_t count,
                                cudaStream_t stream) noexcept {
    return launch_conversion(input, output, count, Fp32ToBf16{}, stream);
}

cudaError_t check_bf16_convert_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index,
                                       convert_kernel<std::uint16_t, float, Bf16ToFp32>,
                                       convert_kernel<float, std::uint16_t, Fp32ToBf16>);
}

}  // namespace eclair::cuda
