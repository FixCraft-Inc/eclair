/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/fp8_convert.hpp"

#include "convert_kernel.hpp"
#include "kernel_image.hpp"

#include "eclair/formats/fp16.hpp"

#include <cuda_fp16.h>

#include <cstdint>

namespace eclair::cuda {
namespace {

using detail::convert_kernel;
using detail::launch_conversion;

// The format is a runtime member: every thread takes the same branch, and
// one kernel per direction keeps the image probes small.
struct Fp8ToFp16 {
    formats::Fp8Format format;
    __device__ std::uint16_t operator()(std::uint8_t bits) const {
        return formats::fp8_to_fp16_bits(bits, format);
    }
};

// binary16 widens exactly to binary32, so encoding the widened value matches
// the reference encode. The hardware widening returns a NaN without its sign,
// so the input's sign bit is put back. Every other value already carries it.
struct Fp16ToFp8 {
    formats::Fp8Format format;
    __device__ std::uint8_t operator()(std::uint16_t bits) const {
        const std::uint32_t wide = __float_as_uint(__half2float(__ushort_as_half(bits))) |
                                   (static_cast<std::uint32_t>(bits & 0x8000U) << 16U);
        return formats::fp32_bits_to_fp8(wide, format);
    }
};

struct Fp32ToFp8 {
    formats::Fp8Format format;
    __device__ std::uint8_t operator()(float value) const {
        return formats::fp32_bits_to_fp8(__float_as_uint(value), format);
    }
};

[[nodiscard]] bool known(formats::Fp8Format format) noexcept {
    return format == formats::Fp8Format::E4M3FN || format == formats::Fp8Format::E5M2;
}

}  // namespace

cudaError_t launch_fp8_to_fp16(const std::uint8_t* input, std::uint16_t* output, std::size_t count,
                               formats::Fp8Format format, cudaStream_t stream) noexcept {
    if (!known(format)) {
        return cudaErrorInvalidValue;
    }
    return launch_conversion(input, output, count, Fp8ToFp16{format}, stream);
}

cudaError_t launch_fp16_to_fp8(const std::uint16_t* input, std::uint8_t* output, std::size_t count,
                               formats::Fp8Format format, cudaStream_t stream) noexcept {
    if (!known(format)) {
        return cudaErrorInvalidValue;
    }
    return launch_conversion(input, output, count, Fp16ToFp8{format}, stream);
}

cudaError_t launch_fp32_to_fp8(const float* input, std::uint8_t* output, std::size_t count,
                               formats::Fp8Format format, cudaStream_t stream) noexcept {
    if (!known(format)) {
        return cudaErrorInvalidValue;
    }
    return launch_conversion(input, output, count, Fp32ToFp8{format}, stream);
}

cudaError_t check_fp8_to_fp16_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index,
                                       convert_kernel<std::uint8_t, std::uint16_t, Fp8ToFp16>);
}

cudaError_t check_fp8_encode_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index,
                                       convert_kernel<std::uint16_t, std::uint8_t, Fp16ToFp8>,
                                       convert_kernel<float, std::uint8_t, Fp32ToFp8>);
}

}  // namespace eclair::cuda
