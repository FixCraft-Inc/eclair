/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/formats/fp8.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Elementwise FP8 conversions between device-resident buffers. The caller owns
// both buffers and the stream until completion. `count` is in elements. Input
// and output must not overlap. Invalid arguments enqueue no work and return
// cudaErrorInvalidValue. Nothing is allocated or synchronized, and success
// means enqueued. Results equal the host functions in eclair/formats bit for
// bit, NaN payloads aside.
namespace eclair::cuda {

// FP8 bytes to binary16 bits, exact (formats::fp8_to_fp16_bits).
[[nodiscard]] cudaError_t launch_fp8_to_fp16(const std::uint8_t* input, std::uint16_t* output,
                                             std::size_t count, formats::Fp8Format format,
                                             cudaStream_t stream) noexcept;

// binary16 bits or FP32 values to FP8 bytes, with formats::encode_fp8's
// rounding, saturation and NaN rules.
[[nodiscard]] cudaError_t launch_fp16_to_fp8(const std::uint16_t* input, std::uint8_t* output,
                                             std::size_t count, formats::Fp8Format format,
                                             cudaStream_t stream) noexcept;
[[nodiscard]] cudaError_t launch_fp32_to_fp8(const float* input, std::uint8_t* output,
                                             std::size_t count, formats::Fp8Format format,
                                             cudaStream_t stream) noexcept;

// Kernel-image probes for the decode and the encode kernels. They return
// cudaSuccess when this build has images the device can load, initialize the
// device's primary context and restore the caller's current device.
[[nodiscard]] cudaError_t check_fp8_to_fp16_image(int device_index) noexcept;
[[nodiscard]] cudaError_t check_fp8_encode_image(int device_index) noexcept;

}  // namespace eclair::cuda
