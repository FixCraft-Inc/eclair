/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Elementwise bfloat16 conversions between device-resident buffers, with the
// ownership and validation rules of eclair/cuda/fp8_convert.hpp. Results equal
// formats::bf16_to_fp32_bits and formats::fp32_bits_to_bf16 bit for bit, NaN
// payloads aside. They use no BF16 hardware instructions.
namespace eclair::cuda {

[[nodiscard]] cudaError_t launch_bf16_to_fp32(const std::uint16_t* input, float* output,
                                              std::size_t count, cudaStream_t stream) noexcept;

[[nodiscard]] cudaError_t launch_fp32_to_bf16(const float* input, std::uint16_t* output,
                                              std::size_t count, cudaStream_t stream) noexcept;

// Kernel-image probe for both directions, as check_fp8_encode_image.
[[nodiscard]] cudaError_t check_bf16_convert_image(int device_index) noexcept;

}  // namespace eclair::cuda
