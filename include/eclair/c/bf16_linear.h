/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#ifndef ECLAIR_C_BF16_LINEAR_H
#define ECLAIR_C_BF16_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECLAIR_BF16_LINEAR_ABI 1u

/* Row-scaled binary16 operands and FP32 output; numerical contract is
 * eclair/ops/bf16_linear.hpp. Zero shifts give an ordinary FP16 linear.
 * Device pointers only, extents in elements. Caller owns buffers/stream.
 * No allocation or synchronization. bias_elements == 0 means no bias. */
typedef struct EclairBf16LinearArgs {
    uint32_t abi;
    const void* activations;
    uint64_t activation_elements;
    const int32_t* activation_shifts;
    uint64_t activation_shift_elements;
    const void* weights;
    uint64_t weight_elements;
    const int32_t* weight_shifts;
    uint64_t weight_shift_elements;
    const float* bias;
    uint64_t bias_elements;
    float* output;
    uint64_t output_elements;
    uint64_t m;
    uint64_t n;
    uint64_t k;
} EclairBf16LinearArgs;

/* cudaError_t, zero when enqueued. Wrong ABI or invalid arguments enqueue
 * nothing. Exceptions never cross this boundary. */
int eclair_bf16_linear(const EclairBf16LinearArgs* args, void* stream);
int eclair_bf16_linear_image(int device_index);

#ifdef __cplusplus
}
#endif
#endif
