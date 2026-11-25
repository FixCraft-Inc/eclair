/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#ifndef ECLAIR_C_W4A16_LINEAR_H
#define ECLAIR_C_W4A16_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECLAIR_W4A16_LINEAR_ABI 1u

/* Experimental eclair/ops/w4a16_linear.hpp contract. Device pointers and
 * element counts, including caller-owned FP32 scratch [splits,m,n]. Buffers
 * and stream stay alive until completion. No allocation or synchronization. */
typedef struct EclairW4A16LinearArgs {
    uint32_t abi;
    const void* activations;
    uint64_t activation_elements;
    const uint32_t* weights;
    uint64_t weight_elements;
    const void* scales;
    uint64_t scale_elements;
    void* output;
    uint64_t output_elements;
    float* workspace;
    uint64_t workspace_elements;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t splits;
} EclairW4A16LinearArgs;

/* cudaError_t, zero when enqueued. Rejected arguments enqueue no work;
 * exceptions never cross the C boundary. Image check is not a quality gate. */
int eclair_w4a16_linear(const EclairW4A16LinearArgs* args, void* stream);
int eclair_w4a16_linear_image(int device_index);

#ifdef __cplusplus
}
#endif
#endif
