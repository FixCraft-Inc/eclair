/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#ifndef ECLAIR_C_W4A16_GEMM_H
#define ECLAIR_C_W4A16_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECLAIR_W4A16_GEMM_ABI 1u

/* Experimental large-M implementation of eclair/ops/w4a16_linear.hpp:
 * Eclair reconstructs the binary16 weights into caller-owned [k,n] scratch,
 * then cuBLAS multiplies with FP32 compute and FP32-only split reductions.
 * The context owns one cuBLAS handle for one device. Create it once, use it
 * from one thread at a time with that device current, and destroy it after
 * its last call completed. Creation allocates cuBLAS's default workspace. */
typedef struct EclairW4A16Gemm EclairW4A16Gemm;

/* Device pointers and element counts. The workspace is in bytes, may be
 * empty and must start on a 256-byte boundary. No allocation or
 * synchronization. Buffers and stream stay alive until completion. */
typedef struct EclairW4A16GemmArgs {
    uint32_t abi;
    const void* activations;
    uint64_t activation_elements;
    const uint32_t* weights;
    uint64_t weight_elements;
    const void* scales;
    uint64_t scale_elements;
    void* dequantized;
    uint64_t dequantized_elements;
    void* output;
    uint64_t output_elements;
    void* workspace;
    uint64_t workspace_bytes;
    uint64_t m;
    uint64_t n;
    uint64_t k;
} EclairW4A16GemmArgs;

/* cudaError_t, zero on success. On failure *gemm is NULL. */
int eclair_w4a16_gemm_create(int device_index, EclairW4A16Gemm** gemm);
/* Accepts NULL. */
void eclair_w4a16_gemm_destroy(EclairW4A16Gemm* gemm);
/* cudaError_t, zero when enqueued. Rejected arguments enqueue no work and
 * exceptions never cross the C boundary. */
int eclair_w4a16_gemm(EclairW4A16Gemm* gemm, const EclairW4A16GemmArgs* args, void* stream);
/* The reconstruction kernels' image check. Not a quality gate. */
int eclair_w4a16_gemm_image(int device_index);

#ifdef __cplusplus
}
#endif
#endif
