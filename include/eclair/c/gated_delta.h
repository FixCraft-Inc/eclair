/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#ifndef ECLAIR_C_GATED_DELTA_H
#define ECLAIR_C_GATED_DELTA_H

/* C interface of the gated delta rule state pass in libeclair_kernels. It wraps
 * eclair::cuda::launch_gated_delta_states, whose contract is in
 * eclair/ops/gated_delta.hpp. Device pointers only. The caller owns every
 * buffer and the stream. Nothing is allocated or synchronized. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECLAIR_GATED_DELTA_ABI 1u

/* initial_format: 0 for a zero initial state, 1 binary16, 2 FP32. Every
 * *_elements field is the readable or writable element count of its buffer. */
typedef struct EclairGatedDeltaStatesArgs {
    uint32_t abi; /* ECLAIR_GATED_DELTA_ABI */
    uint32_t initial_format;
    const void* keys;
    uint64_t keys_elements;
    const void* w;
    uint64_t w_elements;
    const void* u;
    uint64_t u_elements;
    const float* gates;
    uint64_t gates_elements;
    const void* initial;
    uint64_t initial_elements;
    const int32_t* sequence_starts;
    const int32_t* chunk_starts;
    void* states;
    uint64_t states_elements;
    void* new_values;
    uint64_t new_values_elements;
    float* final_states;
    uint64_t final_states_elements;
    uint64_t tokens;
    uint64_t sequences;
    uint64_t chunks;
    uint64_t heads;
    uint64_t key_heads;
    uint64_t key_size;
    uint64_t value_size;
} EclairGatedDeltaStatesArgs;

/* Enqueues the state pass on `stream` (a cudaStream_t) and returns a
 * cudaError_t value: 0 when enqueued, 1 (cudaErrorInvalidValue) for a wrong ABI
 * number or arguments the kernel does not support. Never throws. */
int eclair_gated_delta_states(const EclairGatedDeltaStatesArgs* args, void* stream);

/* 0 when this library holds a kernel image the device can load, otherwise the
 * cudaError_t of the failed lookup. */
int eclair_gated_delta_image(int device_index);

#ifdef __cplusplus
}
#endif

#endif /* ECLAIR_C_GATED_DELTA_H */
