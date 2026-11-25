/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#ifndef ECLAIR_C_GATED_DELTA_OUTPUT_H
#define ECLAIR_C_GATED_DELTA_OUTPUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECLAIR_GATED_DELTA_OUTPUT_ABI 1u

/* Device pointers, contiguous arrays described in eclair/ops/gated_delta_output.hpp.
 * Caller owns buffers and stream. No allocation or synchronization. */
typedef struct EclairGatedDeltaOutputArgs {
    uint32_t abi;
    const void* query;
    uint64_t query_elements;
    const void* keys;
    uint64_t keys_elements;
    const void* values;
    uint64_t values_elements;
    const void* states;
    uint64_t states_elements;
    const float* gates;
    uint64_t gates_elements;
    const int32_t* sequence_starts;
    uint64_t sequence_starts_elements;
    const int32_t* chunk_indices;
    uint64_t chunk_indices_elements;
    void* output;
    uint64_t output_elements;
    uint64_t tokens;
    uint64_t sequences;
    uint64_t chunks;
    uint64_t heads;
    uint64_t key_heads;
    uint64_t key_size;
    uint64_t value_size;
    float scale;
} EclairGatedDeltaOutputArgs;

/* Returns cudaError_t (zero for success). Never throws or synchronizes. */
int eclair_gated_delta_output(const EclairGatedDeltaOutputArgs* args, void* stream);
int eclair_gated_delta_output_image(int device_index);

#ifdef __cplusplus
}
#endif

#endif /* ECLAIR_C_GATED_DELTA_OUTPUT_H */
