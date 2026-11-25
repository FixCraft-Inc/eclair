/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#ifndef ECLAIR_C_PAGED_ATTENTION_H
#define ECLAIR_C_PAGED_ATTENTION_H

/* C interface of libeclair_kernels' paged prefill and decode attention, for loaders such as
 * Python's ctypes that should not depend on a C++ or framework ABI. It wraps
 * eclair::cuda::launch_paged_prefill, whose contract is in
 * eclair/ops/paged_attention.hpp. The library links the CUDA runtime
 * dynamically, so a process that already loaded libcudart.so.12 (for
 * example through PyTorch) shares it. Device pointers only. The caller owns
 * every buffer and the stream. Nothing is allocated or synchronized. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ECLAIR_PAGED_PREFILL_ABI 1u

/* Sizes are element counts unless named in bytes. The query and output rows
 * hold heads * head_size binary16 values. Keys and values are int8 cells and
 * share byte strides. Scales are FP32 and share element strides. */
typedef struct EclairPagedPrefillArgs {
    uint32_t abi; /* ECLAIR_PAGED_PREFILL_ABI */
    const void* query;
    uint64_t query_elements;
    uint64_t query_row_stride;
    void* output;
    uint64_t output_elements;
    uint64_t output_row_stride;
    const void* keys;
    const void* values;
    uint64_t cache_bytes; /* Readable bytes from each of keys and values. */
    const float* key_scales;
    const float* value_scales;
    uint64_t scale_elements; /* Readable elements from each scale base. */
    const int32_t* block_table;
    uint64_t block_table_entries;
    uint64_t blocks;
    uint64_t block_size;
    uint64_t kv_heads;
    uint64_t head_size;
    uint64_t block_stride;
    uint64_t slot_stride;
    uint64_t head_stride;
    uint64_t scale_block_stride;
    uint64_t scale_slot_stride;
    uint64_t scale_head_stride;
    uint64_t heads;
    uint64_t rows;
    uint64_t context;
    float scale;
} EclairPagedPrefillArgs;

/* Enqueues the attention on `stream` (a cudaStream_t, NULL for the legacy
 * default stream) and returns a cudaError_t value: 0 when enqueued, 1
 * (cudaErrorInvalidValue) for a wrong ABI number or arguments the kernel does
 * not support. Never throws. */
int eclair_paged_prefill_int8(const EclairPagedPrefillArgs* args, void* stream);

/* 0 when this library holds a kernel image the device can load, otherwise the
 * cudaError_t of the failed lookup. */
int eclair_paged_prefill_image(int device_index);

/* 1 when the kernel implements this head configuration, otherwise 0. Calls with
 * a supported configuration can still fail for their layout or sizes. */
int eclair_paged_prefill_heads_supported(uint64_t heads, uint64_t kv_heads, uint64_t head_size);

#define ECLAIR_PAGED_DECODE_ABI 1u

/* The batched decode of eclair::cuda::launch_paged_decode. Cache fields as in
 * EclairPagedPrefillArgs. Row n of block_tables starts block_table_stride
 * entries after row n - 1 and has max_blocks_per_sequence readable entries.
 * query_starts holds sequences + 1 and seq_lens sequences int32 values, read
 * only on the device. The workspace holds at least
 * eclair_paged_decode_workspace_elements(rows, heads, head_size, splits) FP32
 * values and needs no initialization. */
typedef struct EclairPagedDecodeArgs {
    uint32_t abi; /* ECLAIR_PAGED_DECODE_ABI */
    const void* query;
    uint64_t query_elements;
    uint64_t query_row_stride;
    void* output;
    uint64_t output_elements;
    uint64_t output_row_stride;
    const void* keys;
    const void* values;
    uint64_t cache_bytes;
    const float* key_scales;
    const float* value_scales;
    uint64_t scale_elements;
    const int32_t* block_tables;
    uint64_t block_table_entries;
    uint64_t block_table_stride;
    uint64_t max_blocks_per_sequence;
    const int32_t* query_starts;
    const int32_t* seq_lens;
    float* workspace;
    uint64_t workspace_elements;
    uint64_t blocks;
    uint64_t block_size;
    uint64_t kv_heads;
    uint64_t head_size;
    uint64_t block_stride;
    uint64_t slot_stride;
    uint64_t head_stride;
    uint64_t scale_block_stride;
    uint64_t scale_slot_stride;
    uint64_t scale_head_stride;
    uint64_t heads;
    uint64_t sequences;
    uint64_t rows;
    uint64_t max_rows_per_sequence;
    uint64_t splits;
    float scale;
} EclairPagedDecodeArgs;

/* Enqueues the decode on `stream` and returns a cudaError_t value, as
 * eclair_paged_prefill_int8 does. Never throws, allocates or synchronizes, so
 * a CUDA graph capture of `stream` may record it. */
int eclair_paged_decode_int8(const EclairPagedDecodeArgs* args, void* stream);

/* 0 when this library holds the decode kernel images the device can load. */
int eclair_paged_decode_image(int device_index);

/* 1 when the kernels implement this head configuration for chunks of up to
 * max_rows_per_sequence query rows, otherwise 0. */
int eclair_paged_decode_heads_supported(uint64_t heads, uint64_t kv_heads, uint64_t head_size,
                                        uint64_t max_rows_per_sequence);

/* FP32 workspace values a call needs, or 0 when the count overflows. */
uint64_t eclair_paged_decode_workspace_elements(uint64_t rows, uint64_t heads, uint64_t head_size,
                                                uint64_t splits);

#ifdef __cplusplus
}
#endif

#endif /* ECLAIR_C_PAGED_ATTENTION_H */
