/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/gated_delta.hpp"

#include "kernel_image.hpp"

#include <cuda_fp16.h>
#include <mma.h>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace eclair::cuda {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kTile = 16;  // WMMA tile edge.
constexpr int kChunk = static_cast<int>(ops::kGatedDeltaChunk);
constexpr int kValueBlock = 64;  // State rows per thread block.
constexpr int kWarps = 8;
// Shared rows carry eight extra binary16 values: WMMA needs leading dimensions
// in multiples of eight and 32-byte aligned tiles, and the padding spreads
// rows over banks.
constexpr int kPad = 8;
constexpr int kValueStride = kValueBlock + kPad;

using Accumulator = wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>;
using RowA = wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major>;
using ColumnA = wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::col_major>;
using RowB = wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::row_major>;
using ColumnB = wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major>;

struct Params {
    const __half* keys;
    const __half* w;
    const __half* u;
    const float* gates;
    const __half* initial_binary16;
    const float* initial_float32;
    const std::int32_t* sequence_starts;
    const std::int32_t* chunk_starts;
    __half* states;
    __half* new_values;
    float* final_states;
    int tokens;
    int chunks;
    int heads;
    int key_heads;
    int value_size;
};

// The binary16 state block, the chunk's w and k rows, and its u (then d) and
// gated e rows, followed by the chunk's FP32 gates.
[[nodiscard]] constexpr std::size_t shared_bytes(int key_size) {
    return static_cast<std::size_t>(kValueBlock * (key_size + kPad) +
                                    2 * kChunk * (key_size + kPad) + 2 * kChunk * kValueStride) *
               sizeof(__half) +
           kChunk * sizeof(float);
}

// Copies `rows` rows of `width` binary16 values with 16-byte accesses. Source
// rows at or past `valid` become zeros. Every thread of the block takes part.
__device__ void load_rows(__half* target, int target_stride, const __half* source,
                          long long source_stride, int rows, int valid, int width) {
    const int vectors = width / 8;
    for (int item = static_cast<int>(threadIdx.x); item < rows * vectors;
         item += static_cast<int>(blockDim.x)) {
        const int r = item / vectors;
        const int c = (item - r * vectors) * 8;
        uint4 value = make_uint4(0U, 0U, 0U, 0U);
        if (r < valid) {
            value = *reinterpret_cast<const uint4*>(source + r * source_stride + c);
        }
        *reinterpret_cast<uint4*>(target + r * target_stride + c) = value;
    }
}

__device__ void store_rows(__half* target, long long target_stride, const __half* source,
                           int source_stride, int rows, int width) {
    const int vectors = width / 8;
    for (int item = static_cast<int>(threadIdx.x); item < rows * vectors;
         item += static_cast<int>(blockDim.x)) {
        const int r = item / vectors;
        const int c = (item - r * vectors) * 8;
        *reinterpret_cast<uint4*>(target + r * target_stride + c) =
            *reinterpret_cast<const uint4*>(source + r * source_stride + c);
    }
}

// One block per sequence, head and 64 state rows. The FP32 state stays in WMMA
// accumulators for the whole sequence: warp w holds rows 16 (w / 2) onward and
// half of the key columns. Per chunk, d = u - w S16^T and the update e^T k run
// on tensor cores with FP32 accumulation, and gating, rounding and the stores
// of states and new values follow the contract. WMMA leaves the element-to-
// position mapping of a fragment unspecified but fixed for its type, so each
// warp reads it once from fragments loaded with row and column numbers.
template <int KeySize>
__global__ void __launch_bounds__(kWarps * 32) gated_delta_states_kernel(Params p) {
    constexpr int kKeyStride = KeySize + kPad;
    constexpr int kKeyTiles = KeySize / kTile;
    constexpr int kOwnKeyTiles = kKeyTiles / 2;
    extern __shared__ __align__(32) unsigned char shared[];
    __half* state16 = reinterpret_cast<__half*>(shared);
    __half* w_rows = state16 + kValueBlock * kKeyStride;
    __half* k_rows = w_rows + kChunk * kKeyStride;
    __half* u_rows = k_rows + kChunk * kKeyStride;
    __half* e_rows = u_rows + kChunk * kValueStride;
    float* gates = reinterpret_cast<float*>(e_rows + kChunk * kValueStride);

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int value_base = static_cast<int>(blockIdx.x) * kValueBlock;
    const int n = static_cast<int>(blockIdx.y) / p.heads;
    const int h = static_cast<int>(blockIdx.y) % p.heads;
    const int key_head = h / (p.heads / p.key_heads);
    const int first = p.sequence_starts[n];
    const int last = p.sequence_starts[n + 1];
    const int first_chunk = p.chunk_starts[n];
    // The same answer for the whole block, so returning skips no barrier.
    if (first < 0 || last < first || last > p.tokens || first_chunk < 0 ||
        first_chunk + (last - first + kChunk - 1) / kChunk > p.chunks) {
        return;
    }

    // The w and k rows are free until the first chunk, so they hold each
    // warp's two index matrices.
    float* scratch = reinterpret_cast<float*>(w_rows) + warp * 2 * kTile * kTile;
    for (int element = lane; element < kTile * kTile; element += 32) {
        scratch[element] = static_cast<float>(element / kTile);
        scratch[kTile * kTile + element] = static_cast<float>(element % kTile);
    }
    __syncwarp();
    int row_of[Accumulator::num_elements];
    int column_of[Accumulator::num_elements];
    {
        Accumulator rows;
        Accumulator columns;
        wmma::load_matrix_sync(rows, scratch, kTile, wmma::mem_row_major);
        wmma::load_matrix_sync(columns, scratch + kTile * kTile, kTile, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            row_of[i] = static_cast<int>(rows.x[i]);
            column_of[i] = static_cast<int>(columns.x[i]);
        }
    }

    const int value_tile = warp / 2;
    const int key_tile0 = (warp % 2) * kOwnKeyTiles;
    const long long matrix = static_cast<long long>(p.value_size) * KeySize;
    const long long state_offset = (static_cast<long long>(n) * p.heads + h) * matrix;
    Accumulator state[kOwnKeyTiles];
#pragma unroll
    for (int j = 0; j < kOwnKeyTiles; ++j) {
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            const long long index =
                state_offset +
                static_cast<long long>(value_base + value_tile * kTile + row_of[i]) * KeySize +
                (key_tile0 + j) * kTile + column_of[i];
            state[j].x[i] = p.initial_float32 != nullptr ? p.initial_float32[index]
                            : p.initial_binary16 != nullptr
                                ? __half2float(p.initial_binary16[index])
                                : 0.0F;
        }
    }
    __syncthreads();  // Every warp has read its index matrices.

    int chunk = first_chunk;
    for (int a = first; a < last; a += kChunk, ++chunk) {
        const int valid = min(kChunk, last - a);
#pragma unroll
        for (int j = 0; j < kOwnKeyTiles; ++j) {
#pragma unroll
            for (int i = 0; i < Accumulator::num_elements; ++i) {
                state16[(value_tile * kTile + row_of[i]) * kKeyStride + (key_tile0 + j) * kTile +
                        column_of[i]] = __float2half_rn(state[j].x[i]);
            }
        }
        load_rows(w_rows, kKeyStride, p.w + (static_cast<long long>(a) * p.heads + h) * KeySize,
                  static_cast<long long>(p.heads) * KeySize, kChunk, valid, KeySize);
        load_rows(k_rows, kKeyStride,
                  p.keys + (static_cast<long long>(a) * p.key_heads + key_head) * KeySize,
                  static_cast<long long>(p.key_heads) * KeySize, kChunk, valid, KeySize);
        load_rows(u_rows, kValueStride,
                  p.u + (static_cast<long long>(a) * p.heads + h) * p.value_size + value_base,
                  static_cast<long long>(p.heads) * p.value_size, kChunk, valid, kValueBlock);
        for (int t = static_cast<int>(threadIdx.x); t < kChunk; t += static_cast<int>(blockDim.x)) {
            gates[t] = t < valid ? p.gates[static_cast<long long>(a + t) * p.heads + h] : 0.0F;
        }
        __syncthreads();
        const float g_last = gates[valid - 1];

        store_rows(p.states + (static_cast<long long>(chunk) * p.heads + h) * matrix +
                       static_cast<long long>(value_base) * KeySize,
                   KeySize, state16, kKeyStride, kValueBlock, KeySize);

        // Two of the chunk's 4 x 4 tiles of d per warp.
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int tile = warp * 2 + q;
            const int token_tile = tile / 4;
            const int column_tile = tile % 4;
            Accumulator dot;
            wmma::fill_fragment(dot, 0.0F);
#pragma unroll
            for (int kk = 0; kk < kKeyTiles; ++kk) {
                RowA rows;
                ColumnB state_columns;
                wmma::load_matrix_sync(rows, w_rows + token_tile * kTile * kKeyStride + kk * kTile,
                                       kKeyStride);
                wmma::load_matrix_sync(state_columns,
                                       state16 + column_tile * kTile * kKeyStride + kk * kTile,
                                       kKeyStride);
                wmma::mma_sync(dot, rows, state_columns, dot);
            }
#pragma unroll
            for (int i = 0; i < Accumulator::num_elements; ++i) {
                const int t = token_tile * kTile + row_of[i];
                const int v = column_tile * kTile + column_of[i];
                const float d = __half2float(u_rows[t * kValueStride + v]) - dot.x[i];
                const float e = t < valid ? d * __expf(g_last - gates[t]) : 0.0F;
                u_rows[t * kValueStride + v] = __float2half_rn(d);
                e_rows[t * kValueStride + v] = __float2half_rn(e);
            }
        }
        __syncthreads();

        store_rows(
            p.new_values + (static_cast<long long>(a) * p.heads + h) * p.value_size + value_base,
            static_cast<long long>(p.heads) * p.value_size, u_rows, kValueStride, valid,
            kValueBlock);
        const float decay = __expf(g_last);
#pragma unroll
        for (int j = 0; j < kOwnKeyTiles; ++j) {
#pragma unroll
            for (int i = 0; i < Accumulator::num_elements; ++i) {
                state[j].x[i] *= decay;
            }
        }
#pragma unroll
        for (int tt = 0; tt < kChunk / kTile; ++tt) {
            // Element (v, t) of e^T is e_rows[t][v], a column-major tile.
            ColumnA weights;
            wmma::load_matrix_sync(weights, e_rows + tt * kTile * kValueStride + value_tile * kTile,
                                   kValueStride);
#pragma unroll
            for (int j = 0; j < kOwnKeyTiles; ++j) {
                RowB key_rows;
                wmma::load_matrix_sync(key_rows,
                                       k_rows + tt * kTile * kKeyStride + (key_tile0 + j) * kTile,
                                       kKeyStride);
                wmma::mma_sync(state[j], weights, key_rows, state[j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < kOwnKeyTiles; ++j) {
        wmma::store_matrix_sync(
            p.final_states + state_offset +
                static_cast<long long>(value_base + value_tile * kTile) * KeySize +
                (key_tile0 + j) * kTile,
            state[j], KeySize, wmma::mem_row_major);
    }
}

[[nodiscard]] bool fits_int(std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(INT_MAX);
}

[[nodiscard]] bool aligned(const void* pointer, std::uintptr_t bytes) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0U;
}

template <int KeySize>
[[nodiscard]] cudaError_t launch(const Params& params, unsigned int value_blocks, unsigned int rows,
                                 cudaStream_t stream) noexcept {
    const std::size_t bytes = shared_bytes(KeySize);
    const cudaError_t status =
        cudaFuncSetAttribute(gated_delta_states_kernel<KeySize>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        return status;
    }
    gated_delta_states_kernel<KeySize>
        <<<dim3(value_blocks, rows), kWarps * 32, bytes, stream>>>(params);
    return cudaGetLastError();
}

}  // namespace

bool gated_delta_states_supported(const ops::GatedDeltaStatesShape& shape,
                                  const ops::GatedDeltaStatesBuffers& buffers) noexcept {
    if (ops::validate_gated_delta_states(shape, buffers) != ops::Status::Success ||
        shape.tokens == 0U) {
        return false;
    }
    return (shape.key_size == 64U || shape.key_size == 128U) && shape.value_size % 64U == 0U &&
           fits_int(shape.tokens) && fits_int(shape.chunks) && fits_int(shape.value_size) &&
           shape.sequences <= 65535U / shape.heads && aligned(buffers.keys.data, 16U) &&
           aligned(buffers.w.data, 16U) && aligned(buffers.u.data, 16U) &&
           aligned(buffers.states.data, 16U) && aligned(buffers.new_values.data, 16U) &&
           aligned(buffers.final_states.data, 32U);
}

cudaError_t launch_gated_delta_states(const ops::GatedDeltaStatesShape& shape,
                                      const ops::GatedDeltaStatesBuffers& buffers,
                                      cudaStream_t stream) noexcept {
    if (shape.tokens == 0U &&
        ops::validate_gated_delta_states(shape, buffers) == ops::Status::Success) {
        return cudaSuccess;
    }
    if (!gated_delta_states_supported(shape, buffers)) {
        return cudaErrorInvalidValue;
    }
    const Params params{
        reinterpret_cast<const __half*>(buffers.keys.data),
        reinterpret_cast<const __half*>(buffers.w.data),
        reinterpret_cast<const __half*>(buffers.u.data),
        buffers.gates.data,
        reinterpret_cast<const __half*>(buffers.initial_binary16.data),
        buffers.initial_float32.data,
        buffers.sequence_starts.data,
        buffers.chunk_starts.data,
        reinterpret_cast<__half*>(buffers.states.data),
        reinterpret_cast<__half*>(buffers.new_values.data),
        buffers.final_states.data,
        static_cast<int>(shape.tokens),
        static_cast<int>(shape.chunks),
        static_cast<int>(shape.heads),
        static_cast<int>(shape.key_heads),
        static_cast<int>(shape.value_size),
    };
    const auto value_blocks = static_cast<unsigned int>(shape.value_size / kValueBlock);
    const auto rows = static_cast<unsigned int>(shape.sequences * shape.heads);
    return shape.key_size == 128U ? launch<128>(params, value_blocks, rows, stream)
                                  : launch<64>(params, value_blocks, rows, stream);
}

cudaError_t check_gated_delta_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index, gated_delta_states_kernel<64>,
                                       gated_delta_states_kernel<128>);
}

}  // namespace eclair::cuda
