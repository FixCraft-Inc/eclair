/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/paged_attention.hpp"

#include "kernel_image.hpp"
#include "shared_memory.hpp"

#include <cuda_fp16.h>
#include <mma.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace eclair::cuda {
namespace {

namespace wmma = nvcuda::wmma;
using detail::kMaxSharedBytes;

constexpr int kTile = 16;                                      // WMMA tile edge.
constexpr int kRows = static_cast<int>(kPagedPrefillMinRows);  // Query rows per block.
constexpr int kKeys = 32;  // Keys per loop step: two WMMA tiles.
// Shared K and V rows carry eight extra binary16 values. WMMA needs leading
// dimensions in multiples of eight and 32-byte aligned tiles, and the padding
// spreads the rows of a column fragment over banks.
constexpr int kPad = 8;
// FP32 score rows. A row's probabilities later overwrite its first half as
// binary16, with a leading dimension of 2 * kScoreStride values. Column kKeys
// holds the row's rescale factor and column kKeys + 1 its softmax sum.
constexpr int kScoreStride = kKeys + 4;
constexpr int kAlphaColumn = kKeys;
constexpr int kSumColumn = kKeys + 1;
constexpr int kMaxGroup = static_cast<int>(kPagedPrefillMaxGroup);

using Accumulator = wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>;
using RowOperand = wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::row_major>;
using KeyOperand = wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major>;
using ValueOperand = wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::row_major>;

struct Params {
    const __half* query;
    __half* output;
    const std::int8_t* keys;
    const std::int8_t* values;
    const float* key_scales;
    const float* value_scales;
    const std::int32_t* block_table;
    long long query_row_stride;   // Elements.
    long long output_row_stride;  // Elements.
    long long block_stride;       // Bytes.
    long long slot_stride;        // Bytes.
    long long head_stride;        // Bytes.
    long long scale_block_stride;
    long long scale_slot_stride;
    long long scale_head_stride;
    int blocks;
    int block_size;
    int group;
    int rows;
    int context;
    float scale;
};

// The group's query rows, one K and one V tile, and the group's score rows.
// Head size 256 with six query heads per KV head fills kMaxSharedBytes.
[[nodiscard]] constexpr std::size_t shared_bytes(std::size_t head_size, std::size_t group) {
    return (group * kRows + 2U * kKeys) * (head_size + kPad) * sizeof(__half) +
           group * kRows * kScoreStride * sizeof(float);
}

// Packed words stay in registers while the preceding tile is computed. Each
// warp owns whole tokens, and its lanes load consecutive four-byte words.
template <int HeadSize, int Warps>
struct TileLoad {
    static constexpr int kWords = HeadSize / 128;
    static constexpr int kTokens = (kKeys + Warps - 1) / Warps;
    unsigned int raw[kTokens][2][kWords];
    float scale[kTokens][2];
};

// Lane i holds the metadata of token warp + i * Warps. It is resolved one
// tile before its bytes are fetched, hiding the dependent table/scale loads.
struct TokenCell {
    long long cell = 0;
    float key_scale = 0.0F;
    float value_scale = 0.0F;
    int present = 0;
};

template <int Warps>
__device__ TokenCell resolve_tile(const Params& p, int kv_head, long long first_key, int tokens,
                                  int warp, int lane) {
    constexpr int kTokens = (kKeys + Warps - 1) / Warps;
    TokenCell token;
    const int offset = warp + lane * Warps;
    const long long position = first_key + offset;
    if (lane < kTokens && offset < kKeys && position < tokens) {
        const int entry = static_cast<int>(position / p.block_size);
        const int slot = static_cast<int>(position - entry * p.block_size);
        const int block = p.block_table[entry];
        if (block >= 0 && block < p.blocks) {
            token.present = 1;
            token.cell = block * p.block_stride + slot * p.slot_stride + kv_head * p.head_stride;
            const long long scale_index = block * p.scale_block_stride +
                                          slot * p.scale_slot_stride +
                                          kv_head * p.scale_head_stride;
            token.key_scale = p.key_scales[scale_index];
            token.value_scale = p.value_scales[scale_index];
        }
    }
    return token;
}

template <int HeadSize, int Warps>
__device__ void fetch_tile(const Params& p, const TokenCell& token, int lane,
                           TileLoad<HeadSize, Warps>& tile) {
    using Tile = TileLoad<HeadSize, Warps>;
#pragma unroll
    for (int i = 0; i < Tile::kTokens; ++i) {
        const long long base = __shfl_sync(0xFFFFFFFFU, token.cell, i) + 4 * lane;
        const int here = __shfl_sync(0xFFFFFFFFU, token.present, i);
        tile.scale[i][0] = __shfl_sync(0xFFFFFFFFU, token.key_scale, i);
        tile.scale[i][1] = __shfl_sync(0xFFFFFFFFU, token.value_scale, i);
#pragma unroll
        for (int w = 0; w < Tile::kWords; ++w) {
            tile.raw[i][0][w] = 0U;
            tile.raw[i][1][w] = 0U;
        }
        if (here != 0) {
#pragma unroll
            for (int w = 0; w < Tile::kWords; ++w) {
                tile.raw[i][0][w] = *reinterpret_cast<const unsigned int*>(p.keys + base + 128 * w);
                tile.raw[i][1][w] =
                    *reinterpret_cast<const unsigned int*>(p.values + base + 128 * w);
            }
        }
    }
}

// Exact signed int8 -> FP32 without Volta's quarter-rate conversion pipe.
// The biased bit pattern is 8388608 + (byte xor 128). Keep the subtraction
// separate from the scale product so both FP32 roundings remain as specified.
template <int Byte>
__device__ float exact_byte(unsigned int word) {
    const unsigned int biased = ((word >> (8 * Byte)) ^ 0x80U) & 0xFFU;
    return __fsub_rn(__uint_as_float(0x4B000000U | biased), 8388736.0F);
}

__device__ void store_word(unsigned int word, float scale, __half2* target) {
    target[0] = __floats2half2_rn(__fmul_rn(exact_byte<0>(word), scale),
                                  __fmul_rn(exact_byte<1>(word), scale));
    target[1] = __floats2half2_rn(__fmul_rn(exact_byte<2>(word), scale),
                                  __fmul_rn(exact_byte<3>(word), scale));
}

template <int HeadSize, int Warps>
__device__ void store_tile(const TileLoad<HeadSize, Warps>& tile, int warp, int lane, __half* keys,
                           __half* values) {
    using Tile = TileLoad<HeadSize, Warps>;
#pragma unroll
    for (int i = 0; i < Tile::kTokens; ++i) {
        const int token = warp + i * Warps;
        if (token < kKeys) {
#pragma unroll
            for (int w = 0; w < Tile::kWords; ++w) {
                const int offset = token * (HeadSize + kPad) + 4 * lane + 128 * w;
                store_word(tile.raw[i][0][w], tile.scale[i][0],
                           reinterpret_cast<__half2*>(keys + offset));
                store_word(tile.raw[i][1][w], tile.scale[i][1],
                           reinterpret_cast<__half2*>(values + offset));
            }
        }
    }
}

// Synchronizes the two warps that share one query head. Barrier 0 belongs to
// __syncthreads, so pairs use 1 .. kMaxGroup.
__device__ void pair_sync(int member) {
    asm volatile("bar.sync %0, 64;" ::"r"(member + 1) : "memory");
}

// One block per KV head and 16 query rows. Two warps share each query head of
// the KV head's group: each holds the FP32 output of half of the head size, so
// the accumulators of head size 256 take 64 registers and twelve warps fit an
// SM. Every K and V tile is dequantized once for the whole group. The pair
// splits Q K^T over the head size and adds the halves through shared memory,
// each warp runs the online softmax of eight rows, and both apply the
// probabilities to their half of V. WMMA leaves the element-to-row mapping of
// a fragment unspecified but fixed for its type, so each warp reads it once
// from a fragment loaded with row numbers. Fragments of one type add
// elementwise.
template <int HeadSize, int Group>
__global__ void __launch_bounds__(Group * 64) paged_prefill_kernel(Params p) {
    constexpr int kHalf = HeadSize / 2;
    constexpr int kFragments = kHalf / kTile;
    constexpr int kWarps = Group * 2;
    extern __shared__ __align__(32) unsigned char shared[];
    __half* queries = reinterpret_cast<__half*>(shared);
    __half* keys = queries + p.group * kRows * (HeadSize + kPad);
    __half* values = keys + kKeys * (HeadSize + kPad);
    float* scores = reinterpret_cast<float*>(values + kKeys * (HeadSize + kPad));

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int member = warp / 2;  // Query head within the group.
    const int half = warp % 2;    // Which half of the head size this warp owns.
    const int kv_head = static_cast<int>(blockIdx.y);
    const int head = kv_head * p.group + member;
    // The last tile is shifted back to end at the chunk's last row. Rows it
    // shares with the previous tile are computed again but written once.
    const int nominal_row = static_cast<int>(blockIdx.x) * kRows;
    const int first_row = min(nominal_row, p.rows - kRows);
    const int tokens = p.context + p.rows;
    const int last_key = p.context + first_row + kRows - 1;
    float* head_scores = scores + member * kRows * kScoreStride;
    __half* head_probabilities = reinterpret_cast<__half*>(head_scores);
    __half* query = queries + member * kRows * (HeadSize + kPad);
    {
        constexpr int kVectors = HeadSize / 8;  // 16-byte copies per row.
        const __half* source = p.query + first_row * p.query_row_stride + head * HeadSize;
        for (int item = half * 32 + lane; item < kRows * kVectors; item += 64) {
            const int r = item / kVectors;
            const int v = item - r * kVectors;
            *reinterpret_cast<uint4*>(query + r * (HeadSize + kPad) + 8 * v) =
                *reinterpret_cast<const uint4*>(source + r * p.query_row_stride + 8 * v);
        }
        for (int element = half * 32 + lane; element < kTile * kTile; element += 64) {
            head_scores[(element / kTile) * kScoreStride + element % kTile] =
                static_cast<float>(element / kTile);
        }
    }
    pair_sync(member);
    int row_of[Accumulator::num_elements];
    {
        Accumulator rows;
        wmma::load_matrix_sync(rows, head_scores, kScoreStride, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            row_of[i] = static_cast<int>(rows.x[i]);
        }
    }

    Accumulator output[kFragments];
#pragma unroll
    for (Accumulator& fragment : output) {
        wmma::fill_fragment(fragment, 0.0F);
    }
    // Four lanes share each of the warp's eight rows, eight scores each.
    const int my_row = half * 8 + lane / 4;
    const int my_columns = (lane % 4) * 8;
    const int my_limit = p.context + first_row + my_row;
    float running_max = -INFINITY;
    float running_sum = 0.0F;

    TileLoad<HeadSize, kWarps> tile;
    TokenCell next_cell = resolve_tile<kWarps>(p, kv_head, 0, tokens, warp, lane);
    fetch_tile(p, next_cell, lane, tile);
    next_cell = resolve_tile<kWarps>(p, kv_head, kKeys, tokens, warp, lane);
    for (int tile_index = 0; tile_index <= last_key / kKeys; ++tile_index) {
        const int first_key = tile_index * kKeys;
        __syncthreads();
        store_tile(tile, warp, lane, keys, values);
        __syncthreads();
        if (tile_index < last_key / kKeys) {
            fetch_tile(p, next_cell, lane, tile);
            next_cell = resolve_tile<kWarps>(
                p, kv_head, static_cast<long long>(first_key) + 2 * kKeys, tokens, warp, lane);
        }

        // Two accumulators per key tile, for alternate steps of the head size,
        // halve the chains of dependent MMAs.
        Accumulator tile_scores[2];
        Accumulator odd_scores[2];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            wmma::fill_fragment(tile_scores[n], 0.0F);
            wmma::fill_fragment(odd_scores[n], 0.0F);
        }
#pragma unroll
        for (int k = half * kHalf; k < (half + 1) * kHalf; k += kTile) {
            RowOperand q;
            wmma::load_matrix_sync(q, query + k, HeadSize + kPad);
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                KeyOperand key;
                wmma::load_matrix_sync(key, keys + n * kTile * (HeadSize + kPad) + k,
                                       HeadSize + kPad);
                Accumulator& target = (k / kTile) % 2 == 0 ? tile_scores[n] : odd_scores[n];
                wmma::mma_sync(target, q, key, target);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < Accumulator::num_elements; ++i) {
                tile_scores[n].x[i] += odd_scores[n].x[i];
            }
        }
        if (half == 0) {
            wmma::store_matrix_sync(head_scores, tile_scores[0], kScoreStride, wmma::mem_row_major);
            wmma::store_matrix_sync(head_scores + kTile, tile_scores[1], kScoreStride,
                                    wmma::mem_row_major);
        }
        pair_sync(member);
        if (half == 1) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                Accumulator other;
                wmma::load_matrix_sync(other, head_scores + n * kTile, kScoreStride,
                                       wmma::mem_row_major);
#pragma unroll
                for (int i = 0; i < Accumulator::num_elements; ++i) {
                    tile_scores[n].x[i] += other.x[i];
                }
                wmma::store_matrix_sync(head_scores + n * kTile, tile_scores[n], kScoreStride,
                                        wmma::mem_row_major);
            }
        }
        pair_sync(member);

        float row[8];
        {
            const float4* source =
                reinterpret_cast<const float4*>(head_scores + my_row * kScoreStride + my_columns);
            const float4 low = source[0];
            const float4 high = source[1];
            const float loaded[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                row[c] = first_key + my_columns + c <= my_limit ? loaded[c] * p.scale : -INFINITY;
            }
        }
        float tile_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            tile_max = fmaxf(tile_max, row[c]);
        }
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFU, tile_max, 1));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFU, tile_max, 2));
        // Key 0 is visible to every row, so the maximum is finite from the
        // first step on and alpha is 0 there. fmaxf skips a NaN score, which
        // still turns its probability, the sum and the row into NaN.
        const float new_max = fmaxf(running_max, tile_max);
        const float alpha = __expf(running_max - new_max);
        // A row's four lanes have read their scores before any overwrites the
        // row's first half with probabilities.
        __syncwarp();
        __half2* probabilities = reinterpret_cast<__half2*>(
            head_probabilities + my_row * (2 * kScoreStride) + my_columns);
        float tile_sum = 0.0F;
#pragma unroll
        for (int c = 0; c < 8; c += 2) {
            const __half2 pair =
                __floats2half2_rn(__expf(row[c] - new_max), __expf(row[c + 1] - new_max));
            // Sum what the products will use, so the normalization matches.
            tile_sum += __low2float(pair) + __high2float(pair);
            probabilities[c / 2] = pair;
        }
        tile_sum += __shfl_xor_sync(0xFFFFFFFFU, tile_sum, 1);
        tile_sum += __shfl_xor_sync(0xFFFFFFFFU, tile_sum, 2);
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
        if (lane % 4 == 0) {
            head_scores[my_row * kScoreStride + kAlphaColumn] = alpha;
        }
        pair_sync(member);
        float alpha_of[Accumulator::num_elements];
        bool settled = true;
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            alpha_of[i] = head_scores[row_of[i] * kScoreStride + kAlphaColumn];
            settled = settled && alpha_of[i] == 1.0F;
        }
        // Once every row's maximum has settled, alpha is exactly 1 for the head.
        if (!__all_sync(0xFFFFFFFFU, settled)) {
#pragma unroll
            for (Accumulator& fragment : output) {
#pragma unroll
                for (int i = 0; i < Accumulator::num_elements; ++i) {
                    fragment.x[i] *= alpha_of[i];
                }
            }
        }
        RowOperand weights[2];
        wmma::load_matrix_sync(weights[0], head_probabilities, 2 * kScoreStride);
        wmma::load_matrix_sync(weights[1], head_probabilities + kTile, 2 * kScoreStride);
#pragma unroll
        for (int f = 0; f < kFragments; ++f) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                ValueOperand value;
                wmma::load_matrix_sync(
                    value, values + n * kTile * (HeadSize + kPad) + half * kHalf + f * kTile,
                    HeadSize + kPad);
                wmma::mma_sync(output[f], weights[n], value, output[f]);
            }
        }
    }

    if (lane % 4 == 0) {
        head_scores[my_row * kScoreStride + kSumColumn] = running_sum;
    }
    // Both warps have also finished reading the last probabilities, which the
    // staging below overwrites: half 0 uses columns 0-15, half 1 columns 16-31.
    pair_sync(member);
    float* staging = head_scores + half * kTile;
    const int skip = nominal_row - first_row;
    float inverse_sum[Accumulator::num_elements];
#pragma unroll
    for (int i = 0; i < Accumulator::num_elements; ++i) {
        inverse_sum[i] = 1.0F / head_scores[row_of[i] * kScoreStride + kSumColumn];
    }
#pragma unroll
    for (int f = 0; f < kFragments; ++f) {
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            output[f].x[i] *= inverse_sum[i];
        }
        wmma::store_matrix_sync(staging, output[f], kScoreStride, wmma::mem_row_major);
        __syncwarp();
        for (int element = lane; element < kTile * kTile; element += 32) {
            const int r = element / kTile;
            const int c = element % kTile;
            if (r >= skip) {
                p.output[(first_row + r) * p.output_row_stride + head * HeadSize + half * kHalf +
                         f * kTile + c] = __float2half_rn(staging[r * kScoreStride + c]);
            }
        }
        __syncwarp();
    }
}

// Each quad pair owns an independent 8x8 product. QK splits K across
// quad pairs; PV assigns them adjacent eight-column output tiles.
template <bool RowMajorB = false>
__device__ __forceinline__ void prefill_mma(float (&sum)[8], unsigned int a01, unsigned int a23,
                                            unsigned int b01, unsigned int b23) {
    if constexpr (RowMajorB) {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
            "{%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
            : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3]), "+f"(sum[4]), "+f"(sum[5]),
              "+f"(sum[6]), "+f"(sum[7])
            : "r"(a01), "r"(a23), "r"(b01), "r"(b23));
    } else {
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
            "{%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
            : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3]), "+f"(sum[4]), "+f"(sum[5]),
              "+f"(sum[6]), "+f"(sum[7])
            : "r"(a01), "r"(a23), "r"(b01), "r"(b23));
    }
}

constexpr std::size_t kMmaSharedBytes =
    2U * kKeys * (256U + kPad) * sizeof(__half) + 2U * 6U * 16U * kScoreStride * sizeof(float);

// Six heads, sixteen query rows, two warps per head. Quad pairs split QK
// into eight-row pieces with alternating 16-dimension partials, matching WMMA;
// PV uses eight-row/eight-column pieces.
// Keep 32 cached Q words and 64 output accumulators per lane. With packed
// prefetch and temporaries the sm_70 build must fit 168 registers without
// spills (384 threads share Volta's 65536-register block budget).
__global__ void __launch_bounds__(384) paged_prefill_mma_kernel(Params p) {
    constexpr int kStride = 256 + kPad;
    extern __shared__ __align__(32) unsigned char shared[];
    __half* keys = reinterpret_cast<__half*>(shared);
    __half* values = keys + kKeys * kStride;
    float* scores = reinterpret_cast<float*>(values + kKeys * kStride);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int quad = lane / 4 % 4;
    const int upper = lane / 16;
    const int fragment_row = lane % 4 + 4 * upper;
    const int row_tile = quad % 2;
    const int column_tile = quad / 2;
    const int kv_head = static_cast<int>(blockIdx.y);
    const int half = warp % 2;
    const int head = kv_head * 6 + warp / 2;
    const int nominal_row = static_cast<int>(blockIdx.x) * 16;
    const int first_row = min(nominal_row, p.rows - 16);
    const int last_key = p.context + first_row + 15;
    const int tokens = p.context + p.rows;
    float* head_scores = scores + warp / 2 * 16 * kScoreStride;
    // Each warp writes a separate QK plane. Softmax adds the two halves
    // after the pair barrier, before overwriting plane zero with probabilities.
    float* partial_scores = head_scores + half * 6 * 16 * kScoreStride;
    __half* probabilities = reinterpret_cast<__half*>(head_scores);
    unsigned int query[32];
#pragma unroll
    for (int step = 0; step < 16; ++step) {
        memcpy(query + 2 * step,
               p.query + (first_row + (row_tile * 8 + fragment_row)) * p.query_row_stride +
                   head * 256 + half * 128 + column_tile * 16 + (step / 4) * 32 + (step % 4) * 4,
               2 * sizeof(unsigned int));
    }
    float output[8][8] = {};
    const int my_row = half * 8 + lane / 4;
    const int my_columns = lane % 4 * 8;
    const int my_limit = p.context + first_row + my_row;
    float running_max = -INFINITY;
    float running_sum = 0.0F;
    TileLoad<256, 12> tile;
    TokenCell next = resolve_tile<12>(p, kv_head, 0, tokens, warp, lane);
    fetch_tile(p, next, lane, tile);
    next = resolve_tile<12>(p, kv_head, kKeys, tokens, warp, lane);
    for (int tile_index = 0; tile_index <= last_key / kKeys; ++tile_index) {
        const int first_key = tile_index * kKeys;
        __syncthreads();
        store_tile(tile, warp, lane, keys, values);
        __syncthreads();
        if (tile_index < last_key / kKeys) {
            fetch_tile(p, next, lane, tile);
            next = resolve_tile<12>(p, kv_head, static_cast<long long>(first_key) + 2 * kKeys,
                                    tokens, warp, lane);
        }
#pragma unroll 1
        for (int n = 0; n < 4; ++n) {
            float dots[8] = {};
#pragma unroll
            for (int step = 0; step < 16; ++step) {
                unsigned int key[2];
                memcpy(key,
                       keys + (n * 8 + fragment_row) * kStride + half * 128 + column_tile * 16 +
                           (step / 4) * 32 + (step % 4) * 4,
                       sizeof(key));
                prefill_mma(dots, query[2 * step], query[2 * step + 1], key[0], key[1]);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dots[i] += __shfl_xor_sync(0xFFFFFFFFU, dots[i], 8);
                if (quad < 2) {
                    const int row = row_tile * 8 + (lane & 1) + (i & 2) + 4 * upper;
                    const int col = n * 8 + (i & 4) + (lane & 2) + (i & 1);
                    partial_scores[row * kScoreStride + col] = dots[i];
                }
            }
        }
        pair_sync(warp / 2);
        float row[8];
        float tile_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int index = my_row * kScoreStride + my_columns + c;
            row[c] =
                first_key + my_columns + c <= my_limit
                    ? (head_scores[index] + head_scores[index + 6 * 16 * kScoreStride]) * p.scale
                    : -INFINITY;
            tile_max = fmaxf(tile_max, row[c]);
        }
#pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFU, tile_max, mask));
        }
        const float new_max = fmaxf(running_max, tile_max);
        const float alpha = __expf(running_max - new_max);
        __syncwarp();
        float tile_sum = 0.0F;
#pragma unroll
        for (int c = 0; c < 8; c += 2) {
            const __half2 pair =
                __floats2half2_rn(__expf(row[c] - new_max), __expf(row[c + 1] - new_max));
            tile_sum += __low2float(pair) + __high2float(pair);
            *reinterpret_cast<__half2*>(probabilities + my_row * 2 * kScoreStride + my_columns +
                                        c) = pair;
        }
#pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
            tile_sum += __shfl_xor_sync(0xFFFFFFFFU, tile_sum, mask);
        }
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
        if (lane % 4 == 0) {
            head_scores[my_row * kScoreStride + kAlphaColumn] = alpha;
            head_scores[my_row * kScoreStride + kSumColumn] = running_sum;
        }
        pair_sync(warp / 2);
        bool settled = true;
        float factors[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row_tile * 8 + (lane & 1) + (i & 2) + 4 * upper;
            factors[i] = head_scores[row * kScoreStride + kAlphaColumn];
            settled = settled && factors[i] == 1.0F;
        }
        if (!__all_sync(0xFFFFFFFFU, settled)) {
#pragma unroll
            for (int part = 0; part < 8; ++part) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    output[part][i] *= factors[i];
                }
            }
        }
#pragma unroll 1
        for (int step = 0; step < 8; ++step) {
            unsigned int weights[2];
            memcpy(weights,
                   probabilities + (row_tile * 8 + fragment_row) * 2 * kScoreStride + step * 4,
                   sizeof(weights));
#pragma unroll
            for (int part = 0; part < 8; ++part) {
                unsigned int value[2];
                memcpy(value,
                       values + (step * 4 + lane % 4) * kStride + part * 16 + half * 128 +
                           column_tile * 8 + upper * 4,
                       sizeof(value));
                prefill_mma<true>(output[part], weights[0], weights[1], value[0], value[1]);
            }
        }
        // The next block barrier protects K/V and both score planes.
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = row_tile * 8 + (lane & 1) + (i & 2) + 4 * upper;
        const float sum = head_scores[row * kScoreStride + kSumColumn];
#pragma unroll
        for (int part = 0; part < 8; ++part) {
            const int d = half * 128 + part * 16 + column_tile * 8 + (i & 4) + (lane & 2) + (i & 1);
            if (first_row + row >= nominal_row) {
                p.output[(first_row + row) * p.output_row_stride + head * 256 + d] =
                    __float2half_rn(output[part][i] * (1.0F / sum));
            }
        }
    }
}

[[nodiscard]] bool fits_int(std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(INT_MAX);
}

[[nodiscard]] bool fits_long(std::size_t value) noexcept {
    return value <= static_cast<std::size_t>(LLONG_MAX);
}

[[nodiscard]] bool aligned(const void* pointer, std::uintptr_t bytes) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0U;
}

template <int HeadSize, int Group = 1>
[[nodiscard]] cudaError_t launch(const Params& params, unsigned int row_tiles,
                                 unsigned int kv_heads, cudaStream_t stream) noexcept {
    if (params.group != Group) {
        if constexpr (Group < (HeadSize == 256 ? 6 : kMaxGroup)) {
            return launch<HeadSize, Group + 1>(params, row_tiles, kv_heads, stream);
        }
        return cudaErrorInvalidValue;
    }
    if constexpr (HeadSize == 256 && Group == 6) {
        cudaError_t status = cudaFuncSetAttribute(paged_prefill_mma_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(kMmaSharedBytes));
        if (status != cudaSuccess) {
            return status;
        }
        paged_prefill_mma_kernel<<<dim3(row_tiles, kv_heads), 384, kMmaSharedBytes, stream>>>(
            params);
        return cudaGetLastError();
    }
    const std::size_t bytes = shared_bytes(HeadSize, static_cast<std::size_t>(params.group));
    cudaError_t status =
        cudaFuncSetAttribute(paged_prefill_kernel<HeadSize, Group>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (status == cudaSuccess) {
        status = cudaFuncSetAttribute(paged_prefill_kernel<HeadSize, Group>,
                                      cudaFuncAttributePreferredSharedMemoryCarveout,
                                      cudaSharedmemCarveoutMaxShared);
    }
    if (status != cudaSuccess) {
        return status;
    }
    paged_prefill_kernel<HeadSize, Group>
        <<<dim3(row_tiles, kv_heads), static_cast<unsigned int>(params.group) * 64U, bytes,
           stream>>>(params);
    return cudaGetLastError();
}

}  // namespace

bool paged_prefill_heads_supported(std::size_t heads, std::size_t kv_heads,
                                   std::size_t head_size) noexcept {
    if (kv_heads == 0U || heads % kv_heads != 0U || heads == 0U) {
        return false;
    }
    const std::size_t group = heads / kv_heads;
    return (head_size == 128U || head_size == 256U) && group <= kPagedPrefillMaxGroup &&
           shared_bytes(head_size, group) <= kMaxSharedBytes && kv_heads <= 65535U;
}

bool paged_prefill_supported(const ops::PagedPrefillShape& shape,
                             const ops::PagedInt8CacheLayout& layout,
                             const ops::PagedPrefillBuffers& buffers) noexcept {
    if (ops::validate_paged_prefill(shape, layout, buffers) != ops::Status::Success ||
        shape.rows < kPagedPrefillMinRows) {
        return false;
    }
    const std::size_t tokens = shape.context + shape.rows;  // Validation ruled out overflow.
    return paged_prefill_heads_supported(shape.heads, layout.kv_heads, layout.head_size) &&
           fits_int(tokens) && fits_int(layout.blocks) && fits_int(layout.block_size) &&
           fits_long(shape.query_row_stride) && fits_long(shape.output_row_stride) &&
           fits_long(layout.block_stride) && fits_long(layout.slot_stride) &&
           fits_long(layout.head_stride) && fits_long(layout.scale_block_stride) &&
           fits_long(layout.scale_slot_stride) && fits_long(layout.scale_head_stride) &&
           aligned(buffers.query.data, 32U) && shape.query_row_stride % 16U == 0U &&
           shape.query_row_stride <= UINT_MAX && aligned(buffers.keys.data, 4U) &&
           aligned(buffers.values.data, 4U) && layout.block_stride % 4U == 0U &&
           layout.slot_stride % 4U == 0U && layout.head_stride % 4U == 0U;
}

cudaError_t launch_paged_prefill(const ops::PagedPrefillShape& shape,
                                 const ops::PagedInt8CacheLayout& layout,
                                 const ops::PagedPrefillBuffers& buffers,
                                 cudaStream_t stream) noexcept {
    if (shape.rows == 0U &&
        ops::validate_paged_prefill(shape, layout, buffers) == ops::Status::Success) {
        return cudaSuccess;
    }
    if (!paged_prefill_supported(shape, layout, buffers)) {
        return cudaErrorInvalidValue;
    }
    const Params params{
        reinterpret_cast<const __half*>(buffers.query.data),
        reinterpret_cast<__half*>(buffers.output.data),
        buffers.keys.data,
        buffers.values.data,
        buffers.key_scales.data,
        buffers.value_scales.data,
        buffers.block_table.data,
        static_cast<long long>(shape.query_row_stride),
        static_cast<long long>(shape.output_row_stride),
        static_cast<long long>(layout.block_stride),
        static_cast<long long>(layout.slot_stride),
        static_cast<long long>(layout.head_stride),
        static_cast<long long>(layout.scale_block_stride),
        static_cast<long long>(layout.scale_slot_stride),
        static_cast<long long>(layout.scale_head_stride),
        static_cast<int>(layout.blocks),
        static_cast<int>(layout.block_size),
        static_cast<int>(shape.heads / layout.kv_heads),
        static_cast<int>(shape.rows),
        static_cast<int>(shape.context),
        shape.scale,
    };
    const auto row_tiles = static_cast<unsigned int>((shape.rows + kRows - 1U) / kRows);
    const auto kv_heads = static_cast<unsigned int>(layout.kv_heads);
    return layout.head_size == 256U ? launch<256>(params, row_tiles, kv_heads, stream)
                                    : launch<128>(params, row_tiles, kv_heads, stream);
}

cudaError_t check_paged_prefill_image(int device_index) noexcept {
    return detail::check_kernel_images(
        device_index, paged_prefill_mma_kernel, paged_prefill_kernel<128, 1>,
        paged_prefill_kernel<128, 2>, paged_prefill_kernel<128, 3>, paged_prefill_kernel<128, 4>,
        paged_prefill_kernel<128, 5>, paged_prefill_kernel<128, 6>, paged_prefill_kernel<128, 7>,
        paged_prefill_kernel<128, 8>, paged_prefill_kernel<256, 1>, paged_prefill_kernel<256, 2>,
        paged_prefill_kernel<256, 3>, paged_prefill_kernel<256, 4>, paged_prefill_kernel<256, 5>,
        paged_prefill_kernel<256, 6>);
}

}  // namespace eclair::cuda
