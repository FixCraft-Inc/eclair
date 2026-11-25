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

constexpr int kKeys = 32;  // Keys per tile.
constexpr int kKeySteps = kKeys / 16;
// Shared K, V and query rows carry eight extra binary16 values: WMMA needs
// leading dimensions in multiples of eight, and the padding spreads the rows
// of a column fragment over banks.
constexpr int kPad = 8;
// FP32 score rows. With 32 rows, two rows share a warp's softmax lanes, and
// a stride of 16 banks keeps their reads apart. Otherwise a warp reads one row.
template <int M>
constexpr int kScoreStrideFor = M == 32 ? kKeys + 16 : kKeys + 4;
constexpr int kProbabilityStride = kKeys + 8;  // binary16 probability rows.
constexpr unsigned int kFullMask = 0xFFFFFFFFU;

struct Params {
    const __half* query;
    __half* output;
    const std::int8_t* keys;
    const std::int8_t* values;
    const float* key_scales;
    const float* value_scales;
    const std::int32_t* block_tables;
    const std::int32_t* query_starts;
    const std::int32_t* seq_lens;
    float* maxima;                // [rows][heads][splits]
    float* sums;                  // [rows][heads][splits]
    float* partials;              // [rows][heads][splits][head_size], unnormalized
    long long query_row_stride;   // Elements.
    long long output_row_stride;  // Elements.
    long long block_table_stride;
    long long block_stride;  // Bytes.
    long long slot_stride;   // Bytes.
    long long head_stride;   // Bytes.
    long long scale_block_stride;
    long long scale_slot_stride;
    long long scale_head_stride;
    int blocks;
    int block_size;
    int heads;
    int group;
    int rows;
    int max_rows;
    int max_tokens;  // Readable table entries times block_size, at most INT_MAX.
    int splits;
    float scale;
};

// A sequence's chunk and whether its metadata follows the contract's rules.
struct Chunk {
    int first_row = 0;
    int rows = 0;
    int tokens = 0;
    bool valid = false;
};

__device__ Chunk read_chunk(const Params& p, int sequence) {
    Chunk chunk;
    const int first = p.query_starts[sequence];
    const int last = p.query_starts[sequence + 1];
    const int tokens = p.seq_lens[sequence];
    chunk.valid = first >= 0 && last >= first && last <= p.rows && last - first <= p.max_rows &&
                  last - first <= tokens && tokens <= p.max_tokens;
    chunk.first_row = first;
    chunk.rows = last - first;
    chunk.tokens = tokens;
    return chunk;
}

// Keys per split, whole tiles, so only the last used split is short. The
// partial and merge kernels derive the same value.
__device__ int split_length(int tokens, int splits) {
    const long long per_split = (static_cast<long long>(tokens) + splits - 1) / splits;
    return static_cast<int>((per_split + kKeys - 1) / kKeys * kKeys);
}

// The maximum that keeps NaN, so a row whose visible scores are all NaN is
// not mistaken for one that sees no key.
__device__ float nan_max(float a, float b) {
    return isnan(a) || isnan(b) ? NAN : fmaxf(a, b);
}

// Warps per block. Rows of 16 or 32 query heads need more registers and
// shared memory than two blocks per SM allow, so their blocks have 16 warps
// to keep as many warps resident. A block of 8 rows has 8 warps and two fit.
template <int M>
constexpr int kWarpsFor = M == 8 ? 8 : 16;

// One tile's K and V bytes and scales, held in registers while the previous
// tile is computed. Warp w owns tokens w, w + Warps, ...; word j of lane l is
// bytes 128 j + 4 l of a cell, so each load reads 128 contiguous bytes. V
// cells are only 4-byte aligned in vLLM's layout. The words stay packed until
// they are converted: unpacking bytes at the load would wait for it there.
template <int HeadSize, int Warps>
struct TileLoad {
    static constexpr int kWords = HeadSize / 128;
    static constexpr int kTokens = kKeys / Warps;
    unsigned int raw[kTokens][2][kWords];
    float scale[kTokens][2];
};

// The cell and scales of one of a warp's tile tokens, held by lane i <
// kTokens for the warp's i-th token. They are resolved a tile before their
// bytes are loaded, so the dependent table and scale loads have a whole tile
// of computation to arrive.
struct TokenCell {
    long long cell = 0;
    float key_scale = 0.0F;
    float value_scale = 0.0F;
    int present = 0;
};

// Positions at or past `end`, and blocks outside [0, blocks), stay absent:
// their bytes load as zeros with zero scales, so they never reach a product
// as anything but zero.
template <int Warps>
__device__ TokenCell resolve_tile(const Params& p, const std::int32_t* table, int kv_head,
                                  int first_key, int end, int warp, int lane) {
    constexpr int kTokens = kKeys / Warps;
    TokenCell token;
    const int position = first_key + warp + (lane % kTokens) * Warps;
    if (lane < kTokens && position < end) {
        const int entry = position / p.block_size;
        const int slot = position - entry * p.block_size;
        const int block = table[entry];
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
        const long long base = __shfl_sync(kFullMask, token.cell, i) + 4 * lane;
        const int here = __shfl_sync(kFullMask, token.present, i);
        tile.scale[i][0] = __shfl_sync(kFullMask, token.key_scale, i);
        tile.scale[i][1] = __shfl_sync(kFullMask, token.value_scale, i);
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

// Four int8 values, lowest address first, as exact binary16 pairs without
// the conversion pipe: 0x6400 | (b ^ 0x80) is the binary16 value 1152 + b.
__device__ void exact_halves(unsigned int word, __half2& low, __half2& high) {
    word ^= 0x80808080U;
    const unsigned int low_bits = __byte_perm(word, 0x64646464U, 0x4140U);
    const unsigned int high_bits = __byte_perm(word, 0x64646464U, 0x4342U);
    const __half2 offset = __float2half2_rn(1152.0F);
    __half2 pair;
    memcpy(&pair, &low_bits, sizeof(pair));
    low = __hsub2(pair, offset);
    memcpy(&pair, &high_bits, sizeof(pair));
    high = __hsub2(pair, offset);
}

// Stores a fetched tile in shared binary16 rows. Keys stay exact integers
// and their scales go to `key_scales`, to multiply the FP32 scores. Values
// are binary16(int8 * binary16(scale)), two roundings where the contract has
// one. Both stay within the contract's tolerance.
template <int HeadSize, int Warps>
__device__ void store_tile(const TileLoad<HeadSize, Warps>& tile, int warp, int lane, __half* keys,
                           __half* values, float* key_scales) {
    using Tile = TileLoad<HeadSize, Warps>;
    constexpr int kStride = HeadSize + kPad;
#pragma unroll
    for (int i = 0; i < Tile::kTokens; ++i) {
        const int token = warp + i * Warps;
        if (lane == 0) {
            key_scales[token] = tile.scale[i][0];
        }
        const __half2 value_scale = __float2half2_rn(tile.scale[i][1]);
#pragma unroll
        for (int w = 0; w < Tile::kWords; ++w) {
            __half2 low;
            __half2 high;
            __half2* key = reinterpret_cast<__half2*>(keys + token * kStride + 4 * lane + 128 * w);
            exact_halves(tile.raw[i][0][w], low, high);
            key[0] = low;
            key[1] = high;
            __half2* value =
                reinterpret_cast<__half2*>(values + token * kStride + 4 * lane + 128 * w);
            exact_halves(tile.raw[i][1][w], low, high);
            value[0] = __hmul2(low, value_scale);
            value[1] = __hmul2(high, value_scale);
        }
    }
}

// Query rows (M of them), one K and one V tile, the score partials (8 rows
// per warp), the probabilities, one rescale factor per row and the tile's key
// scales. The K and V tiles later stage the FP32 outputs.
template <int HeadSize, int M>
constexpr std::size_t kSharedBytes =
    static_cast<std::size_t>((M + 2 * kKeys) * (HeadSize + kPad)) * sizeof(__half) +
    static_cast<std::size_t>(8 * kWarpsFor<M> * kScoreStrideFor<M>) * sizeof(float) +
    static_cast<std::size_t>(M * kProbabilityStride) * sizeof(__half) +
    static_cast<std::size_t>(M + kKeys) * sizeof(float);

// One block per context split, KV head and sequence. Row m = r * group + h
// of the block's M rows is query row r of the chunk at query head
// kv_head * group + h, so every K and V tile is loaded and dequantized once
// for all rows and heads that read it. The next tile's bytes are fetched into
// registers while the current one is computed.
//
// Q K^T: the M x 32 scores of a tile are kScoreFragments WMMA tiles of M x
// 256 / M; the warps split them and the head size, and the softmax adds the
// parts. Each row's online softmax runs on Warps * 32 / M lanes in FP32.
// Probabilities are rounded to binary16 and also summed that way. P V: the
// output accumulators are tiles of at most 16 rows, because tiles of 8 value
// columns are not 32-byte aligned; each warp owns whole tiles, or half of a
// tile's keys when there are fewer tiles than warps. Element-to-row positions
// of a WMMA accumulator are unspecified but fixed for its type, so they are
// read once from a fragment loaded with row numbers.
//
// The split writes, per row and head, its maximum, its sum and its
// unnormalized FP32 output. Rows that see no key of the split write -inf and
// zeros, which the merge skips.
template <int HeadSize, int M>
__global__ void __launch_bounds__(kWarpsFor<M> * 32, M == 8 ? 2 : 1) paged_decode_kernel(Params p) {
    constexpr int kWarps = kWarpsFor<M>;
    constexpr int kThreads = kWarps * 32;
    constexpr int kStride = HeadSize + kPad;
    constexpr int kScoreStride = kScoreStrideFor<M>;
    constexpr int kScoreColumns = 256 / M;
    constexpr int kScoreFragments = kKeys / kScoreColumns;
    constexpr int kParts = kWarps / kScoreFragments;
    constexpr int kStepsPerPart = HeadSize / 16 / kParts;
    constexpr int kTileRows = M < 16 ? M : 16;
    constexpr int kTileColumns = 256 / kTileRows;
    constexpr int kRowTiles = M / kTileRows;
    constexpr int kColumnTiles = HeadSize / kTileColumns;
    constexpr int kOutputs = kRowTiles * kColumnTiles;
    constexpr int kKeyParts = kOutputs >= kWarps ? 1 : kWarps / kOutputs;
    constexpr int kPerWarp = kOutputs >= kWarps ? kOutputs / kWarps : 1;
    constexpr int kLanesPerRow = kThreads / M;
    constexpr int kKeysPerLane = kKeys / kLanesPerRow;
    static_assert(kStepsPerPart >= 1 && kKeyParts <= kKeySteps, "unsupported head size");
    static_assert(kKeyParts > 1 || kOutputs % kWarps == 0, "outputs must divide over warps");
    static_assert(kColumnTiles % kPerWarp == 0, "a warp's outputs share one row tile");
    static_assert(kLanesPerRow <= 32 && kParts * M == 8 * kWarps, "softmax rows stay in a warp");

    using ScoreAccumulator = wmma::fragment<wmma::accumulator, M, kScoreColumns, 16, float>;
    using QueryOperand =
        wmma::fragment<wmma::matrix_a, M, kScoreColumns, 16, __half, wmma::row_major>;
    using KeyOperand =
        wmma::fragment<wmma::matrix_b, M, kScoreColumns, 16, __half, wmma::col_major>;
    using OutputAccumulator = wmma::fragment<wmma::accumulator, kTileRows, kTileColumns, 16, float>;
    using WeightOperand =
        wmma::fragment<wmma::matrix_a, kTileRows, kTileColumns, 16, __half, wmma::row_major>;
    using ValueOperand =
        wmma::fragment<wmma::matrix_b, kTileRows, kTileColumns, 16, __half, wmma::row_major>;

    extern __shared__ __align__(32) unsigned char shared[];
    __half* queries = reinterpret_cast<__half*>(shared);
    __half* keys = queries + M * kStride;
    __half* values = keys + kKeys * kStride;
    float* scores = reinterpret_cast<float*>(values + kKeys * kStride);
    __half* probabilities = reinterpret_cast<__half*>(scores + kParts * M * kScoreStride);
    float* alphas = reinterpret_cast<float*>(probabilities + M * kProbabilityStride);
    float* key_scales = alphas + M;

    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int split = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int sequence = static_cast<int>(blockIdx.z);
    const Chunk chunk = read_chunk(p, sequence);
    if (!chunk.valid || chunk.rows == 0) {
        return;
    }
    const int per_split = split_length(chunk.tokens, p.splits);
    if (static_cast<long long>(split) * per_split >= chunk.tokens) {
        return;
    }
    const int begin = split * per_split;
    const int end = static_cast<int>(
        min(static_cast<long long>(chunk.tokens), static_cast<long long>(begin) + per_split));
    const int used_rows = chunk.rows * p.group;
    const std::int32_t* table = p.block_tables + sequence * p.block_table_stride;

    for (int i = static_cast<int>(threadIdx.x); i < M * HeadSize; i += kThreads) {
        const int m = i / HeadSize;
        const int d = i - m * HeadSize;
        __half value = __float2half_rn(0.0F);
        if (m < used_rows) {
            const int r = m / p.group;
            const int h = m - r * p.group;
            value = p.query[(chunk.first_row + r) * p.query_row_stride +
                            (kv_head * p.group + h) * HeadSize + d];
        }
        queries[m * kStride + d] = value;
    }
    for (int i = static_cast<int>(threadIdx.x); i < kTileRows * kScoreStride; i += kThreads) {
        scores[i] = static_cast<float>(i / kScoreStride);
    }
    __syncthreads();
    int row_of[OutputAccumulator::num_elements];
    {
        OutputAccumulator rows;
        wmma::load_matrix_sync(rows, scores, kScoreStride, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < OutputAccumulator::num_elements; ++i) {
            row_of[i] = static_cast<int>(rows.x[i]);
        }
    }

    const int score_fragment = warp % kScoreFragments;
    const int score_part = warp / kScoreFragments;
    // A warp always multiplies the same rows and head elements, so its query
    // operands stay in registers for the whole split.
    QueryOperand query[kStepsPerPart];
#pragma unroll
    for (int step = 0; step < kStepsPerPart; ++step) {
        wmma::load_matrix_sync(query[step], queries + (score_part * kStepsPerPart + step) * 16,
                               kStride);
    }
    const int first_output = (warp % (kWarps / kKeyParts)) * kPerWarp;
    const int key_part = warp / (kWarps / kKeyParts);
    const int row_tile = first_output / kColumnTiles;
    const int my_row = static_cast<int>(threadIdx.x) / kLanesPerRow;
    // Lane `sub` of a row owns keys sub, sub + kLanesPerRow, ... of each tile.
    const int sub = static_cast<int>(threadIdx.x) % kLanesPerRow;
    // Keys before `limit` are visible to this lane's row: its causal length,
    // cut to the split. Padding rows see none.
    int limit = begin;
    if (my_row < used_rows) {
        const int r = my_row / p.group;
        limit = min(end, chunk.tokens - (chunk.rows - 1 - r));
    }
    float running_max = -INFINITY;
    float running_sum = 0.0F;

    OutputAccumulator output[kPerWarp];
#pragma unroll
    for (OutputAccumulator& fragment : output) {
        wmma::fill_fragment(fragment, 0.0F);
    }
    TileLoad<HeadSize, kWarps> next;
    TokenCell cell = resolve_tile<kWarps>(p, table, kv_head, begin, end, warp, lane);
    fetch_tile<HeadSize, kWarps>(p, cell, lane, next);
    cell = resolve_tile<kWarps>(p, table, kv_head, begin + kKeys, end, warp, lane);
    for (int first_key = begin; first_key < end; first_key += kKeys) {
        __syncthreads();
        store_tile<HeadSize, kWarps>(next, warp, lane, keys, values, key_scales);
        __syncthreads();
        if (first_key + kKeys < end) {
            fetch_tile<HeadSize, kWarps>(p, cell, lane, next);
            cell = resolve_tile<kWarps>(p, table, kv_head, first_key + 2 * kKeys, end, warp, lane);
        }
        {
            ScoreAccumulator tile_scores;
            wmma::fill_fragment(tile_scores, 0.0F);
#pragma unroll
            for (int step = 0; step < kStepsPerPart; ++step) {
                const int k = (score_part * kStepsPerPart + step) * 16;
                KeyOperand key;
                wmma::load_matrix_sync(key, keys + score_fragment * kScoreColumns * kStride + k,
                                       kStride);
                wmma::mma_sync(tile_scores, query[step], key, tile_scores);
            }
            wmma::store_matrix_sync(
                scores + score_part * M * kScoreStride + score_fragment * kScoreColumns,
                tile_scores, kScoreStride, wmma::mem_row_major);
        }
        __syncthreads();

        float row[kKeysPerLane];
        float tile_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < kKeysPerLane; ++c) {
            const int key = sub + c * kLanesPerRow;
            float sum = 0.0F;
#pragma unroll
            for (int part = 0; part < kParts; ++part) {
                sum += scores[(part * M + my_row) * kScoreStride + key];
            }
            row[c] = first_key + key < limit ? sum * key_scales[key] * p.scale : -INFINITY;
            tile_max = nan_max(tile_max, row[c]);
        }
#pragma unroll
        for (int offset = kLanesPerRow / 2; offset > 0; offset /= 2) {
            tile_max = nan_max(tile_max, __shfl_xor_sync(kFullMask, tile_max, offset));
        }
        // A row that has seen no visible key keeps zero weights and alpha 1.
        // A NaN maximum makes alpha, the weights and the row NaN.
        const float new_max = nan_max(running_max, tile_max);
        const bool empty = new_max == -INFINITY;
        const float alpha = empty ? 1.0F : __expf(running_max - new_max);
        float tile_sum = 0.0F;
#pragma unroll
        for (int c = 0; c < kKeysPerLane; ++c) {
            const __half weight = __float2half_rn(empty ? 0.0F : __expf(row[c] - new_max));
            // Sum what the products will use, so the normalization matches.
            tile_sum += __half2float(weight);
            probabilities[my_row * kProbabilityStride + sub + c * kLanesPerRow] = weight;
        }
#pragma unroll
        for (int offset = kLanesPerRow / 2; offset > 0; offset /= 2) {
            tile_sum += __shfl_xor_sync(kFullMask, tile_sum, offset);
        }
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
        if (sub == 0) {
            alphas[my_row] = alpha;
        }
        __syncthreads();

        float alpha_of[OutputAccumulator::num_elements];
        bool settled = true;
#pragma unroll
        for (int i = 0; i < OutputAccumulator::num_elements; ++i) {
            alpha_of[i] = alphas[row_tile * kTileRows + row_of[i]];
            settled = settled && alpha_of[i] == 1.0F;
        }
        // Once every row's maximum has settled, alpha is exactly 1.
        if (!__all_sync(kFullMask, settled)) {
#pragma unroll
            for (OutputAccumulator& fragment : output) {
#pragma unroll
                for (int i = 0; i < OutputAccumulator::num_elements; ++i) {
                    fragment.x[i] *= alpha_of[i];
                }
            }
        }
#pragma unroll
        for (int step = 0; step < kKeySteps / kKeyParts; ++step) {
            const int key_step = key_part * (kKeySteps / kKeyParts) + step;
            WeightOperand weights;
            wmma::load_matrix_sync(
                weights, probabilities + row_tile * kTileRows * kProbabilityStride + key_step * 16,
                kProbabilityStride);
#pragma unroll
            for (int i = 0; i < kPerWarp; ++i) {
                const int column_tile = (first_output + i) % kColumnTiles;
                ValueOperand value;
                wmma::load_matrix_sync(
                    value, values + key_step * 16 * kStride + column_tile * kTileColumns, kStride);
                wmma::mma_sync(output[i], weights, value, output[i]);
            }
        }
    }

    // Every warp has finished reading the last V tile, which staging overwrites.
    __syncthreads();
    float* staging = reinterpret_cast<float*>(keys);  // M rows of kStride FP32 values.
    for (int part = 0; part < kKeyParts; ++part) {
        if (key_part == part) {
#pragma unroll
            for (int i = 0; i < kPerWarp; ++i) {
                const int column_tile = (first_output + i) % kColumnTiles;
                float* target =
                    staging + row_tile * kTileRows * kStride + column_tile * kTileColumns;
                if (part > 0) {
                    OutputAccumulator earlier;
                    wmma::load_matrix_sync(earlier, target, kStride, wmma::mem_row_major);
#pragma unroll
                    for (int e = 0; e < OutputAccumulator::num_elements; ++e) {
                        output[i].x[e] += earlier.x[e];
                    }
                }
                wmma::store_matrix_sync(target, output[i], kStride, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }
    for (int i = static_cast<int>(threadIdx.x); i < used_rows * HeadSize; i += kThreads) {
        const int m = i / HeadSize;
        const int d = i - m * HeadSize;
        const int r = m / p.group;
        const long long index = (static_cast<long long>(chunk.first_row + r) * p.heads +
                                 kv_head * p.group + m % p.group) *
                                    p.splits +
                                split;
        p.partials[index * HeadSize + d] = staging[m * kStride + d];
    }
    if (sub == 0 && my_row < used_rows) {
        const int r = my_row / p.group;
        const long long index = (static_cast<long long>(chunk.first_row + r) * p.heads +
                                 kv_head * p.group + my_row % p.group) *
                                    p.splits +
                                split;
        p.maxima[index] = running_max;
        p.sums[index] = running_sum;
    }
}

// Each Volta quad pair multiplies its own row-major 8x4 A by a column-major
// 4x8 B. A row and B column are lane%4 + 4*(lane/16). Accumulator i is at
// row (lane&1)+(i&2)+4*(lane/16), column (i&4)+(lane&2)+(i&1).
template <bool RowMajorB = false>
__device__ __forceinline__ void decode_mma(float (&sum)[8], unsigned int a01, unsigned int a23,
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

__device__ __forceinline__ unsigned int decode_pair_bits(__half2 value) {
    unsigned int bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__device__ TokenCell decode_cell(const Params& p, const std::int32_t* table, int kv_head,
                                 int position, int end) {
    TokenCell token;
    if (position < end) {
        const int entry = position / p.block_size;
        const int block = table[entry];
        if (block >= 0 && block < p.blocks) {
            const int slot = position - entry * p.block_size;
            token.present = 1;
            token.cell = block * p.block_stride + slot * p.slot_stride + kv_head * p.head_stride;
            const long long index = block * p.scale_block_stride + slot * p.scale_slot_stride +
                                    kv_head * p.scale_head_stride;
            token.key_scale = p.key_scales[index];
            token.value_scale = p.value_scales[index];
        }
    }
    return token;
}

// 256-dimensional, 24-row tile, selected inside the 32-row capacity. QK warps own eight heads
// and eight keys; their quad pairs split the head dimension into four parts.
// PV warps own eight heads and 64 output columns. K and V enter mma fragments
// directly from registers. Only scores/probabilities cross warps, requiring
// two block barriers per 64-key tile. Eight-byte K alignment permits uint2
// loads; V keeps the contract's four-byte alignment. WMMA covers other shapes.
constexpr std::size_t kDecodeMmaSharedBytes = 32U * 256U * sizeof(float);

__global__ void __launch_bounds__(384) paged_decode_mma_kernel(Params p) {
    constexpr int kQueryStride = 264;
    constexpr int kScoreStride = 68;
    constexpr int kProbabilityStrideMma = 72;
    extern __shared__ __align__(32) unsigned char shared[];
    __half* queries = reinterpret_cast<__half*>(shared);
    float* scores = reinterpret_cast<float*>(queries + 24 * kQueryStride);
    __half* probabilities = reinterpret_cast<__half*>(scores + 24 * kScoreStride);
    float* alphas = reinterpret_cast<float*>(probabilities + 24 * kProbabilityStrideMma);
    float* key_scales = alphas + 24;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int quad = lane / 4 % 4;
    const int upper = lane / 16;
    const int row_tile = warp / 4;
    const int fragment_row = lane % 4 + 4 * upper;
    const int split = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int sequence = static_cast<int>(blockIdx.z);
    const Chunk chunk = read_chunk(p, sequence);
    if (!chunk.valid || chunk.rows == 0) {
        return;
    }
    const int per_split = split_length(chunk.tokens, p.splits);
    if (static_cast<long long>(split) * per_split >= chunk.tokens) {
        return;
    }
    const int begin = split * per_split;
    const int end = static_cast<int>(
        min(static_cast<long long>(chunk.tokens), static_cast<long long>(begin) + per_split));
    const int used_rows = chunk.rows * p.group;
    const std::int32_t* table = p.block_tables + sequence * p.block_table_stride;
    for (int i = static_cast<int>(threadIdx.x); i < 24 * 256; i += 384) {
        const int m = i / 256;
        const int d = i % 256;
        queries[m * kQueryStride + d] =
            m < used_rows ? p.query[(chunk.first_row + m / p.group) * p.query_row_stride +
                                    (kv_head * p.group + m % p.group) * 256 + d]
                          : __float2half_rn(0.0F);
    }
    __syncthreads();
    unsigned int query[32];
#pragma unroll
    for (int step = 0; step < 16; ++step) {
        memcpy(&query[2 * step],
               queries + (row_tile * 8 + fragment_row) * kQueryStride + quad * 64 + step * 4,
               2 * sizeof(unsigned int));
    }
    float output[2][8] = {};
    const int my_row = static_cast<int>(threadIdx.x) / 16;
    const int sub = lane % 16;
    int limit = begin;
    if (my_row < used_rows) {
        limit = min(end, chunk.tokens - (chunk.rows - 1 - my_row / p.group));
    }
    float running_max = -INFINITY;
    float running_sum = 0.0F;
    for (int first_key = begin; first_key < end; first_key += 64) {
        // Resolve each tile token once per warp. The QK and PV layouts
        // select it with shuffles instead of repeating table divisions for
        // every four-token PV step.
        TokenCell metadata[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            metadata[half] = decode_cell(p, table, kv_head, first_key + half * 32 + lane, end);
            if (warp == 0) {
                key_scales[half * 32 + lane] = metadata[half].key_scale;
            }
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int token_index = warp % 4 * 8 + fragment_row;
            const long long key_cell = __shfl_sync(kFullMask, metadata[half].cell, token_index);
            const int key_present = __shfl_sync(kFullMask, metadata[half].present, token_index);
            float dots[8] = {};
#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                uint2 words = {};
                if (key_present != 0) {
                    words =
                        *reinterpret_cast<const uint2*>(p.keys + key_cell + quad * 64 + pair * 8);
                }
                __half2 low;
                __half2 high;
                exact_halves(words.x, low, high);
                decode_mma(dots, query[4 * pair], query[4 * pair + 1], decode_pair_bits(low),
                           decode_pair_bits(high));
                exact_halves(words.y, low, high);
                decode_mma(dots, query[4 * pair + 2], query[4 * pair + 3], decode_pair_bits(low),
                           decode_pair_bits(high));
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dots[i] += __shfl_xor_sync(kFullMask, dots[i], 4);
                dots[i] += __shfl_xor_sync(kFullMask, dots[i], 8);
                if (quad == 0) {
                    const int m = row_tile * 8 + (lane & 1) + (i & 2) + 4 * upper;
                    const int key = half * 32 + warp % 4 * 8 + (i & 4) + (lane & 2) + (i & 1);
                    scores[m * kScoreStride + key] = dots[i];
                }
            }
        }
        __syncthreads();
        float row[4];
        float tile_max = -INFINITY;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int key = sub + c * 16;
            row[c] = first_key + key < limit
                         ? scores[my_row * kScoreStride + key] * key_scales[key] * p.scale
                         : -INFINITY;
            tile_max = nan_max(tile_max, row[c]);
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            tile_max = nan_max(tile_max, __shfl_xor_sync(kFullMask, tile_max, offset));
        }
        const float new_max = nan_max(running_max, tile_max);
        const bool empty = new_max == -INFINITY;
        const float alpha = empty ? 1.0F : __expf(running_max - new_max);
        float tile_sum = 0.0F;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const __half weight = __float2half_rn(empty ? 0.0F : __expf(row[c] - new_max));
            probabilities[my_row * kProbabilityStrideMma + sub + c * 16] = weight;
            tile_sum += __half2float(weight);
        }
#pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            tile_sum += __shfl_xor_sync(kFullMask, tile_sum, offset);
        }
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
        if (sub == 0) {
            alphas[my_row] = alpha;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = row_tile * 8 + (lane & 1) + (i & 2) + 4 * upper;
            const float factor = alphas[m];
            output[0][i] *= factor;
            output[1][i] *= factor;
        }
#pragma unroll
        for (int step = 0; step < 16; ++step) {
            // Row-major B owns four adjacent output columns at token
            // lane%4. Its fragment comes from one packed int8 word, so no
            // register transpose or shared V tile is needed.
            const int token = step % 8 * 4 + lane % 4;
            const long long cell = __shfl_sync(kFullMask, metadata[step / 8].cell, token);
            const int present = __shfl_sync(kFullMask, metadata[step / 8].present, token);
            const __half2 scale =
                __float2half2_rn(__shfl_sync(kFullMask, metadata[step / 8].value_scale, token));
            unsigned int weights[2];
            memcpy(weights,
                   probabilities + (row_tile * 8 + fragment_row) * kProbabilityStrideMma + step * 4,
                   sizeof(weights));
#pragma unroll
            for (int part = 0; part < 2; ++part) {
                const int column = warp % 4 * 64 + part * 32 + quad * 8 + upper * 4;
                unsigned int bytes = 0;
                if (present != 0) {
                    bytes = *reinterpret_cast<const unsigned int*>(p.values + cell + column);
                }
                __half2 low;
                __half2 high;
                exact_halves(bytes, low, high);
                decode_mma<true>(output[part], weights[0], weights[1],
                                 decode_pair_bits(__hmul2(low, scale)),
                                 decode_pair_bits(__hmul2(high, scale)));
            }
        }
        // The next iteration writes scores, not probabilities. All warps
        // finish reading the latter before its next write at the score barrier.
    }
    __syncthreads();
    float* staging = reinterpret_cast<float*>(shared);
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int m = row_tile * 8 + (lane & 1) + (i & 2) + 4 * upper;
            const int d = warp % 4 * 64 + part * 32 + quad * 8 + (i & 4) + (lane & 2) + (i & 1);
            staging[m * 256 + d] = output[part][i];
        }
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < used_rows * 256; i += 384) {
        const int m = i / 256;
        const long long index = (static_cast<long long>(chunk.first_row + m / p.group) * p.heads +
                                 kv_head * p.group + m % p.group) *
                                    p.splits +
                                split;
        p.partials[index * 256 + i % 256] = staging[i];
    }
    if (sub == 0 && my_row < used_rows) {
        const long long index =
            (static_cast<long long>(chunk.first_row + my_row / p.group) * p.heads +
             kv_head * p.group + my_row % p.group) *
                p.splits +
            split;
        p.maxima[index] = running_max;
        p.sums[index] = running_sum;
    }
}

// Sum of `value` over the block's HeadSize threads, returned to every thread.
// `scratch` holds one value per warp.
template <int HeadSize, typename Combine>
__device__ float block_reduce(float value, float* scratch, Combine combine) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value = combine(value, __shfl_xor_sync(kFullMask, value, offset));
    }
    const int warp = static_cast<int>(threadIdx.x) / 32;
    __syncthreads();  // Earlier readers of scratch are done.
    if (threadIdx.x % 32 == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    value = scratch[0];
#pragma unroll
    for (int w = 1; w < HeadSize / 32; ++w) {
        value = combine(value, scratch[w]);
    }
    return value;
}

// One block per query head, chunk row and sequence, one thread per head
// element. A split whose maximum is -inf saw no key of that row and gets
// weight 0, and its partial output is not read, since a masked NaN value may
// sit in it. A NaN maximum makes the row NaN through its weight.
template <int HeadSize>
__global__ void __launch_bounds__(HeadSize) paged_decode_merge_kernel(Params p) {
    __shared__ float weights[kPagedDecodeMaxSplits];
    __shared__ float scratch[HeadSize / 32];
    const int head = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    const int sequence = static_cast<int>(blockIdx.z);
    const int d = static_cast<int>(threadIdx.x);
    const Chunk chunk = read_chunk(p, sequence);
    if (!chunk.valid || r >= chunk.rows) {
        return;
    }
    const int per_split = split_length(chunk.tokens, p.splits);
    const int used = (chunk.tokens + per_split - 1) / per_split;
    const long long base =
        (static_cast<long long>(chunk.first_row + r) * p.heads + head) * p.splits;
    float largest = -INFINITY;
    for (int s = d; s < used; s += HeadSize) {
        weights[s] = p.maxima[base + s];
        largest = fmaxf(largest, weights[s]);
    }
    largest =
        block_reduce<HeadSize>(largest, scratch, [](float a, float b) { return fmaxf(a, b); });
    float total = 0.0F;
    for (int s = d; s < used; s += HeadSize) {
        const float weight = weights[s] == -INFINITY ? 0.0F : __expf(weights[s] - largest);
        weights[s] = weight;
        total += weight * p.sums[base + s];
    }
    total = block_reduce<HeadSize>(total, scratch, [](float a, float b) { return a + b; });
    float sum = 0.0F;
    for (int s = 0; s < used; ++s) {
        const float weight = weights[s];
        if (weight != 0.0F) {
            sum += weight * p.partials[(base + s) * HeadSize + d];
        }
    }
    p.output[(chunk.first_row + r) * p.output_row_stride + head * HeadSize + d] =
        __float2half_rn(sum / total);
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

// The smallest row capacity of the partial kernel that holds every chunk.
[[nodiscard]] int row_capacity(std::size_t group_rows) noexcept {
    return group_rows <= 8U ? 8 : group_rows <= 16U ? 16 : 32;
}

template <int HeadSize, int M>
[[nodiscard]] cudaError_t launch(const Params& params, unsigned int splits, unsigned int kv_heads,
                                 unsigned int sequences, cudaStream_t stream) noexcept {
    constexpr std::size_t bytes = kSharedBytes<HeadSize, M>;
    static_assert(bytes <= kMaxSharedBytes, "shared memory exceeds Volta's limit");
    cudaError_t status =
        cudaFuncSetAttribute(paged_decode_kernel<HeadSize, M>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (status == cudaSuccess) {
        status = cudaFuncSetAttribute(paged_decode_kernel<HeadSize, M>,
                                      cudaFuncAttributePreferredSharedMemoryCarveout,
                                      cudaSharedmemCarveoutMaxShared);
    }
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (HeadSize == 256 && M == 32) {
        if (params.max_rows * params.group <= 24 && aligned(params.keys, 8U) &&
            params.block_stride % 8 == 0 && params.slot_stride % 8 == 0 &&
            params.head_stride % 8 == 0) {
            paged_decode_mma_kernel<<<dim3(splits, kv_heads, sequences), 384, kDecodeMmaSharedBytes,
                                      stream>>>(params);
        } else {
            paged_decode_kernel<HeadSize, M>
                <<<dim3(splits, kv_heads, sequences), kWarpsFor<M> * 32, bytes, stream>>>(params);
        }
    } else {
        paged_decode_kernel<HeadSize, M>
            <<<dim3(splits, kv_heads, sequences), kWarpsFor<M> * 32, bytes, stream>>>(params);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    paged_decode_merge_kernel<HeadSize>
        <<<dim3(static_cast<unsigned int>(params.heads), static_cast<unsigned int>(params.max_rows),
                sequences),
           HeadSize, 0, stream>>>(params);
    return cudaGetLastError();
}

template <int HeadSize>
[[nodiscard]] cudaError_t launch_head_size(const Params& params, int capacity, unsigned int splits,
                                           unsigned int kv_heads, unsigned int sequences,
                                           cudaStream_t stream) noexcept {
    switch (capacity) {
        case 8:
            return launch<HeadSize, 8>(params, splits, kv_heads, sequences, stream);
        case 16:
            return launch<HeadSize, 16>(params, splits, kv_heads, sequences, stream);
        default:
            return launch<HeadSize, 32>(params, splits, kv_heads, sequences, stream);
    }
}

}  // namespace

bool paged_decode_heads_supported(std::size_t heads, std::size_t kv_heads, std::size_t head_size,
                                  std::size_t max_rows_per_sequence) noexcept {
    if (kv_heads == 0U || heads == 0U || heads % kv_heads != 0U || max_rows_per_sequence == 0U) {
        return false;
    }
    const std::size_t group = heads / kv_heads;
    return (head_size == 128U || head_size == 256U) &&
           max_rows_per_sequence <= kPagedDecodeMaxGroupRows / group && kv_heads <= 65535U &&
           heads <= 65535U;
}

bool paged_decode_supported(const ops::PagedDecodeShape& shape,
                            const ops::PagedInt8CacheLayout& layout,
                            const ops::PagedDecodeBuffers& buffers) noexcept {
    if (ops::validate_paged_decode(shape, layout, buffers) != ops::Status::Success ||
        shape.rows == 0U || shape.sequences == 0U) {
        return false;
    }
    return paged_decode_heads_supported(shape.heads, layout.kv_heads, layout.head_size,
                                        shape.max_rows_per_sequence) &&
           shape.sequences <= 65535U && shape.splits <= kPagedDecodeMaxSplits &&
           fits_int(shape.rows) && fits_int(layout.blocks) && fits_int(layout.block_size) &&
           fits_long(shape.query_row_stride) && fits_long(shape.output_row_stride) &&
           fits_long(shape.block_table_stride) && fits_long(layout.block_stride) &&
           fits_long(layout.slot_stride) && fits_long(layout.head_stride) &&
           fits_long(layout.scale_block_stride) && fits_long(layout.scale_slot_stride) &&
           fits_long(layout.scale_head_stride) && aligned(buffers.keys.data, 4U) &&
           aligned(buffers.values.data, 4U) && layout.block_stride % 4U == 0U &&
           layout.slot_stride % 4U == 0U && layout.head_stride % 4U == 0U;
}

cudaError_t launch_paged_decode(const ops::PagedDecodeShape& shape,
                                const ops::PagedInt8CacheLayout& layout,
                                const ops::PagedDecodeBuffers& buffers,
                                cudaStream_t stream) noexcept {
    if ((shape.rows == 0U || shape.sequences == 0U) &&
        ops::validate_paged_decode(shape, layout, buffers) == ops::Status::Success) {
        return cudaSuccess;
    }
    if (!paged_decode_supported(shape, layout, buffers)) {
        return cudaErrorInvalidValue;
    }
    // Validation bounded max_blocks_per_sequence * block_size.
    const std::size_t max_tokens = shape.max_blocks_per_sequence * layout.block_size;
    const std::size_t group = shape.heads / layout.kv_heads;
    const std::size_t entries = shape.rows * shape.heads * shape.splits;
    float* const workspace = buffers.workspace.data;
    const Params params{
        reinterpret_cast<const __half*>(buffers.query.data),
        reinterpret_cast<__half*>(buffers.output.data),
        buffers.keys.data,
        buffers.values.data,
        buffers.key_scales.data,
        buffers.value_scales.data,
        buffers.block_tables.data,
        buffers.query_starts.data,
        buffers.seq_lens.data,
        workspace,
        workspace + entries,
        workspace + 2U * entries,
        static_cast<long long>(shape.query_row_stride),
        static_cast<long long>(shape.output_row_stride),
        static_cast<long long>(shape.block_table_stride),
        static_cast<long long>(layout.block_stride),
        static_cast<long long>(layout.slot_stride),
        static_cast<long long>(layout.head_stride),
        static_cast<long long>(layout.scale_block_stride),
        static_cast<long long>(layout.scale_slot_stride),
        static_cast<long long>(layout.scale_head_stride),
        static_cast<int>(layout.blocks),
        static_cast<int>(layout.block_size),
        static_cast<int>(shape.heads),
        static_cast<int>(group),
        static_cast<int>(shape.rows),
        static_cast<int>(shape.max_rows_per_sequence),
        fits_int(max_tokens) ? static_cast<int>(max_tokens) : INT_MAX,
        static_cast<int>(shape.splits),
        shape.scale,
    };
    const int capacity = row_capacity(shape.max_rows_per_sequence * group);
    const auto splits = static_cast<unsigned int>(shape.splits);
    const auto kv_heads = static_cast<unsigned int>(layout.kv_heads);
    const auto sequences = static_cast<unsigned int>(shape.sequences);
    return layout.head_size == 256U
               ? launch_head_size<256>(params, capacity, splits, kv_heads, sequences, stream)
               : launch_head_size<128>(params, capacity, splits, kv_heads, sequences, stream);
}

cudaError_t check_paged_decode_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index, paged_decode_kernel<128, 8>,
                                       paged_decode_kernel<128, 16>, paged_decode_kernel<128, 32>,
                                       paged_decode_kernel<256, 8>, paged_decode_kernel<256, 16>,
                                       paged_decode_kernel<256, 32>, paged_decode_merge_kernel<128>,
                                       paged_decode_merge_kernel<256>, paged_decode_mma_kernel);
}

}  // namespace eclair::cuda
