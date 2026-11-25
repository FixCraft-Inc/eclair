/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Internal to eclair_cuda: the FP16 tensor-core tile loop shared by linear
// operations with binary16 activations and weights that become binary16 in
// shared memory. Include only from CUDA units. Not a public header.

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace eclair::cuda::detail {

// 64x64 output tiles and 32-deep K stages. Four warps form a 2x2 grid, each
// owning 32x32 outputs as 2x2 WMMA fragments. Each thread stages 16 values of
// A and 16 of W per K stage.
constexpr std::uint32_t kTileM = 64U;
constexpr std::uint32_t kTileN = 64U;
constexpr std::uint32_t kTileK = 32U;
constexpr std::uint32_t kFragment = 16U;
constexpr std::uint32_t kWarpTile = 32U;
constexpr unsigned int kTileThreads = 128U;
constexpr std::uint32_t kPerThread = 16U;
// WMMA needs a leading dimension that is a multiple of 8 halves. The extra 8
// halves give an 80-byte row stride, which staggers shared-memory banks.
constexpr std::uint32_t kStageLd = kTileK + 8U;
constexpr std::size_t kStageBytes = 2U * kTileM * kStageLd * sizeof(__half);
constexpr std::size_t kOutputBytes = (kTileThreads / 32U) * kWarpTile * kWarpTile * sizeof(float);
constexpr std::size_t kTileSharedBytes = kStageBytes > kOutputBytes ? kStageBytes : kOutputBytes;
constexpr std::size_t kMaxTileBlocks = 65535U;

static_assert(kTileM * kTileK == kTileThreads * kPerThread, "An A stage is one load per thread");
static_assert(kTileN * kTileK == kTileThreads * kPerThread, "A W stage is one load per thread");

// Each thread stages row threadIdx.x / 2 of a tile, from column 0 or 16.
[[nodiscard]] __device__ inline std::uint32_t stage_row() {
    return threadIdx.x / 2U;
}

[[nodiscard]] __device__ inline std::uint32_t stage_column() {
    return (threadIdx.x % 2U) * kPerThread;
}

// The product's dimensions, each below 2^31. The launcher sets
// `vector_activations` when K and A's base pointer allow 16-byte loads of
// every row.
struct TileShape {
    std::uint32_t m;
    std::uint32_t n;
    std::uint32_t k;
    bool vector_activations;
};

// One thread's 16 binary16 values of a stage row.
struct Fp16Slice {
    uint4 words[2];
};

// The 16 values of `row` from column `k` of a row-major [rows, columns]
// binary16 matrix. Elements outside the matrix hold zero bits, which are +0.
// `vector` allows two 16-byte loads when the slice is whole.
__device__ inline void load_fp16_slice(const std::uint16_t* matrix, std::uint32_t rows,
                                       std::uint32_t columns, std::uint32_t row, std::uint32_t k,
                                       bool vector, Fp16Slice& slice) {
    if (vector && k + kPerThread <= columns && row < rows) {
        const auto* source =
            reinterpret_cast<const uint4*>(matrix + static_cast<std::size_t>(row) * columns + k);
        slice.words[0] = source[0];
        slice.words[1] = source[1];
        return;
    }
    std::uint32_t words[8] = {};
    for (std::uint32_t i = 0; i < kPerThread && row < rows && k + i < columns; ++i) {
        words[i / 2U] |=
            static_cast<std::uint32_t>(matrix[static_cast<std::size_t>(row) * columns + k + i])
            << (16U * (i % 2U));
    }
    slice.words[0] = uint4{words[0], words[1], words[2], words[3]};
    slice.words[1] = uint4{words[4], words[5], words[6], words[7]};
}

// Writes a slice to the thread's slot in a stage tile.
__device__ inline void store_fp16_slice(const Fp16Slice& slice, __half* tile) {
    auto* target = reinterpret_cast<uint4*>(tile + stage_row() * kStageLd + stage_column());
    target[0] = slice.words[0];
    target[1] = slice.words[1];
}

// Weights already in binary16: W[N,K], row-major. The launcher sets `vector`
// as for the activations.
struct Fp16Weights {
    const std::uint16_t* data;
    std::uint32_t n;
    std::uint32_t k;
    bool vector;

    using Registers = Fp16Slice;

    __device__ void load(std::uint32_t n_begin, std::uint32_t k_begin, Registers& registers) const {
        load_fp16_slice(data, n, k, n_begin + stage_row(), k_begin + stage_column(), vector,
                        registers);
    }

    __device__ void store(std::uint32_t, std::uint32_t, const Registers& registers,
                          __half* b_tile) const {
        store_fp16_slice(registers, b_tile);
    }
};

// An optional FP32 bias[N]. Null when absent, which adds zero.
struct Bias {
    const float* data;

    [[nodiscard]] __device__ float operator[](std::uint32_t n) const {
        return data == nullptr ? 0.0F : data[n];
    }
};

struct BiasEpilogue {
    Bias bias;

    [[nodiscard]] __device__ float operator()(float sum, std::uint32_t, std::uint32_t n) const {
        return sum + bias[n];
    }
};

// output[m,n] = epilogue(sum_k A[m,k] * B[k,n], m, n), with FP16 WMMA and FP32
// accumulation, grid-striding over output tiles. B[k,n] is whatever binary16
// value `weights` stores for W[n,k]. The next stage's global loads are issued
// before the current stage's MMAs, so memory latency overlaps tensor work.
//
// Weights provides a `Registers` type, `load(n_begin, k_begin, registers)`,
// which fetches the thread's raw share of a stage from global memory, and
// `store(n_begin, k_begin, registers, b_tile)`, which writes those values as
// binary16 to the thread's slot. Values outside W must be stored as +0, so
// padding contributes nothing. Epilogue maps (sum, m, n) to the output value.
template <typename Weights, typename Epilogue>
__global__ void __launch_bounds__(kTileThreads)
    linear_tile_kernel(TileShape shape, const std::uint16_t* activations, Weights weights,
                       Epilogue epilogue, float* output) {
    using namespace nvcuda;
    // Stage tiles during the K loop. The same storage then holds each warp's
    // FP32 sums for the bounds-checked epilogue store.
    __shared__ __align__(32) unsigned char storage[kTileSharedBytes];
    auto* a_tile = reinterpret_cast<__half*>(storage);
    auto* b_tile = a_tile + kTileM * kStageLd;
    const std::uint32_t warp = threadIdx.x / 32U;
    const std::uint32_t lane = threadIdx.x % 32U;
    const std::uint32_t warp_m = (warp / 2U) * kWarpTile;
    const std::uint32_t warp_n = (warp % 2U) * kWarpTile;
    const std::uint32_t tile_rows = 1U + (shape.m - 1U) / kTileM;
    const std::uint32_t tile_columns = 1U + (shape.n - 1U) / kTileN;
    for (std::size_t tile = blockIdx.x; tile < static_cast<std::size_t>(tile_rows) * tile_columns;
         tile += gridDim.x) {
        const auto m_begin = static_cast<std::uint32_t>(tile / tile_columns) * kTileM;
        const auto n_begin = static_cast<std::uint32_t>(tile % tile_columns) * kTileN;
        wmma::fragment<wmma::accumulator, kFragment, kFragment, kFragment, float> sums[2][2];
#pragma unroll
        for (auto& row : sums) {
            for (auto& fragment : row) {
                wmma::fill_fragment(fragment, 0.0F);
            }
        }
        Fp16Slice a_stage;
        typename Weights::Registers w_stage;
        load_fp16_slice(activations, shape.m, shape.k, m_begin + stage_row(), stage_column(),
                        shape.vector_activations, a_stage);
        weights.load(n_begin, 0U, w_stage);
        for (std::uint32_t k_begin = 0; k_begin < shape.k; k_begin += kTileK) {
            // Every warp has finished reading the previous stage or sums.
            __syncthreads();
            store_fp16_slice(a_stage, a_tile);
            weights.store(n_begin, k_begin, w_stage, b_tile);
            __syncthreads();
            if (k_begin + kTileK < shape.k) {
                load_fp16_slice(activations, shape.m, shape.k, m_begin + stage_row(),
                                k_begin + kTileK + stage_column(), shape.vector_activations,
                                a_stage);
                weights.load(n_begin, k_begin + kTileK, w_stage);
            }
#pragma unroll
            for (std::uint32_t kk = 0; kk < kTileK; kk += kFragment) {
                wmma::fragment<wmma::matrix_a, kFragment, kFragment, kFragment, __half,
                               wmma::row_major>
                    a[2];
                wmma::fragment<wmma::matrix_b, kFragment, kFragment, kFragment, __half,
                               wmma::col_major>
                    b[2];
#pragma unroll
                for (std::uint32_t i = 0; i < 2U; ++i) {
                    wmma::load_matrix_sync(a[i], a_tile + (warp_m + i * kFragment) * kStageLd + kk,
                                           kStageLd);
                    wmma::load_matrix_sync(b[i], b_tile + (warp_n + i * kFragment) * kStageLd + kk,
                                           kStageLd);
                }
#pragma unroll
                for (std::uint32_t i = 0; i < 2U; ++i) {
#pragma unroll
                    for (std::uint32_t j = 0; j < 2U; ++j) {
                        wmma::mma_sync(sums[i][j], a[i], b[j], sums[i][j]);
                    }
                }
            }
        }
        __syncthreads();
        float* staged = reinterpret_cast<float*>(storage) + warp * kWarpTile * kWarpTile;
#pragma unroll
        for (std::uint32_t i = 0; i < 2U; ++i) {
#pragma unroll
            for (std::uint32_t j = 0; j < 2U; ++j) {
                wmma::store_matrix_sync(staged + i * kFragment * kWarpTile + j * kFragment,
                                        sums[i][j], kWarpTile, wmma::mem_row_major);
            }
        }
        __syncwarp();
        for (std::uint32_t index = lane; index < kWarpTile * kWarpTile; index += 32U) {
            const std::uint32_t m = m_begin + warp_m + index / kWarpTile;
            const std::uint32_t n = n_begin + warp_n + index % kWarpTile;
            if (m < shape.m && n < shape.n) {
                output[static_cast<std::size_t>(m) * shape.n + n] = epilogue(staged[index], m, n);
            }
        }
    }
}

// One block per output tile, capped. A grid-stride loop covers the rest.
[[nodiscard]] inline unsigned int tile_blocks(const TileShape& shape) noexcept {
    const std::size_t tiles =
        static_cast<std::size_t>(1U + (shape.m - 1U) / kTileM) * (1U + (shape.n - 1U) / kTileN);
    return static_cast<unsigned int>(std::min(tiles, kMaxTileBlocks));
}

}  // namespace eclair::cuda::detail
