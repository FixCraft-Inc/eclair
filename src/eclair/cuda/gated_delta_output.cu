/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/gated_delta_output.hpp"

#include "kernel_image.hpp"

#include <cuda_fp16.h>
#include <mma.h>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace eclair::cuda {
namespace {

namespace wmma = nvcuda::wmma;
constexpr int kTile = 16;
constexpr int kChunk = 64;
constexpr int kValues = 64;
constexpr int kStride = kChunk + 8;
constexpr int kWarps = 4;
using Accumulator = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
using RowA = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using ColumnB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using RowB = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;

struct Params {
    const __half* query;
    const __half* keys;
    const __half* values;
    const __half* states;
    const float* gates;
    const std::int32_t* starts;
    const std::int32_t* indices;
    __half* output;
    int tokens;
    int sequences;
    int heads;
    int key_heads;
    int value_size;
    float scale;
};

template <int K>
constexpr std::size_t kSharedBytes = (3U * kChunk * (K + 8U) + kChunk * kStride) * sizeof(__half) +
                                     kChunk * kStride * sizeof(float) + kChunk * sizeof(float);

// Copies only real rows; a partial sequence's unused slots become zero.
__device__ void load_rows(__half* target, int stride, const __half* source, long long source_stride,
                          int width, int valid) {
    for (int item = static_cast<int>(threadIdx.x); item < kChunk * (width / 8);
         item += kWarps * 32) {
        const int row = item / (width / 8);
        const int column = (item % (width / 8)) * 8;
        uint4 word = make_uint4(0, 0, 0, 0);
        if (row < valid) {
            word = *reinterpret_cast<const uint4*>(source + row * source_stride + column);
        }
        *reinterpret_cast<uint4*>(target + row * stride + column) = word;
    }
}

// One block per chunk, value head and 64 value columns. Four warps own 16
// token rows each, retaining the gated prior in registers while they compute
// the causal correlations and their product with the new values.
template <int K>
__global__ void __launch_bounds__(128) gated_delta_output_kernel(Params p) {
    const int chunk = static_cast<int>(blockIdx.z);
    const int sequence = p.indices[2LL * chunk];
    const int local = p.indices[2LL * chunk + 1];
    if (sequence < 0 || sequence >= p.sequences || local < 0) {
        return;
    }
    const int first = p.starts[sequence];
    const int last = p.starts[sequence + 1];
    const long long begin = static_cast<long long>(first) + static_cast<long long>(local) * kChunk;
    if (first < 0 || last < first || last > p.tokens || begin >= last) {
        return;
    }
    const int valid = static_cast<int>(min(static_cast<long long>(kChunk), last - begin));
    const int head = static_cast<int>(blockIdx.y);
    const int key_head = head / (p.heads / p.key_heads);
    const int value_base = static_cast<int>(blockIdx.x) * kValues;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    constexpr int kKeyStride = K + 8;
    extern __shared__ __align__(32) unsigned char shared[];
    __half* q = reinterpret_cast<__half*>(shared);
    __half* k = q + kChunk * kKeyStride;
    __half* state = k + kChunk * kKeyStride;
    __half* v = state + kValues * kKeyStride;
    float* scores = reinterpret_cast<float*>(v + kChunk * kStride);
    float* gates = scores + kChunk * kStride;
    float* my_scores = scores + warp * kTile * kStride;
    __half* weights = reinterpret_cast<__half*>(my_scores);
    load_rows(q, kKeyStride, p.query + (begin * p.key_heads + key_head) * K,
              static_cast<long long>(p.key_heads) * K, K, valid);
    load_rows(k, kKeyStride, p.keys + (begin * p.key_heads + key_head) * K,
              static_cast<long long>(p.key_heads) * K, K, valid);
    load_rows(
        state, kKeyStride,
        p.states +
            ((static_cast<long long>(chunk) * p.heads + head) * p.value_size + value_base) * K,
        K, K, kValues);
    load_rows(v, kStride, p.values + (begin * p.heads + head) * p.value_size + value_base,
              static_cast<long long>(p.heads) * p.value_size, kValues, valid);
    for (int row = static_cast<int>(threadIdx.x); row < kChunk; row += kWarps * 32) {
        gates[row] = row < valid ? p.gates[(begin + row) * p.heads + head] : 0.0F;
    }
    for (int element = lane; element < kTile * kTile; element += 32) {
        my_scores[(element / kTile) * kStride + element % kTile] =
            static_cast<float>(element / kTile);
    }
    __syncthreads();
    int row_of[Accumulator::num_elements];
    {
        Accumulator rows;
        wmma::load_matrix_sync(rows, my_scores, kStride, wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            row_of[i] = static_cast<int>(rows.x[i]);
        }
    }
    Accumulator prior[4], correlations[4];
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        wmma::fill_fragment(prior[tile], 0.0F);
        wmma::fill_fragment(correlations[tile], 0.0F);
    }
#pragma unroll
    for (int column = 0; column < K; column += kTile) {
        RowA query;
        wmma::load_matrix_sync(query, q + warp * kTile * kKeyStride + column, kKeyStride);
#pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            ColumnB key, snapshot;
            wmma::load_matrix_sync(key, k + tile * kTile * kKeyStride + column, kKeyStride);
            wmma::load_matrix_sync(snapshot, state + tile * kTile * kKeyStride + column,
                                   kKeyStride);
            wmma::mma_sync(correlations[tile], query, key, correlations[tile]);
            wmma::mma_sync(prior[tile], query, snapshot, prior[tile]);
        }
    }
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            prior[tile].x[i] *= __expf(gates[warp * kTile + row_of[i]]);
        }
        wmma::store_matrix_sync(my_scores + tile * kTile, correlations[tile], kStride,
                                wmma::mem_row_major);
    }
    __syncwarp();
    // Each lane owns two scores of one row. All reads precede aliasing the
    // FP32 score rows with their binary16 correlations.
    for (int row = 0; row < kTile; ++row) {
        const int position = warp * kTile + row;
        const float left = my_scores[row * kStride + 2 * lane];
        const float right = my_scores[row * kStride + 2 * lane + 1];
        const float g = gates[position];
        const bool left_allowed = position < valid && 2 * lane <= position;
        const bool right_allowed = position < valid && 2 * lane + 1 <= position;
        const __half2 pair =
            __floats2half2_rn(left_allowed ? left * __expf(g - gates[2 * lane]) : 0.0F,
                              right_allowed ? right * __expf(g - gates[2 * lane + 1]) : 0.0F);
        __syncwarp();
        reinterpret_cast<__half2*>(weights + row * (2 * kStride))[lane] = pair;
        __syncwarp();
    }
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        RowA a;
        wmma::load_matrix_sync(a, weights + step * kTile, 2 * kStride);
#pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            RowB value;
            wmma::load_matrix_sync(value, v + step * kTile * kStride + tile * kTile, kStride);
            Accumulator contribution;
            wmma::fill_fragment(contribution, 0.0F);
            wmma::mma_sync(contribution, a, value, contribution);
#pragma unroll
            for (int i = 0; i < Accumulator::num_elements; ++i) {
                prior[tile].x[i] += contribution.x[i];
            }
        }
    }
    __syncwarp();
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
        for (int i = 0; i < Accumulator::num_elements; ++i) {
            prior[tile].x[i] *= p.scale;
        }
        wmma::store_matrix_sync(my_scores, prior[tile], kStride, wmma::mem_row_major);
        __syncwarp();
        for (int element = lane; element < kTile * kTile; element += 32) {
            const int row = warp * kTile + element / kTile;
            if (row < valid) {
                p.output[((begin + row) * p.heads + head) * p.value_size + value_base +
                         tile * kTile + element % kTile] =
                    __float2half_rn(my_scores[(element / kTile) * kStride + element % kTile]);
            }
        }
        __syncwarp();
    }
}

[[nodiscard]] bool aligned(const void* pointer) noexcept {
    return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
}

template <int K>
[[nodiscard]] cudaError_t launch(const Params& p, unsigned int chunks,
                                 cudaStream_t stream) noexcept {
    const auto bytes = kSharedBytes<K>;
    const auto status =
        cudaFuncSetAttribute(gated_delta_output_kernel<K>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        return status;
    }
    gated_delta_output_kernel<K><<<dim3(static_cast<unsigned int>(p.value_size / kValues),
                                        static_cast<unsigned int>(p.heads), chunks),
                                   kWarps * 32, bytes, stream>>>(p);
    return cudaGetLastError();
}

}  // namespace

bool gated_delta_output_supported(const ops::GatedDeltaOutputShape& s,
                                  const ops::GatedDeltaOutputBuffers& b) noexcept {
    return ops::validate_gated_delta_output(s, b) == ops::Status::Success && s.tokens != 0U &&
           (s.key_size == 64U || s.key_size == 128U) && s.value_size % kValues == 0U &&
           s.tokens <= INT_MAX && s.sequences <= INT_MAX && s.chunks <= 65535U &&
           s.heads <= 65535U && s.value_size <= INT_MAX && aligned(b.query.data) &&
           aligned(b.keys.data) && aligned(b.values.data) && aligned(b.states.data) &&
           aligned(b.output.data);
}

cudaError_t launch_gated_delta_output(const ops::GatedDeltaOutputShape& s,
                                      const ops::GatedDeltaOutputBuffers& b,
                                      cudaStream_t stream) noexcept {
    if (s.tokens == 0U && ops::validate_gated_delta_output(s, b) == ops::Status::Success) {
        return cudaSuccess;
    }
    if (!gated_delta_output_supported(s, b)) {
        return cudaErrorInvalidValue;
    }
    const Params p{reinterpret_cast<const __half*>(b.query.data),
                   reinterpret_cast<const __half*>(b.keys.data),
                   reinterpret_cast<const __half*>(b.values.data),
                   reinterpret_cast<const __half*>(b.states.data),
                   b.gates.data,
                   b.sequence_starts.data,
                   b.chunk_indices.data,
                   reinterpret_cast<__half*>(b.output.data),
                   static_cast<int>(s.tokens),
                   static_cast<int>(s.sequences),
                   static_cast<int>(s.heads),
                   static_cast<int>(s.key_heads),
                   static_cast<int>(s.value_size),
                   s.scale};
    return s.key_size == 128U ? launch<128>(p, static_cast<unsigned int>(s.chunks), stream)
                              : launch<64>(p, static_cast<unsigned int>(s.chunks), stream);
}

cudaError_t check_gated_delta_output_image(int device_index) noexcept {
    return detail::check_kernel_images(device_index, gated_delta_output_kernel<64>,
                                       gated_delta_output_kernel<128>);
}

}  // namespace eclair::cuda
