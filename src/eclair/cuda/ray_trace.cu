/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/ray_trace.hpp"
#include "eclair/cuda/ray_preview.hpp"
#include "eclair/render/shading.hpp"
#include "kernel_image.hpp"

#include <cuda_runtime.h>

namespace eclair::cuda {
namespace {
__global__ void ray_trace_kernel(ops::RayScene scene, ops::RayCamera camera,
                                 ops::RayTraceShape shape, std::uint8_t* rgba) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < shape.width * shape.height)
        render::ray_pixel(scene, camera, shape, i, rgba + static_cast<std::size_t>(i) * 4);
}
__global__ void preview_sample_kernel(ops::RayScene scene, ops::RayCamera camera,
                                      ops::RayTraceShape shape, render::PreviewPixel* current,
                                      render::LightingLayout layout,
                                      const render::SurfaceLighting* lighting) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < shape.width * shape.height)
        current[i] = render::preview_sample(scene, camera, shape, i, layout, lighting);
}
__global__ void lighting_sample_kernel(ops::RayScene scene, render::LightingLayout layout,
                                       ops::RayTraceShape shape, render::SurfaceLighting* current) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < layout.tiles * render::kLightingTilePixels)
        current[i] = render::lighting_sample(scene, layout, shape, i);
}
__global__ void lighting_spatial_kernel(std::uint32_t count, const render::SurfaceLighting* current,
                                        const render::Vec3* input, render::Vec3* output,
                                        bool vertical) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = render::lighting_spatial(current, input, i, vertical);
}
__global__ void lighting_temporal_kernel(std::uint32_t count, render::SurfaceLighting* current,
                                         const render::SurfaceLighting* previous,
                                         const render::Vec3* spatial, bool valid, bool stationary) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        current[i] = render::lighting_temporal(previous[i], spatial[i], valid, stationary,
                                               current[i].direct);
}
__global__ void preview_boundary_kernel(ops::RayTraceShape shape,
                                        const render::PreviewPixel* current, std::uint32_t* queue) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool boundary =
        i < shape.width * shape.height && render::preview_boundary(current, shape, i);
    // One atomic reservation per nonempty warp, followed by coalesced writes.
    // Every lane reaches the ballot, including the last partial image warp.
    const unsigned mask = __ballot_sync(0xffffffffU, boundary);
    if (!boundary) return;
    const unsigned lane = threadIdx.x & 31U;
    const int leader = __ffs(static_cast<int>(mask)) - 1;
    unsigned base = lane == static_cast<unsigned>(leader) ? atomicAdd(queue, __popc(mask)) : 0;
    base = __shfl_sync(mask, base, leader);
    const unsigned preceding = __popc(mask & ((1U << lane) - 1));
    queue[1 + base + preceding] = i;
}
__global__ void preview_antialias_kernel(ops::RayScene scene, ops::RayCamera camera,
                                         ops::RayTraceShape shape,
                                         const render::PreviewPixel* current,
                                         const std::uint32_t* queue, std::uint8_t* rgba,
                                         render::LightingLayout layout,
                                         const render::SurfaceLighting* lighting) {
    const std::uint32_t count = queue[0];
    for (std::uint32_t packed = blockIdx.x * blockDim.x + threadIdx.x; packed < count;
         packed += gridDim.x * blockDim.x) {
        const auto i = queue[1 + packed];
        const auto color =
            render::coverage_preview(current, scene, camera, shape, i, layout, lighting);
        render::store_pixel(color, rgba + static_cast<std::size_t>(i) * 4);
    }
}
__global__ void preview_spatial_kernel(ops::RayTraceShape shape,
                                       const render::PreviewPixel* current,
                                       const render::Vec3* input, render::Vec3* scratch, int stride,
                                       bool vertical, bool cached_lighting) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < shape.width * shape.height)
        scratch[i] =
            render::spatial_preview(current, input, shape, i, stride, vertical, cached_lighting);
}
__global__ void preview_temporal_kernel(ops::RayScene scene, ops::RayScene previous_scene,
                                        ops::RayCamera previous_camera, ops::RayTraceShape shape,
                                        bool valid_history, bool stationary_scene,
                                        render::PreviewPixel* current,
                                        const render::PreviewPixel* previous,
                                        const render::Vec3* scratch, std::uint8_t* rgba,
                                        bool cached_lighting) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shape.width * shape.height) return;
    const auto result = render::temporal_preview(
        current, previous, scratch, scene, previous_scene, previous_camera, shape, i,
        valid_history && !(cached_lighting && current[i].filterable == 1), stationary_scene);
    // Neighbor reads use only geometry, which is immutable during this pass.
    current[i].color = result.color;
    current[i].history = result.history;
    render::store_pixel(result.color, rgba + static_cast<std::size_t>(i) * 4);
}
}  // namespace
cudaError_t launch_ray_trace(const ops::RayScene& scene, const ops::RayCamera& camera,
                             const ops::RayTraceShape& shape, ops::BufferView<std::uint8_t> rgba,
                             cudaStream_t stream) noexcept {
    std::size_t bytes = 0;
    if (ops::validate_ray_trace(scene, camera, shape) != ops::Status::Success ||
        ops::ray_frame_bytes(shape, &bytes) != ops::Status::Success || rgba.data == nullptr ||
        rgba.size < bytes)
        return cudaErrorInvalidValue;
    const std::uint32_t pixels = shape.width * shape.height;
    ray_trace_kernel<<<(pixels + 127) / 128, 128, 0, stream>>>(scene, camera, shape, rgba.data);
    return cudaGetLastError();
}
cudaError_t check_ray_trace_image(int device_index) noexcept {
    return detail::check_kernel_images(
        device_index, ray_trace_kernel, preview_sample_kernel, preview_boundary_kernel,
        preview_antialias_kernel, preview_spatial_kernel, preview_temporal_kernel,
        lighting_sample_kernel, lighting_spatial_kernel, lighting_temporal_kernel);
}
cudaError_t launch_ray_preview(
    const ops::RayScene& scene, const ops::RayCamera& camera, const ops::RayScene& previous_scene,
    const ops::RayCamera& previous_camera, const ops::RayTraceShape& shape, bool valid_history,
    ops::BufferView<render::PreviewPixel> current,
    ops::BufferView<const render::PreviewPixel> previous, ops::BufferView<render::Vec3> scratch,
    ops::BufferView<std::uint8_t> rgba, cudaStream_t stream, LightingBuffers lighting) noexcept {
    std::size_t bytes = 0;
    if (ops::validate_ray_trace(scene, camera, shape) != ops::Status::Success ||
        ops::ray_frame_bytes(shape, &bytes) != ops::Status::Success ||
        (valid_history &&
         ops::validate_ray_trace(previous_scene, previous_camera, shape) != ops::Status::Success))
        return cudaErrorInvalidValue;
    const std::size_t count = bytes / 4;
    if (!current.data || !previous.data || !scratch.data || !rgba.data ||
        current.data == previous.data || current.size < count || previous.size < count ||
        scratch.size < count * 2 || rgba.size < bytes)
        return cudaErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((count + 127) / 128);
    const auto layout = render::lighting_layout(scene);
    const auto lighting_count = layout.tiles * render::kLightingTilePixels;
    const bool cached = lighting.current.data != nullptr;
    const bool stationary = valid_history && render::stationary_preview_scene(
                                                 scene, camera, previous_scene, previous_camera);
    if (cached) {
        if (!lighting.previous.data || !lighting.scratch.data ||
            lighting.current.data == lighting.previous.data ||
            lighting.current.size < lighting_count || lighting.previous.size < lighting_count ||
            lighting.scratch.size < static_cast<std::size_t>(lighting_count) * 2)
            return cudaErrorInvalidValue;
        if (lighting_count > 0) {
            const unsigned grid = (lighting_count + 127) / 128;
            lighting_sample_kernel<<<grid, 128, 0, stream>>>(scene, layout, shape,
                                                             lighting.current.data);
            auto status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            lighting_spatial_kernel<<<grid, 128, 0, stream>>>(
                lighting_count, lighting.current.data, nullptr, lighting.scratch.data, false);
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            lighting_spatial_kernel<<<grid, 128, 0, stream>>>(
                lighting_count, lighting.current.data, lighting.scratch.data,
                lighting.scratch.data + lighting_count, true);
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            lighting_temporal_kernel<<<grid, 128, 0, stream>>>(
                lighting_count, lighting.current.data, lighting.previous.data,
                lighting.scratch.data + lighting_count, valid_history,
                valid_history &&
                    render::stationary_preview_scene(scene, camera, previous_scene, camera));
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
        }
    } else if (lighting.previous.data || lighting.scratch.data) {
        return cudaErrorInvalidValue;
    }
    preview_sample_kernel<<<blocks, 128, 0, stream>>>(scene, camera, shape, current.data, layout,
                                                      lighting.current.data);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    const unsigned passes = render::kPreviewSpatialPasses;
    for (unsigned pass = 0; pass < passes; ++pass) {
        const auto* input = pass == 0 ? nullptr : scratch.data + (pass % 2 == 0 ? count : 0);
        auto* output = scratch.data + (pass % 2 == 0 ? 0 : count);
        preview_spatial_kernel<<<blocks, 128, 0, stream>>>(shape, current.data, input, output,
                                                           1 << (pass / 2), pass % 2 != 0, cached);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    preview_temporal_kernel<<<blocks, 128, 0, stream>>>(
        scene, previous_scene, previous_camera, shape, valid_history, stationary, current.data,
        previous.data, scratch.data + count, rgba.data, cached);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    // Resolve coverage only after surface lighting/history is immutable.
    // Scratch is now dead; reuse it for [count, boundary pixel indices].
    auto* queue = reinterpret_cast<std::uint32_t*>(scratch.data);
    status = cudaMemsetAsync(queue, 0, sizeof(std::uint32_t), stream);
    if (status != cudaSuccess) return status;
    preview_boundary_kernel<<<blocks, 128, 0, stream>>>(shape, current.data, queue);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    preview_antialias_kernel<<<256, 128, 0, stream>>>(scene, camera, shape, current.data, queue,
                                                      rgba.data, layout, lighting.current.data);
    return cudaGetLastError();
}
}  // namespace eclair::cuda
