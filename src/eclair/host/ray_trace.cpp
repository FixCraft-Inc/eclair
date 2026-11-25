/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/host/ray_trace.hpp"

#include "eclair/render/shading.hpp"

namespace eclair::host {
ops::Status ray_trace_reference(const ops::RayScene& scene, const ops::RayCamera& camera,
                                const ops::RayTraceShape& shape,
                                ops::BufferView<std::uint8_t> rgba) noexcept {
    const auto status = ops::validate_ray_trace(scene, camera, shape);
    if (status != ops::Status::Success) return status;
    std::size_t bytes = 0;
    if (ops::ray_frame_bytes(shape, &bytes) != ops::Status::Success)
        return ops::Status::SizeOverflow;
    if (rgba.data == nullptr) return ops::Status::InvalidArgument;
    if (rgba.size < bytes) return ops::Status::BufferTooSmall;
    for (std::uint32_t i = 0; i < shape.width * shape.height; ++i)
        render::ray_pixel(scene, camera, shape, i, rgba.data + static_cast<std::size_t>(i) * 4);
    return ops::Status::Success;
}
}  // namespace eclair::host
