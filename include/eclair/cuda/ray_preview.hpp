/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/render/denoise.hpp"
#include <cuda_runtime_api.h>

namespace eclair::cuda {
struct LightingBuffers {
    ops::BufferView<render::SurfaceLighting> current;
    ops::BufferView<const render::SurfaceLighting> previous;
    ops::BufferView<render::Vec3> scratch;
};
// Supplying LightingBuffers enables object-space diffuse lighting and split
// Fresnel specular integration. Each atlas frame holds
// lighting_layout(scene).tiles * kLightingTilePixels elements (at least one
// allocated element for fully specular scenes); its scratch holds twice that
// many Vec3s. Swap both histories together. All six allocations and rgba must
// be disjoint. Direct atlas lighting updates each frame, while indirect light
// accumulates in object coordinates. Topology/material/light changes require
// valid_history=false, including changes that retain the same tile count.
// Experimental spatial/temporal preview, separate from the unfiltered oracle.
// All device buffers are caller-owned, disjoint, and valid until stream completion.
// current/previous each hold width*height PreviewPixels; scratch holds twice that many
// Vec3s. Scratch contents are unspecified and temporarily reused for an integer
// boundary queue after filtering. Diffuse history contains unmixed surface lighting;
// specular history contains pixel radiance. RGBA resolves boundary coverage separately.
// Swap history frames only after completion. Set
// valid_history=false on first use, camera cuts, scene topology/material/light changes or mode
// changes. Diffuse rigid-box/camera motion is reprojected; specular history requires identity of
// the entire scene and camera. Without an atlas, rough conductors use the fine
// spatial pair; with one, broad lobes use multiscale filtering, glossy lobes
// use strides 1/2, and glass uses a one-pixel median pair. No
// allocation or synchronization; rgba remains the only PCIe image payload.
[[nodiscard]] cudaError_t launch_ray_preview(
    const ops::RayScene& scene, const ops::RayCamera& camera, const ops::RayScene& previous_scene,
    const ops::RayCamera& previous_camera, const ops::RayTraceShape& shape, bool valid_history,
    ops::BufferView<render::PreviewPixel> current,
    ops::BufferView<const render::PreviewPixel> previous, ops::BufferView<render::Vec3> scratch,
    ops::BufferView<std::uint8_t> rgba, cudaStream_t stream,
    LightingBuffers lighting = {}) noexcept;
}  // namespace eclair::cuda
