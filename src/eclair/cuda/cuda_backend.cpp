/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/cuda_backend.hpp"

#include "eclair/cuda/architecture.hpp"
#include "eclair/cuda/bf16_convert.hpp"
#include "eclair/cuda/bf16_linear.hpp"
#include "eclair/cuda/capabilities.hpp"
#include "eclair/cuda/fp8_convert.hpp"
#include "eclair/cuda/fp8_linear.hpp"
#include "eclair/cuda/gated_delta.hpp"
#include "eclair/cuda/gated_delta_output.hpp"
#include "eclair/cuda/paged_attention.hpp"
#include "eclair/cuda/ray_trace.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>
#include <utility>

#ifndef ECLAIR_CUDA_BUILD_ARCHITECTURES
#error "ECLAIR_CUDA_BUILD_ARCHITECTURES must name the configured CUDA architectures"
#endif

namespace eclair::cuda {
namespace {

[[nodiscard]] std::string describe(cudaError_t status) {
    return std::string(cudaGetErrorName(status)) + ", " + cudaGetErrorString(status);
}

[[nodiscard]] KernelImage kernel_image(cudaError_t probe_status) {
    if (probe_status == cudaSuccess) {
        return {.loadable = true, .error = {}};
    }
    return {.loadable = false, .error = cudaGetErrorName(probe_status)};
}

// CUDA encodes versions as 1000 * major + 10 * minor.
[[nodiscard]] std::string version_text(int encoded) {
    return std::to_string(encoded / 1000) + "." + std::to_string((encoded % 1000) / 10);
}

[[nodiscard]] std::string toolchain_summary() {
    std::string summary;
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess && driver_version > 0) {
        summary += "driver API " + version_text(driver_version);
    } else {
        summary += "no CUDA driver";
    }
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        summary += ", runtime " + version_text(runtime_version);
    }
    summary += ", kernels for sm " ECLAIR_CUDA_BUILD_ARCHITECTURES;
    return summary;
}

[[nodiscard]] capability::DeviceReport probe_device(int device_index) {
    capability::DeviceReport device;
    device.id = std::string(kBackendName) + ":" + std::to_string(device_index);

    DeviceCapabilities properties{};
    const cudaError_t status = query_device_capabilities(device_index, &properties);
    if (status != cudaSuccess) {
        device.name = "device query failed: " + describe(status);
        static_cast<void>(cudaGetLastError());
        return device;
    }

    const ComputeCapability capability{properties.compute_major, properties.compute_minor};
    KernelProbe probe;
    probe.fp8_decode = kernel_image(check_fp8_to_fp16_image(device_index));
    probe.fp8_encode = kernel_image(check_fp8_encode_image(device_index));
    probe.bf16_convert = kernel_image(check_bf16_convert_image(device_index));
    // The tensor-core kernels are validated on sm_70 only, so other devices are
    // not probed for them and report the operations as not implemented.
    if (properties.compute_major == 7 && properties.compute_minor == 0) {
        probe.fp8_linear = kernel_image(check_fp8_linear_image(device_index));
        probe.bf16_linear = kernel_image(check_bf16_linear_image(device_index));
        probe.paged_prefill = kernel_image(check_paged_prefill_image(device_index));
        probe.paged_decode = kernel_image(check_paged_decode_image(device_index));
        probe.gated_delta = kernel_image(check_gated_delta_image(device_index));
        probe.gated_delta_output = kernel_image(check_gated_delta_output_image(device_index));
        probe.ray_trace = kernel_image(check_ray_trace_image(device_index));
    }

    device.name = properties.name;
    device.architecture = architecture_name(capability);
    device.architecture_family = std::string(architecture_family(capability));
    device.memory_bytes = static_cast<std::uint64_t>(properties.global_memory_bytes);
    if (properties.multiprocessor_count > 0) {
        device.compute_units = static_cast<std::uint32_t>(properties.multiprocessor_count);
    }
    device.hardware = hardware_units(capability);
    device.features = evaluate_features(capability, probe);
    return device;
}

}  // namespace

capability::BackendReport probe_backend() {
    capability::BackendReport report;
    report.name = std::string(kBackendName);
    report.label = "NVIDIA GPU";

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        report.state = capability::BackendState::Unavailable;
        report.detail = describe(status) + " (" + toolchain_summary() + ")";
        static_cast<void>(cudaGetLastError());
        return report;
    }
    if (device_count <= 0) {
        report.state = capability::BackendState::Unavailable;
        report.detail = "no CUDA devices (" + toolchain_summary() + ")";
        return report;
    }

    report.state = capability::BackendState::Available;
    report.detail = toolchain_summary();
    for (int device_index = 0; device_index < device_count; ++device_index) {
        report.devices.push_back(probe_device(device_index));
    }
    return report;
}

}  // namespace eclair::cuda
