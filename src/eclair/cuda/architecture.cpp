/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "eclair/cuda/architecture.hpp"

#include "eclair/capability/feature.hpp"

#include <iterator>
#include <string>
#include <utility>

namespace eclair::cuda {
namespace {

using capability::ExecutionPath;
using capability::Feature;
using capability::FeatureState;
using capability::FeatureStatus;
using capability::HardwareSupport;
using capability::HardwareUnit;

[[nodiscard]] constexpr int encode(ComputeCapability capability) noexcept {
    return capability.major * 10 + capability.minor;
}

[[nodiscard]] HardwareSupport present_if(bool present) noexcept {
    return present ? HardwareSupport::Present : HardwareSupport::Absent;
}

// RT cores vary by product rather than by capability on some architectures.
// GTX 16-series Turing parts are sm_75 without RT cores, for example.
[[nodiscard]] HardwareSupport ray_tracing_cores(int version) noexcept {
    switch (version) {
        case 86:   // GA10x
        case 89:   // AD10x
        case 120:  // GB20x
            return HardwareSupport::Present;
        case 70:  // GV100
        case 72:  // Xavier
        case 80:  // GA100
        case 90:  // GH100
            return HardwareSupport::Absent;
        default:
            return version < 70 ? HardwareSupport::Absent : HardwareSupport::Unknown;
    }
}

[[nodiscard]] HardwareSupport unit_support(HardwareUnit unit, int version) noexcept {
    switch (unit) {
        case HardwareUnit::Fp16TensorCores:
            return present_if(version >= 70);
        case HardwareUnit::Int8TensorCores:
            return present_if(version == 72 || version >= 75);
        case HardwareUnit::Tf32TensorCores:
        case HardwareUnit::Bf16Arithmetic:
            return present_if(version >= 80);
        case HardwareUnit::Fp8TensorCores:
            return present_if(version == 89 || version >= 90);
        case HardwareUnit::RayTracingCores:
            return ray_tracing_cores(version);
    }
    return HardwareSupport::Unknown;
}

[[nodiscard]] bool present(HardwareUnit unit, int version) noexcept {
    return unit_support(unit, version) == HardwareSupport::Present;
}

// Format conversions are generic software kernels, enabled on any device that
// can load them. The detail says whether the device has units Eclair leaves
// unused, so the report never implies those units do the work.
[[nodiscard]] FeatureStatus software_conversion(Feature feature, const KernelImage& image,
                                                std::string detail) {
    if (!image.loadable) {
        return {feature, FeatureState::Unsupported, ExecutionPath::None,
                "no loadable kernel image, " + image.error};
    }
    return {feature, FeatureState::Enabled, ExecutionPath::Emulated, std::move(detail)};
}

// Tensor-core linear kernels are validated on sm_70 only. The backend probes
// their images there and nowhere else, so other devices report the operation
// as not implemented rather than guessing.
[[nodiscard]] FeatureStatus volta_linear(Feature feature, int version, const KernelImage& image,
                                         const std::string& name, std::string detail) {
    if (!present(HardwareUnit::Fp16TensorCores, version)) {
        return {feature, FeatureState::Unsupported, ExecutionPath::None,
                name + " linear requires FP16 tensor cores"};
    }
    if (version != 70) {
        return {feature, FeatureState::NotImplemented, ExecutionPath::None,
                name + " linear backend is validated for sm_70 only"};
    }
    if (!image.loadable) {
        return {feature, FeatureState::Unsupported, ExecutionPath::None,
                "no loadable " + name + " linear kernel, " + image.error};
    }
    return {feature, FeatureState::Enabled, ExecutionPath::Emulated, std::move(detail)};
}

// The model kernels (paged attention, the Gated DeltaNet state pass) are
// tensor-core kernels validated on sm_70 only, probed like the linear kernels.
// A path is Emulated when the device lacks units for the operation's number
// format (int8 caches without int8 tensor cores) and Hardware when FP16
// tensor cores and FP32 do the work natively.
[[nodiscard]] FeatureStatus volta_kernel(Feature feature, int version, const KernelImage& image,
                                         const std::string& name, ExecutionPath path,
                                         std::string detail) {
    if (!present(HardwareUnit::Fp16TensorCores, version)) {
        return {feature, FeatureState::Unsupported, ExecutionPath::None,
                name + " requires FP16 tensor cores"};
    }
    if (version != 70) {
        return {feature, FeatureState::NotImplemented, ExecutionPath::None,
                name + " is validated for sm_70 only"};
    }
    if (!image.loadable) {
        return {feature, FeatureState::Unsupported, ExecutionPath::None,
                "no loadable " + name + " kernel, " + image.error};
    }
    return {feature, FeatureState::Enabled, path, std::move(detail)};
}

[[nodiscard]] FeatureStatus evaluate(Feature feature, int version, const KernelProbe& probe) {
    const bool fp8_hardware = present(HardwareUnit::Fp8TensorCores, version);
    switch (feature) {
        case Feature::Fp8E4m3fnDecode:
        case Feature::Fp8E5m2Decode:
            return software_conversion(feature, probe.fp8_decode,
                                       fp8_hardware ? "software decode to FP16, FP8 units not used"
                                                    : "software decode to FP16, no FP8 hardware");
        case Feature::Fp8E4m3fnEncode:
        case Feature::Fp8E5m2Encode:
            return software_conversion(feature, probe.fp8_encode,
                                       fp8_hardware
                                           ? "software encode from FP16 or FP32, FP8 units not used"
                                           : "software encode from FP16 or FP32, no FP8 hardware");
        case Feature::Fp8Gemm:
            return volta_linear(
                feature, version, probe.fp8_linear, "FP8",
                "scaled FP8 linear: FP16 operands, FP32 accumulation; contiguous matrices");
        case Feature::Bf16Compat:
            return software_conversion(
                feature, probe.bf16_convert,
                present(HardwareUnit::Bf16Arithmetic, version)
                    ? "software conversion to and from FP32, BF16 units not used"
                    : "software conversion to and from FP32, no BF16 hardware");
        case Feature::Bf16Gemm:
            return volta_linear(feature, version, probe.bf16_linear, "BF16",
                                "row-scaled BF16, CUDA cores for M<=4, FP16 tensor cores "
                                "otherwise; FP32 accumulation");
        case Feature::PagedAttentionPrefill:
            return volta_kernel(feature, version, probe.paged_prefill, "paged prefill attention",
                                ExecutionPath::Emulated,
                                "int8 cache widened to FP16 for the tensor cores (no int8 "
                                "tensor cores), FP32 softmax");
        case Feature::PagedAttentionDecode:
            return volta_kernel(feature, version, probe.paged_decode, "paged decode attention",
                                ExecutionPath::Emulated,
                                "int8 cache widened to FP16 for the tensor cores (no int8 "
                                "tensor cores), context split into slices and merged in FP32");
        case Feature::GatedDeltaStates:
            return volta_kernel(feature, version, probe.gated_delta, "Gated DeltaNet state pass",
                                ExecutionPath::Hardware, "FP16 tensor cores, FP32 state");
        case Feature::GatedDeltaOutput:
            return volta_kernel(feature, version, probe.gated_delta_output,
                                "Gated DeltaNet output pass", ExecutionPath::Hardware,
                                "FP16 tensor cores, FP32 sums");
        case Feature::RayTracingEmulation:
            if (version == 70) {
                return software_conversion(
                    feature, probe.ray_trace,
                    "bounded analytic scenes, FP32 CUDA-core tracing; no RT cores");
            }
            switch (ray_tracing_cores(version)) {
                case HardwareSupport::Present:
                    return {feature, FeatureState::NotApplicable, ExecutionPath::None,
                            "native RT cores present"};
                case HardwareSupport::Absent:
                    return {feature, FeatureState::NotImplemented, ExecutionPath::None,
                            "ray-tracing backend is validated for sm_70 only"};
                case HardwareSupport::Unknown:
                    return {feature, FeatureState::NotImplemented, ExecutionPath::None,
                            "ray-tracing backend is validated for sm_70 only; hardware support "
                            "unknown"};
            }
            break;
    }
    return {feature, FeatureState::Unsupported, ExecutionPath::None, "unrecognized feature"};
}

}  // namespace

std::string architecture_name(ComputeCapability capability) {
    return "sm_" + std::to_string(encode(capability));
}

std::string_view architecture_family(ComputeCapability capability) noexcept {
    switch (capability.major) {
        case 3:
            return "Kepler";
        case 5:
            return "Maxwell";
        case 6:
            return "Pascal";
        case 7:
            return capability.minor >= 5 ? "Turing" : "Volta";
        case 8:
            return capability.minor == 9 ? "Ada Lovelace" : "Ampere";
        case 9:
            return "Hopper";
        case 10:
        case 11:
        case 12:
            return "Blackwell";
        default:
            return {};
    }
}

std::vector<capability::HardwareUnitStatus> hardware_units(ComputeCapability capability) {
    constexpr HardwareUnit kUnits[] = {
        HardwareUnit::Fp16TensorCores, HardwareUnit::Int8TensorCores, HardwareUnit::Tf32TensorCores,
        HardwareUnit::Bf16Arithmetic,  HardwareUnit::Fp8TensorCores,  HardwareUnit::RayTracingCores,
    };
    static_assert(std::size(kUnits) == capability::kHardwareUnitCount);
    const int version = encode(capability);
    std::vector<capability::HardwareUnitStatus> units;
    units.reserve(std::size(kUnits));
    for (const HardwareUnit unit : kUnits) {
        units.push_back({unit, unit_support(unit, version)});
    }
    return units;
}

std::vector<FeatureStatus> evaluate_features(ComputeCapability capability,
                                             const KernelProbe& probe) {
    const int version = encode(capability);
    std::vector<FeatureStatus> features;
    features.reserve(capability::kFeatureCount);
    for (const capability::FeatureInfo& info : capability::feature_catalog()) {
        features.push_back(evaluate(info.feature, version, probe));
    }
    return features;
}

}  // namespace eclair::cuda
