/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "eclair/capability/feature.hpp"
#include "eclair/capability/policy.hpp"
#include "eclair/capability/report.hpp"
#include "eclair/cuda/architecture.hpp"
#include "eclair/host/host_backend.hpp"

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {

using eclair::capability::ExecutionPath;
using eclair::capability::Feature;
using eclair::capability::FeaturePolicy;
using eclair::capability::FeatureState;
using eclair::capability::FeatureStatus;
using eclair::capability::HardwareSupport;
using eclair::capability::HardwareUnit;
using eclair::cuda::ComputeCapability;
using eclair::cuda::KernelImage;
using eclair::cuda::KernelProbe;

[[nodiscard]] const FeatureStatus& find(const std::vector<FeatureStatus>& features,
                                        Feature feature) {
    for (const FeatureStatus& status : features) {
        if (status.feature == feature) {
            return status;
        }
    }
    static const FeatureStatus kMissing{};
    ECLAIR_CHECK(false && "feature missing from report");
    return kMissing;
}

[[nodiscard]] HardwareSupport unit(ComputeCapability capability, HardwareUnit wanted) {
    for (const auto& status : eclair::cuda::hardware_units(capability)) {
        if (status.unit == wanted) {
            return status.support;
        }
    }
    ECLAIR_CHECK(false && "hardware unit missing from report");
    return HardwareSupport::Unknown;
}

[[nodiscard]] KernelProbe loadable() {
    KernelProbe probe;
    probe.fp8_decode.loadable = true;
    probe.fp8_encode.loadable = true;
    probe.fp8_linear.loadable = true;
    probe.bf16_convert.loadable = true;
    probe.bf16_linear.loadable = true;
    probe.paged_prefill.loadable = true;
    probe.paged_decode.loadable = true;
    probe.gated_delta.loadable = true;
    probe.gated_delta_output.loadable = true;
    probe.ray_trace.loadable = true;
    return probe;
}

[[nodiscard]] bool mentions(const FeatureStatus& status, std::string_view text) {
    return status.detail.find(text) != std::string::npos;
}

void check_catalog() {
    const auto catalog = eclair::capability::feature_catalog();
    ECLAIR_CHECK(catalog.size() == eclair::capability::kFeatureCount);
    std::set<std::string_view> keys;
    for (const auto& info : catalog) {
        ECLAIR_CHECK(keys.insert(info.key).second);
        ECLAIR_CHECK(!info.title.empty());
        ECLAIR_CHECK(!info.summary.empty());
        ECLAIR_CHECK(eclair::capability::parse_feature_key(info.key) == info.feature);
        ECLAIR_CHECK(&eclair::capability::feature_info(info.feature) == &info);
    }
    ECLAIR_CHECK(!eclair::capability::parse_feature_key("fp8"));
    ECLAIR_CHECK(!eclair::capability::parse_feature_key("FP8.GEMM"));
    ECLAIR_CHECK(!eclair::capability::parse_feature_key(""));
}

void check_policy() {
    FeaturePolicy policy;
    ECLAIR_CHECK(policy.empty());
    ECLAIR_CHECK(!policy.disable_list(" fp8.gemm , ,bf16.compat "));
    ECLAIR_CHECK(policy.is_disabled(Feature::Fp8Gemm));
    ECLAIR_CHECK(policy.is_disabled(Feature::Bf16Compat));
    ECLAIR_CHECK(!policy.is_disabled(Feature::Fp8E4m3fnDecode));

    FeaturePolicy group;
    ECLAIR_CHECK(!group.disable_list("fp8.e4m3fn.*"));
    ECLAIR_CHECK(group.is_disabled(Feature::Fp8E4m3fnDecode));
    ECLAIR_CHECK(group.is_disabled(Feature::Fp8E4m3fnEncode));
    ECLAIR_CHECK(!group.is_disabled(Feature::Fp8E5m2Decode));
    ECLAIR_CHECK(!group.disable_list("fp8.*"));
    ECLAIR_CHECK(group.is_disabled(Feature::Fp8Gemm));
    ECLAIR_CHECK(!group.is_disabled(Feature::RayTracingEmulation));

    // A rejected list changes nothing, even when earlier entries were valid.
    FeaturePolicy rejected;
    const std::optional<std::string> error = rejected.disable_list("fp8.gemm,fp9.*");
    ECLAIR_CHECK(error && error->find("fp9.*") != std::string::npos);
    ECLAIR_CHECK(rejected.empty());
    ECLAIR_CHECK(rejected.disable_list(".*"));
    ECLAIR_CHECK(rejected.disable_list("fp8"));
    ECLAIR_CHECK(rejected.disable_list("*"));
    ECLAIR_CHECK(rejected.empty());
    ECLAIR_CHECK(!rejected.disable_list(""));
    ECLAIR_CHECK(rejected.empty());
}

void check_apply_policy() {
    eclair::capability::BackendReport report;
    report.devices.push_back({});
    report.devices[0].features = {
        {Feature::Fp8E4m3fnDecode, FeatureState::Enabled, ExecutionPath::Emulated, "decode"},
        {Feature::Fp8Gemm, FeatureState::NotImplemented, ExecutionPath::Emulated,
         "not implemented"},
        {Feature::Fp8E5m2Decode, FeatureState::Enabled, ExecutionPath::Emulated, "kept"},
    };
    FeaturePolicy policy;
    policy.disable(Feature::Fp8E4m3fnDecode);
    policy.disable(Feature::Fp8Gemm);
    eclair::capability::apply_policy(policy, report);
    const auto& features = report.devices[0].features;
    ECLAIR_CHECK(features[0].state == FeatureState::Disabled);
    ECLAIR_CHECK(features[0].detail.find("policy") != std::string::npos);
    // Policy cannot turn a unimplemented feature into a disabled one.
    ECLAIR_CHECK(features[1].state == FeatureState::NotImplemented);
    ECLAIR_CHECK(features[2].state == FeatureState::Enabled);
}

void check_volta() {
    constexpr ComputeCapability kVolta{7, 0};
    ECLAIR_CHECK(eclair::cuda::architecture_name(kVolta) == "sm_70");
    ECLAIR_CHECK(eclair::cuda::architecture_family(kVolta) == "Volta");
    ECLAIR_CHECK(unit(kVolta, HardwareUnit::Fp16TensorCores) == HardwareSupport::Present);
    ECLAIR_CHECK(unit(kVolta, HardwareUnit::Int8TensorCores) == HardwareSupport::Absent);
    ECLAIR_CHECK(unit(kVolta, HardwareUnit::Tf32TensorCores) == HardwareSupport::Absent);
    ECLAIR_CHECK(unit(kVolta, HardwareUnit::Bf16Arithmetic) == HardwareSupport::Absent);
    ECLAIR_CHECK(unit(kVolta, HardwareUnit::Fp8TensorCores) == HardwareSupport::Absent);
    ECLAIR_CHECK(unit(kVolta, HardwareUnit::RayTracingCores) == HardwareSupport::Absent);

    const auto features = eclair::cuda::evaluate_features(kVolta, loadable());
    ECLAIR_CHECK(features.size() == eclair::capability::kFeatureCount);
    for (std::size_t index = 0; index < features.size(); ++index) {
        ECLAIR_CHECK(features[index].feature ==
                     eclair::capability::feature_catalog()[index].feature);
        ECLAIR_CHECK(!features[index].detail.empty());
    }
    const FeatureStatus& decode = find(features, Feature::Fp8E4m3fnDecode);
    ECLAIR_CHECK(decode.state == FeatureState::Enabled);
    ECLAIR_CHECK(decode.path == ExecutionPath::Emulated);
    ECLAIR_CHECK(find(features, Feature::Fp8E5m2Decode).state == FeatureState::Enabled);
    for (const Feature encode : {Feature::Fp8E4m3fnEncode, Feature::Fp8E5m2Encode}) {
        const FeatureStatus& status = find(features, encode);
        ECLAIR_CHECK(status.state == FeatureState::Enabled);
        ECLAIR_CHECK(status.path == ExecutionPath::Emulated);
        ECLAIR_CHECK(mentions(status, "no FP8 hardware"));
    }
    const FeatureStatus& gemm = find(features, Feature::Fp8Gemm);
    ECLAIR_CHECK(gemm.state == FeatureState::Enabled);
    ECLAIR_CHECK(gemm.path == ExecutionPath::Emulated);
    const FeatureStatus& bf16 = find(features, Feature::Bf16Compat);
    ECLAIR_CHECK(bf16.state == FeatureState::Enabled);
    ECLAIR_CHECK(bf16.path == ExecutionPath::Emulated);
    ECLAIR_CHECK(mentions(bf16, "no BF16 hardware"));
    const FeatureStatus& bf16_gemm = find(features, Feature::Bf16Gemm);
    ECLAIR_CHECK(bf16_gemm.state == FeatureState::Enabled);
    ECLAIR_CHECK(bf16_gemm.path == ExecutionPath::Emulated);
    ECLAIR_CHECK(mentions(bf16_gemm, "FP16 tensor cores"));
    // Volta has no int8 tensor cores, so the int8 cache is widened for the FP16
    // ones. The Gated DeltaNet state pass is FP16 and FP32 math the units run.
    for (const Feature attention :
         {Feature::PagedAttentionPrefill, Feature::PagedAttentionDecode}) {
        const FeatureStatus& status = find(features, attention);
        ECLAIR_CHECK(status.state == FeatureState::Enabled);
        ECLAIR_CHECK(status.path == ExecutionPath::Emulated);
        ECLAIR_CHECK(mentions(status, "no int8 tensor cores"));
    }
    const FeatureStatus& gated_delta = find(features, Feature::GatedDeltaStates);
    ECLAIR_CHECK(gated_delta.state == FeatureState::Enabled);
    ECLAIR_CHECK(gated_delta.path == ExecutionPath::Hardware);
    ECLAIR_CHECK(find(features, Feature::GatedDeltaOutput).state == FeatureState::Enabled);
    ECLAIR_CHECK(find(features, Feature::GatedDeltaOutput).path == ExecutionPath::Hardware);
    const FeatureStatus& rt = find(features, Feature::RayTracingEmulation);
    ECLAIR_CHECK(rt.state == FeatureState::Enabled);
    ECLAIR_CHECK(rt.path == ExecutionPath::Emulated);
}

// Negative controls: each kernel family's missing image disables exactly the
// features it backs, and the detail carries the CUDA error.
void check_missing_kernel_image() {
    constexpr std::string_view kNoImage = "cudaErrorNoKernelImageForDevice";
    auto no_ray = loadable();
    no_ray.ray_trace = {false, std::string(kNoImage)};
    const auto no_ray_features = eclair::cuda::evaluate_features({7, 0}, no_ray);
    ECLAIR_CHECK(find(no_ray_features, Feature::RayTracingEmulation).state ==
                 FeatureState::Unsupported);
    ECLAIR_CHECK(mentions(find(no_ray_features, Feature::RayTracingEmulation), kNoImage));
    ECLAIR_CHECK(find(no_ray_features, Feature::Fp8Gemm).state == FeatureState::Enabled);
    KernelProbe probe;
    probe.fp8_decode.error = kNoImage;
    const auto features = eclair::cuda::evaluate_features({7, 0}, probe);
    const FeatureStatus& decode = find(features, Feature::Fp8E4m3fnDecode);
    ECLAIR_CHECK(decode.state == FeatureState::Unsupported);
    ECLAIR_CHECK(decode.path == ExecutionPath::None);
    ECLAIR_CHECK(mentions(decode, kNoImage));
    ECLAIR_CHECK(find(features, Feature::Fp8Gemm).state == FeatureState::Unsupported);
    ECLAIR_CHECK(find(features, Feature::Bf16Compat).state == FeatureState::Unsupported);

    probe = loadable();
    probe.fp8_linear = {.loadable = false, .error = std::string(kNoImage)};
    const auto no_linear = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(find(no_linear, Feature::Fp8E4m3fnDecode).state == FeatureState::Enabled);
    ECLAIR_CHECK(find(no_linear, Feature::Fp8Gemm).state == FeatureState::Unsupported);
    ECLAIR_CHECK(find(no_linear, Feature::Bf16Gemm).state == FeatureState::Enabled);

    probe = loadable();
    probe.bf16_linear = {.loadable = false, .error = std::string(kNoImage)};
    const auto no_bf16_linear = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(find(no_bf16_linear, Feature::Bf16Gemm).state == FeatureState::Unsupported);
    ECLAIR_CHECK(find(no_bf16_linear, Feature::Bf16Gemm).path == ExecutionPath::None);
    ECLAIR_CHECK(mentions(find(no_bf16_linear, Feature::Bf16Gemm), kNoImage));
    ECLAIR_CHECK(find(no_bf16_linear, Feature::Bf16Compat).state == FeatureState::Enabled);
    ECLAIR_CHECK(find(no_bf16_linear, Feature::Fp8Gemm).state == FeatureState::Enabled);

    probe = loadable();
    probe.fp8_encode = {.loadable = false, .error = std::string(kNoImage)};
    const auto no_encode = eclair::cuda::evaluate_features({7, 0}, probe);
    for (const Feature encode : {Feature::Fp8E4m3fnEncode, Feature::Fp8E5m2Encode}) {
        ECLAIR_CHECK(find(no_encode, encode).state == FeatureState::Unsupported);
        ECLAIR_CHECK(mentions(find(no_encode, encode), kNoImage));
    }
    ECLAIR_CHECK(find(no_encode, Feature::Fp8E5m2Decode).state == FeatureState::Enabled);
    ECLAIR_CHECK(find(no_encode, Feature::Bf16Compat).state == FeatureState::Enabled);

    probe = loadable();
    probe.bf16_convert = {.loadable = false, .error = std::string(kNoImage)};
    const auto no_bf16 = eclair::cuda::evaluate_features({7, 0}, probe);
    ECLAIR_CHECK(find(no_bf16, Feature::Bf16Compat).state == FeatureState::Unsupported);
    ECLAIR_CHECK(find(no_bf16, Feature::Bf16Compat).path == ExecutionPath::None);
    ECLAIR_CHECK(find(no_bf16, Feature::Fp8E4m3fnEncode).state == FeatureState::Enabled);

    // Each model kernel's image backs exactly its own feature.
    struct Kernel {
        KernelImage KernelProbe::* image;
        Feature feature;
    };
    for (const Kernel kernel :
         {Kernel{&KernelProbe::paged_prefill, Feature::PagedAttentionPrefill},
          Kernel{&KernelProbe::paged_decode, Feature::PagedAttentionDecode},
          Kernel{&KernelProbe::gated_delta, Feature::GatedDeltaStates},
          Kernel{&KernelProbe::gated_delta_output, Feature::GatedDeltaOutput}}) {
        probe = loadable();
        probe.*kernel.image = {.loadable = false, .error = std::string(kNoImage)};
        const auto missing = eclair::cuda::evaluate_features({7, 0}, probe);
        for (const Feature feature : {Feature::PagedAttentionPrefill, Feature::PagedAttentionDecode,
                                      Feature::GatedDeltaStates, Feature::GatedDeltaOutput}) {
            const FeatureStatus& status = find(missing, feature);
            if (feature == kernel.feature) {
                ECLAIR_CHECK(status.state == FeatureState::Unsupported);
                ECLAIR_CHECK(status.path == ExecutionPath::None);
                ECLAIR_CHECK(mentions(status, kNoImage));
            } else {
                ECLAIR_CHECK(status.state == FeatureState::Enabled);
            }
        }
    }
}

void check_other_architectures() {
    ECLAIR_CHECK(eclair::cuda::architecture_family({7, 5}) == "Turing");
    ECLAIR_CHECK(unit({7, 5}, HardwareUnit::Int8TensorCores) == HardwareSupport::Present);
    // GTX 16-series parts share sm_75 with RTX parts but have no RT cores.
    ECLAIR_CHECK(unit({7, 5}, HardwareUnit::RayTracingCores) == HardwareSupport::Unknown);

    ECLAIR_CHECK(unit({8, 0}, HardwareUnit::Bf16Arithmetic) == HardwareSupport::Present);
    ECLAIR_CHECK(unit({8, 0}, HardwareUnit::RayTracingCores) == HardwareSupport::Absent);
    // The conversion runs in software everywhere. With BF16 units present the
    // detail says they are unused, never that they do the work.
    const FeatureStatus& ampere_bf16 =
        find(eclair::cuda::evaluate_features({8, 0}, loadable()), Feature::Bf16Compat);
    ECLAIR_CHECK(ampere_bf16.path == ExecutionPath::Emulated);
    ECLAIR_CHECK(mentions(ampere_bf16, "BF16 units not used"));
    // The BF16 linear kernels are validated on sm_70 only.
    const FeatureStatus& ampere_bf16_gemm =
        find(eclair::cuda::evaluate_features({8, 0}, loadable()), Feature::Bf16Gemm);
    ECLAIR_CHECK(ampere_bf16_gemm.state == FeatureState::NotImplemented);
    ECLAIR_CHECK(ampere_bf16_gemm.path == ExecutionPath::None);
    ECLAIR_CHECK(mentions(ampere_bf16_gemm, "sm_70 only"));
    // So are the model kernels, even when an image would load.
    const auto ampere = eclair::cuda::evaluate_features({8, 0}, loadable());
    for (const Feature feature : {Feature::PagedAttentionPrefill, Feature::PagedAttentionDecode,
                                  Feature::GatedDeltaStates, Feature::GatedDeltaOutput}) {
        ECLAIR_CHECK(find(ampere, feature).state == FeatureState::NotImplemented);
        ECLAIR_CHECK(find(ampere, feature).path == ExecutionPath::None);
        ECLAIR_CHECK(mentions(find(ampere, feature), "sm_70 only"));
    }

    ECLAIR_CHECK(eclair::cuda::architecture_family({8, 6}) == "Ampere");
    ECLAIR_CHECK(unit({8, 6}, HardwareUnit::RayTracingCores) == HardwareSupport::Present);
    ECLAIR_CHECK(unit({8, 6}, HardwareUnit::Fp8TensorCores) == HardwareSupport::Absent);
    ECLAIR_CHECK(
        find(eclair::cuda::evaluate_features({8, 6}, loadable()), Feature::RayTracingEmulation)
            .state == FeatureState::NotApplicable);

    ECLAIR_CHECK(eclair::cuda::architecture_family({8, 9}) == "Ada Lovelace");
    ECLAIR_CHECK(unit({8, 9}, HardwareUnit::Fp8TensorCores) == HardwareSupport::Present);
    const auto ada = eclair::cuda::evaluate_features({8, 9}, loadable());
    ECLAIR_CHECK(find(ada, Feature::Fp8Gemm).path == ExecutionPath::None);
    ECLAIR_CHECK(mentions(find(ada, Feature::Fp8E4m3fnEncode), "FP8 units not used"));

    ECLAIR_CHECK(unit({9, 0}, HardwareUnit::Fp8TensorCores) == HardwareSupport::Present);
    ECLAIR_CHECK(unit({9, 0}, HardwareUnit::RayTracingCores) == HardwareSupport::Absent);
    ECLAIR_CHECK(eclair::cuda::architecture_name({12, 0}) == "sm_120");
    ECLAIR_CHECK(eclair::cuda::architecture_family({12, 0}) == "Blackwell");

    // Pascal predates tensor cores, so the FP8 linear path cannot apply.
    ECLAIR_CHECK(eclair::cuda::architecture_family({6, 1}) == "Pascal");
    ECLAIR_CHECK(unit({6, 1}, HardwareUnit::Fp16TensorCores) == HardwareSupport::Absent);
    ECLAIR_CHECK(
        find(eclair::cuda::evaluate_features({6, 1}, loadable()), Feature::Fp8Gemm).state ==
        FeatureState::Unsupported);
    ECLAIR_CHECK(
        find(eclair::cuda::evaluate_features({6, 1}, loadable()), Feature::Bf16Gemm).state ==
        FeatureState::Unsupported);
    const auto pascal = eclair::cuda::evaluate_features({6, 1}, loadable());
    for (const Feature feature : {Feature::PagedAttentionPrefill, Feature::PagedAttentionDecode,
                                  Feature::GatedDeltaStates, Feature::GatedDeltaOutput}) {
        ECLAIR_CHECK(find(pascal, feature).state == FeatureState::Unsupported);
        ECLAIR_CHECK(mentions(find(pascal, feature), "FP16 tensor cores"));
    }
    ECLAIR_CHECK(eclair::cuda::architecture_family({99, 0}).empty());
}

void check_host_backend() {
    const auto report = eclair::host::probe_backend();
    ECLAIR_CHECK(report.name == "host");
    ECLAIR_CHECK(report.state == eclair::capability::BackendState::Available);
    ECLAIR_CHECK(report.devices.size() == 1U);
    const auto& features = report.devices[0].features;
    ECLAIR_CHECK(features.size() == eclair::capability::kFeatureCount);
    // Each Ready state needs its reference to give the contract's known answer.
    for (const Feature feature :
         {Feature::Fp8E4m3fnDecode, Feature::Fp8E5m2Decode, Feature::Fp8E4m3fnEncode,
          Feature::Fp8E5m2Encode, Feature::Bf16Compat, Feature::Bf16Gemm,
          Feature::PagedAttentionPrefill, Feature::PagedAttentionDecode, Feature::GatedDeltaStates,
          Feature::GatedDeltaOutput}) {
        const FeatureStatus& status = find(features, feature);
        ECLAIR_CHECK(status.state == FeatureState::Enabled);
        ECLAIR_CHECK(status.path == ExecutionPath::HostReference);
    }
    ECLAIR_CHECK(find(features, Feature::Fp8Gemm).state == FeatureState::Enabled);
    ECLAIR_CHECK(find(features, Feature::RayTracingEmulation).state == FeatureState::Enabled);
    ECLAIR_CHECK(find(features, Feature::RayTracingEmulation).path == ExecutionPath::HostReference);
}

void check_spellings() {
    ECLAIR_CHECK(eclair::capability::to_string(FeatureState::NotApplicable) == "n/a");
    ECLAIR_CHECK(eclair::capability::to_string(ExecutionPath::HostReference) == "host-reference");
    ECLAIR_CHECK(eclair::capability::to_string(HardwareUnit::Fp8TensorCores) == "fp8_tensor_cores");
    ECLAIR_CHECK(eclair::capability::to_string(eclair::capability::BackendState::NotBuilt) ==
                 "not-built");
}

}  // namespace

int main() {
    check_catalog();
    check_policy();
    check_apply_policy();
    check_volta();
    check_missing_kernel_image();
    check_other_architectures();
    check_host_backend();
    check_spellings();
    return eclair::test::finish();
}
