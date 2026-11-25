/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "eclair/capability/feature.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace eclair::capability {

class FeaturePolicy;

// What Eclair can do with a feature on one device right now.
enum class FeatureState : std::uint8_t {
    Enabled,         // An Eclair implementation exists and passed its probe.
    Disabled,        // Enabled, but turned off by policy.
    NotImplemented,  // No implementation for this device.
    Unsupported,     // No Eclair path exists for this device or build.
    NotApplicable,   // Meaningless here, for example emulating present hardware.
};

inline constexpr std::size_t kFeatureStateCount = 5U;

// How an available implementation executes. Unimplemented features use None.
enum class ExecutionPath : std::uint8_t {
    None,
    Hardware,       // Dedicated hardware units.
    Emulated,       // Device software built from other instructions.
    HostReference,  // Portable C++ on the host.
};

// Dedicated units that matter for Eclair's numerical formats. Absence of a
// unit is a fact about the device, not an Eclair limitation.
enum class HardwareUnit : std::uint8_t {
    Fp16TensorCores,
    Int8TensorCores,
    Tf32TensorCores,
    Bf16Arithmetic,
    Fp8TensorCores,
    RayTracingCores,
};

inline constexpr std::size_t kHardwareUnitCount = 6U;

enum class HardwareSupport : std::uint8_t {
    Present,
    Absent,
    Unknown,  // The architecture alone does not decide it.
};

struct FeatureStatus {
    Feature feature = Feature::Fp8E4m3fnDecode;
    FeatureState state = FeatureState::Unsupported;
    ExecutionPath path = ExecutionPath::None;
    std::string detail;
};

struct HardwareUnitStatus {
    HardwareUnit unit = HardwareUnit::Fp16TensorCores;
    HardwareSupport support = HardwareSupport::Unknown;
};

struct DeviceReport {
    std::string id;  // Backend-qualified, for example "cuda:0".
    std::string name;
    std::string architecture;         // Backend spelling, for example "sm_70".
    std::string architecture_family;  // For example "Volta". May be empty.
    std::optional<std::uint64_t> memory_bytes;
    std::optional<std::uint32_t> compute_units;
    std::vector<HardwareUnitStatus> hardware;
    std::vector<FeatureStatus> features;
};

enum class BackendState : std::uint8_t {
    Available,
    Unavailable,  // Built in, but the probe failed or found no device.
    NotBuilt,     // Excluded at configure time.
};

struct BackendReport {
    std::string name;
    BackendState state = BackendState::Unavailable;
    std::string detail;
    std::vector<DeviceReport> devices;
    // Machine identity stays in name; labels and role explain the backend to
    // people without teaching vendor-specific names to the capability layer.
    std::string label;
    bool is_reference = false;
};

[[nodiscard]] std::string_view to_string(FeatureState state) noexcept;
[[nodiscard]] std::string_view to_string(ExecutionPath path) noexcept;
[[nodiscard]] std::string_view to_string(HardwareUnit unit) noexcept;
[[nodiscard]] std::string_view to_string(HardwareSupport support) noexcept;
[[nodiscard]] std::string_view to_string(BackendState state) noexcept;

// Short display label for a hardware unit, such as "FP16 TC".
[[nodiscard]] std::string_view display_label(HardwareUnit unit) noexcept;

// Turns each enabled feature the policy disables into Disabled. Other states
// are left alone: a policy cannot enable or un-plan anything.
void apply_policy(const FeaturePolicy& policy, BackendReport& report);

}  // namespace eclair::capability
