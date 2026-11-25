/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "text_report.hpp"

#include "eclair/capability/feature.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace eclair::smi {
namespace {

using capability::BackendReport;
using capability::BackendState;
using capability::DeviceReport;
using capability::ExecutionPath;
using capability::FeatureState;
using capability::HardwareSupport;

[[nodiscard]] std::string_view label(const BackendReport& backend) noexcept {
    return backend.label.empty() ? std::string_view(backend.name) : std::string_view(backend.label);
}

[[nodiscard]] std::string_view state_word(FeatureState state) noexcept {
    switch (state) {
        case FeatureState::Enabled:
            return "Ready";
        case FeatureState::Disabled:
            return "Disabled";
        case FeatureState::NotImplemented:
            return "Not implemented";
        case FeatureState::Unsupported:
            return "Unavailable";
        case FeatureState::NotApplicable:
            return "Not applicable";
    }
    return "Unknown";
}

[[nodiscard]] Tone tone_for(FeatureState state) noexcept {
    switch (state) {
        case FeatureState::Enabled:
            return Tone::Good;
        case FeatureState::Disabled:
            return Tone::Warning;
        case FeatureState::NotImplemented:
            return Tone::Dim;
        case FeatureState::Unsupported:
            return Tone::Bad;
        case FeatureState::NotApplicable:
            return Tone::Dim;
    }
    return Tone::Plain;
}

[[nodiscard]] std::string_view path_word(ExecutionPath path) noexcept {
    switch (path) {
        case ExecutionPath::None:
            return "";
        case ExecutionPath::Hardware:
            return "native on GPU";
        case ExecutionPath::Emulated:
            return "emulated on GPU";
        case ExecutionPath::HostReference:
            return "CPU reference";
    }
    return "unknown";
}

// Display groups, by key prefix. The catalog itself stays flat.
enum class Group : std::uint8_t { Formats, Kernels, Other };

constexpr std::array<Group, 3> kGroups{Group::Formats, Group::Kernels, Group::Other};

[[nodiscard]] Group group_of(std::string_view key) noexcept {
    if (key.starts_with("fp8.") || key.starts_with("bf16.")) {
        return Group::Formats;
    }
    if (key.starts_with("attention.") || key.starts_with("gated_delta.")) {
        return Group::Kernels;
    }
    return Group::Other;
}

[[nodiscard]] std::string_view group_title(Group group) noexcept {
    switch (group) {
        case Group::Formats:
            return "Number formats";
        case Group::Kernels:
            return "Model kernels";
        case Group::Other:
            return "Other";
    }
    return "Other";
}

// Eclair's own integrations that call an operation, by feature key. These are
// facts about this repository, so they live with the presentation rather than
// in the vendor- and framework-neutral catalog.
[[nodiscard]] std::string_view used_by(std::string_view key) noexcept {
    if (key == "fp8.gemm") {
        return "PyTorch, ComfyUI";
    }
    if (key == "rt.emulation") {
        return "eclair-rt";
    }
    if (key.starts_with("attention.paged_int8.") || key.starts_with("gated_delta.")) {
        return "eclair-vllm";
    }
    return "";
}

[[nodiscard]] std::string format_bytes(std::uint64_t bytes) {
    constexpr double kGibibyte = 1024.0 * 1024.0 * 1024.0;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.1f GiB", static_cast<double>(bytes) / kGibibyte);
    return buffer.data();
}

[[nodiscard]] std::size_t ready_count(const DeviceReport& device) noexcept {
    return static_cast<std::size_t>(std::count_if(device.features.begin(), device.features.end(),
                                                  [](const capability::FeatureStatus& status) {
                                                      return status.state == FeatureState::Enabled;
                                                  }));
}

[[nodiscard]] std::string padded(std::string_view text, std::size_t width) {
    std::string result(text);
    if (result.size() < width) {
        result.append(width - result.size(), ' ');
    }
    return result;
}

// Words of `text` on lines of at most `width` columns, each starting with
// `indent`. A word longer than a line gets a line of its own.
void wrap(std::ostream& out, std::string_view text, std::string_view indent, std::size_t width,
          const Style& style) {
    std::string line;
    const auto flush = [&] {
        if (!line.empty()) {
            out << indent << paint(style, Tone::Dim, line) << '\n';
            line.clear();
        }
    };
    while (!text.empty()) {
        const std::size_t space = text.find(' ');
        const std::string_view word = text.substr(0, space);
        text = space == std::string_view::npos ? std::string_view() : text.substr(space + 1U);
        if (!line.empty() && indent.size() + line.size() + 1U + word.size() > width) {
            flush();
        }
        line += line.empty() ? "" : " ";
        line += word;
    }
    flush();
}

[[nodiscard]] bool shown(const capability::FeatureStatus& status, bool details) noexcept {
    return details || status.state == FeatureState::Enabled ||
           status.state == FeatureState::Disabled;
}

// One table per display group, with columns shared across the device's
// tables: operation, status, how it runs and, for an accelerator, which
// Eclair software calls it. The CPU reference is not what integrations call.
void render_features(std::ostream& out, const DeviceReport& device, const Style& style,
                     bool details, bool accelerator) {
    std::size_t title_width = 0;
    std::size_t state_width = std::string_view("Status").size();
    std::size_t path_width = std::string_view("Runs as").size();
    bool any_user = false;
    std::size_t total = 0;
    for (const capability::FeatureStatus& status : device.features) {
        if (!shown(status, details)) {
            continue;
        }
        const auto& info = capability::feature_info(status.feature);
        title_width =
            std::max({title_width, info.title.size(), group_title(group_of(info.key)).size()});
        state_width = std::max(state_width, state_word(status.state).size());
        path_width = std::max(path_width, path_word(status.path).size());
        any_user = any_user || (accelerator && !used_by(info.key).empty());
        ++total;
    }
    if (total == 0U) {
        out << "\n  No Eclair operations are ready on this device.\n";
        return;
    }
    for (const Group group : kGroups) {
        bool header = false;
        for (const capability::FeatureStatus& status : device.features) {
            const auto& info = capability::feature_info(status.feature);
            if (group_of(info.key) != group || !shown(status, details)) {
                continue;
            }
            if (!header) {
                std::string line = "  " + padded(group_title(group), title_width) + "  " +
                                   padded("Status", state_width) + "  ";
                line += any_user ? padded("Runs as", path_width) + "  Used by" : "Runs as";
                out << '\n' << paint(style, Tone::Bold, line) << '\n';
                header = true;
            }
            const std::string_view word = state_word(status.state);
            const std::string_view path = path_word(status.path);
            const std::string_view user = accelerator ? used_by(info.key) : std::string_view();
            out << "  " << padded(info.title, title_width) << "  "
                << paint(style, tone_for(status.state), word);
            // Pad only up to a cell that follows, so no line ends in spaces.
            if (!path.empty() || !user.empty()) {
                out << std::string(state_width - word.size(), ' ') << "  ";
                out << (user.empty() ? std::string(path)
                                     : padded(path, path_width) + "  " + std::string(user));
            }
            out << '\n';
            if (details) {
                wrap(out, std::string(info.key) + ": " + status.detail, "    ", 78, style);
            }
        }
    }
}

// "FP16 tensor cores present, FP8 tensor cores absent", wrapped at 78 columns
// by the uncolored width when --details lists every unit.
void render_hardware(std::ostream& out, const DeviceReport& device, const Style& style,
                     bool details) {
    std::string line;
    std::size_t width = 0;
    for (const capability::HardwareUnitStatus& unit : device.hardware) {
        if (!details && unit.unit != capability::HardwareUnit::Fp16TensorCores &&
            unit.unit != capability::HardwareUnit::Fp8TensorCores) {
            continue;
        }
        const std::string_view word = unit.support == HardwareSupport::Present  ? "present"
                                      : unit.support == HardwareSupport::Absent ? "absent"
                                                                                : "unknown";
        const std::string name(capability::display_label(unit.unit));
        const std::size_t item = name.size() + 1U + word.size();
        if (width != 0U && width + 2U + item > 78U) {
            out << line << ",\n";
            line.clear();
            width = 0;
        }
        // Each item follows a two-column indent or a ", " separator.
        line += width == 0U ? "  " : ", ";
        width += 2U + item;
        line +=
            name + " " +
            paint(style, unit.support == HardwareSupport::Present ? Tone::Good : Tone::Dim, word);
    }
    if (!line.empty()) {
        out << line << '\n';
    }
}

// "NVIDIA GPU 0  Tesla PG500-216": the backend's label, the index of a
// "backend:N" device ID, and the device name when it adds something.
[[nodiscard]] std::string device_title(const BackendReport& backend, const DeviceReport& device) {
    std::string title(label(backend));
    const std::size_t colon = device.id.rfind(':');
    if (colon != std::string::npos) {
        title += " " + device.id.substr(colon + 1U);
    }
    if (device.name != label(backend)) {
        title += "  " + device.name;
    }
    return title;
}

void render_device(std::ostream& out, const BackendReport& backend, const DeviceReport& device,
                   const Style& style, bool details) {
    out << '\n' << paint(style, Tone::Bold, device_title(backend, device)) << '\n';
    if (backend.is_reference) {
        out << "  Portable checks for numerical correctness, no GPU acceleration.\n";
    } else {
        std::string summary;
        const auto add = [&summary](const std::string& part) {
            summary += (summary.empty() ? "" : ", ") + part;
        };
        if (!device.architecture_family.empty() || !device.architecture.empty()) {
            add(device.architecture_family.empty()
                    ? device.architecture
                    : device.architecture_family + " " + device.architecture);
        }
        if (device.compute_units) {
            add(std::to_string(*device.compute_units) + " compute units");
        }
        if (device.memory_bytes) {
            add(format_bytes(*device.memory_bytes));
        }
        if (!summary.empty()) {
            out << "  " << summary << '\n';
        }
        if (!backend.detail.empty()) {
            out << "  CUDA " << backend.detail << '\n';
        }
    }
    render_hardware(out, device, style, details);
    if (details) {
        out << "  " << paint(style, Tone::Dim, "Device ID: " + device.id) << '\n';
    }
    render_features(out, device, style, details, !backend.is_reference);
}

}  // namespace

void render_report(std::ostream& out, const SystemReport& report, const Style& style,
                   bool details) {
    out << paint(style, Tone::Bold, "Eclair " + report.version) << '\n';
    const bool has_accelerator = std::any_of(
        report.backends.begin(), report.backends.end(), [](const BackendReport& backend) {
            return !backend.is_reference && !backend.devices.empty();
        });
    for (const BackendReport& backend : report.backends) {
        if (backend.state != BackendState::Available) {
            out << '\n'
                << paint(style, Tone::Warning, std::string(label(backend)) + ": unavailable")
                << "\n  " << backend.detail << '\n';
            continue;
        }
        if (backend.is_reference && !details && has_accelerator) {
            for (const DeviceReport& device : backend.devices) {
                out << "\nCPU reference: " << ready_count(device)
                    << " known-answer checks pass (--details lists them).\n";
            }
            continue;
        }
        for (const DeviceReport& device : backend.devices) {
            render_device(out, backend, device, style, details);
        }
    }
    if (!report.disabled_keys.empty()) {
        out << "\nDisabled by " << report.policy_source << ": ";
        for (std::size_t i = 0; i < report.disabled_keys.size(); ++i) {
            out << (i == 0U ? "" : ", ") << report.disabled_keys[i];
        }
        out << '\n';
    }
    out << "\nReady: built in, and the GPU kernel loads (or the CPU check gives the known\n"
        << "answer). Speed, accuracy and model support come from tests, not this report.\n"
        << paint(style, Tone::Dim,
                 "--details: how each one runs   --catalog: what each one does   --json: scripts")
        << '\n';
}

void render_list(std::ostream& out, const SystemReport& report, const Style& style) {
    for (const BackendReport& backend : report.backends) {
        for (const DeviceReport& device : backend.devices) {
            out << paint(style, Tone::Bold, device.id) << "  " << label(backend) << "  "
                << device.name;
            if (device.memory_bytes) {
                out << "  " << format_bytes(*device.memory_bytes);
            }
            out << "  " << ready_count(device) << " operations ready\n";
        }
    }
}

void render_catalog(std::ostream& out, const Style& style) {
    bool first = true;
    for (const Group group : kGroups) {
        bool header = false;
        for (const capability::FeatureInfo& info : capability::feature_catalog()) {
            if (group_of(info.key) != group) {
                continue;
            }
            if (!header) {
                out << (first ? "" : "\n") << paint(style, Tone::Bold, group_title(group)) << '\n';
                header = true;
                first = false;
            }
            out << "  " << info.title << "  (" << info.key << ")\n";
            wrap(out, info.summary, "    ", 78, style);
            const std::string_view user = used_by(info.key);
            if (!user.empty()) {
                out << "    Used by " << user << '\n';
            }
        }
    }
}

}  // namespace eclair::smi
