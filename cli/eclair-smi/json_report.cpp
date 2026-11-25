/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "json_report.hpp"

#include "eclair/capability/feature.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

namespace eclair::smi {
namespace {

// Pretty-printing writer that places commas and indentation. Callers pair
// every begin with its end, and objects alternate key() and a value.
class JsonWriter {
public:
    explicit JsonWriter(std::ostream& out) : out_(out) {}

    void begin_object() { open('{'); }
    void end_object() { close('}'); }
    void begin_array() { open('['); }
    void end_array() { close(']'); }

    void key(std::string_view name) {
        separate();
        out_ << json_quote(name) << ": ";
        after_key_ = true;
    }

    void value(std::string_view text) {
        separate();
        out_ << json_quote(text);
    }

    void value(std::uint64_t number) {
        separate();
        out_ << number;
    }

    void null() {
        separate();
        out_ << "null";
    }

    template <typename Number>
    void optional_value(const std::optional<Number>& number) {
        if (number) {
            value(static_cast<std::uint64_t>(*number));
        } else {
            null();
        }
    }

    void finish() { out_ << '\n'; }

private:
    void open(char bracket) {
        separate();
        out_ << bracket;
        first_.push_back(true);
    }

    void close(char bracket) {
        const bool empty = first_.back();
        first_.pop_back();
        if (!empty) {
            newline();
        }
        out_ << bracket;
    }

    // Emits the comma and line break that precede a key or array element.
    void separate() {
        if (after_key_) {
            after_key_ = false;
            return;
        }
        if (first_.empty()) {
            return;
        }
        if (!first_.back()) {
            out_ << ',';
        }
        first_.back() = false;
        newline();
    }

    void newline() {
        out_ << '\n';
        for (std::size_t depth = 0; depth < first_.size(); ++depth) {
            out_ << "  ";
        }
    }

    std::ostream& out_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

void write_device(JsonWriter& json, const capability::DeviceReport& device) {
    json.begin_object();
    json.key("id");
    json.value(device.id);
    json.key("name");
    json.value(device.name);
    json.key("architecture");
    json.value(device.architecture);
    json.key("architecture_family");
    json.value(device.architecture_family);
    json.key("memory_bytes");
    json.optional_value(device.memory_bytes);
    json.key("compute_units");
    json.optional_value(device.compute_units);
    json.key("hardware");
    json.begin_object();
    for (const capability::HardwareUnitStatus& unit : device.hardware) {
        json.key(capability::to_string(unit.unit));
        json.value(capability::to_string(unit.support));
    }
    json.end_object();
    json.key("features");
    json.begin_array();
    for (const capability::FeatureStatus& status : device.features) {
        json.begin_object();
        json.key("key");
        json.value(capability::feature_info(status.feature).key);
        json.key("state");
        json.value(capability::to_string(status.state));
        json.key("path");
        json.value(capability::to_string(status.path));
        json.key("detail");
        json.value(status.detail);
        json.end_object();
    }
    json.end_array();
    json.end_object();
}

}  // namespace

std::string json_quote(std::string_view text) {
    std::string quoted;
    quoted.reserve(text.size() + 2U);
    quoted.push_back('"');
    for (const char character : text) {
        switch (character) {
            case '"':
                quoted += "\\\"";
                break;
            case '\\':
                quoted += "\\\\";
                break;
            case '\n':
                quoted += "\\n";
                break;
            case '\r':
                quoted += "\\r";
                break;
            case '\t':
                quoted += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(character) < 0x20U) {
                    std::array<char, 8> escape{};
                    std::snprintf(escape.data(), escape.size(), "\\u%04x",
                                  static_cast<unsigned int>(static_cast<unsigned char>(character)));
                    quoted += escape.data();
                } else {
                    quoted.push_back(character);
                }
        }
    }
    quoted.push_back('"');
    return quoted;
}

void render_json_report(std::ostream& out, const SystemReport& report) {
    JsonWriter json(out);
    json.begin_object();
    json.key("schema");
    json.value(kReportSchema);
    json.key("version");
    json.value(report.version);
    json.key("policy");
    json.begin_object();
    json.key("source");
    json.value(report.policy_source);
    json.key("disabled");
    json.begin_array();
    for (const std::string_view key : report.disabled_keys) {
        json.value(key);
    }
    json.end_array();
    json.end_object();
    json.key("backends");
    json.begin_array();
    for (const capability::BackendReport& backend : report.backends) {
        json.begin_object();
        json.key("name");
        json.value(backend.name);
        json.key("label");
        json.value(backend.label);
        json.key("role");
        json.value(backend.is_reference ? "reference" : "accelerator");
        json.key("state");
        json.value(capability::to_string(backend.state));
        json.key("detail");
        json.value(backend.detail);
        json.key("devices");
        json.begin_array();
        for (const capability::DeviceReport& device : backend.devices) {
            write_device(json, device);
        }
        json.end_array();
        json.end_object();
    }
    json.end_array();
    json.end_object();
    json.finish();
}

void render_json_catalog(std::ostream& out) {
    JsonWriter json(out);
    json.begin_object();
    json.key("schema");
    json.value(kCatalogSchema);
    json.key("features");
    json.begin_array();
    for (const capability::FeatureInfo& info : capability::feature_catalog()) {
        json.begin_object();
        json.key("key");
        json.value(info.key);
        json.key("title");
        json.value(info.title);
        json.key("summary");
        json.value(info.summary);
        json.end_object();
    }
    json.end_array();
    json.end_object();
    json.finish();
}

}  // namespace eclair::smi
