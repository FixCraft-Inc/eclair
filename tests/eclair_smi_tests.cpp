/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "check.hpp"

#include "json_report.hpp"
#include "options.hpp"
#include "system_report.hpp"
#include "terminal.hpp"
#include "text_report.hpp"

#include "eclair/capability/policy.hpp"
#include "eclair/cuda/architecture.hpp"
#include "eclair/host/host_backend.hpp"

#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using eclair::smi::ColorMode;
using eclair::smi::Command;
using eclair::smi::ParseResult;

[[nodiscard]] ParseResult parse(std::initializer_list<std::string_view> arguments) {
    const std::vector<std::string_view> list(arguments);
    return eclair::smi::parse_options(list);
}

[[nodiscard]] bool contains(std::string_view text, std::string_view part) {
    return text.find(part) != std::string_view::npos;
}

// The first line of `text` that contains `part`, or an empty view.
[[nodiscard]] std::string_view line_with(std::string_view text, std::string_view part) {
    const std::size_t at = text.find(part);
    if (at == std::string_view::npos) {
        return {};
    }
    const std::size_t begin =
        text.rfind('\n', at) == std::string_view::npos ? 0U : text.rfind('\n', at) + 1U;
    const std::size_t end = text.find('\n', at);
    return text.substr(begin, end == std::string_view::npos ? std::string_view::npos : end - begin);
}

// A table row: the operation's line also shows the given cells.
[[nodiscard]] bool row(std::string_view text, std::string_view operation,
                       std::initializer_list<std::string_view> cells) {
    const std::string_view line = line_with(text, operation);
    if (line.empty()) {
        return false;
    }
    for (const std::string_view cell : cells) {
        if (!contains(line, cell)) {
            return false;
        }
    }
    return true;
}

// A system with one Volta device built from the host-only architecture model,
// so rendering is tested without CUDA.
[[nodiscard]] eclair::smi::SystemReport sample_report() {
    eclair::capability::DeviceReport device;
    device.id = "cuda:0";
    device.name = "Tesla \"PG500\"\t216";
    device.architecture = "sm_70";
    device.architecture_family = "Volta";
    device.memory_bytes = 34072559616U;
    device.compute_units = 72U;
    device.hardware = eclair::cuda::hardware_units({7, 0});
    eclair::cuda::KernelProbe probe;
    probe.fp8_decode.loadable = true;
    probe.fp8_encode.loadable = true;
    probe.fp8_linear.loadable = true;
    probe.bf16_convert.loadable = true;
    probe.bf16_linear.loadable = true;
    probe.paged_prefill.loadable = true;
    probe.paged_decode.loadable = true;
    probe.gated_delta.loadable = true;
    device.features = eclair::cuda::evaluate_features({7, 0}, probe);

    eclair::capability::BackendReport cuda;
    cuda.name = "cuda";
    cuda.label = "NVIDIA GPU";
    cuda.state = eclair::capability::BackendState::Available;
    cuda.detail = "driver API 13.0, runtime 12.9, kernels for sm 70";
    cuda.devices.push_back(device);

    eclair::smi::SystemReport report;
    report.version = "0.0.0-test";
    report.backends.push_back(cuda);
    report.backends.push_back(eclair::host::probe_backend());
    return report;
}

void check_options() {
    ECLAIR_CHECK(!parse({}).error);
    ECLAIR_CHECK(parse({"--details"}).options.details);
    ECLAIR_CHECK(parse({"--details=yes"}).error.has_value());
    ECLAIR_CHECK(parse({}).options.command == Command::Report);
    ECLAIR_CHECK(parse({"-L"}).options.command == Command::List);
    ECLAIR_CHECK(parse({"--catalog", "--json"}).options.json);
    ECLAIR_CHECK(parse({"-i", "cuda:0"}).options.device == "cuda:0");
    ECLAIR_CHECK(parse({"--id=0"}).options.device == "0");
    ECLAIR_CHECK(parse({"--color", "always"}).options.color == ColorMode::Always);
    ECLAIR_CHECK(parse({"--color=never"}).options.color == ColorMode::Never);
    ECLAIR_CHECK(parse({"--no-color"}).options.color == ColorMode::Never);

    const ParseResult disabled = parse({"--disable", "fp8.*", "--disable=bf16.compat"});
    ECLAIR_CHECK(disabled.options.disable_lists.size() == 2U);
    ECLAIR_CHECK(disabled.options.disable_lists[1] == "bf16.compat");

    // Help wins over other commands, and version over reports.
    ECLAIR_CHECK(parse({"--list", "--help"}).options.command == Command::Help);
    ECLAIR_CHECK(parse({"-V"}).options.command == Command::Version);

    for (const auto& arguments :
         {std::vector<std::string_view>{"--bogus"}, std::vector<std::string_view>{"extra"},
          std::vector<std::string_view>{"-i"}, std::vector<std::string_view>{"--id="},
          std::vector<std::string_view>{"--color", "sometimes"},
          std::vector<std::string_view>{"--json=yes"},
          std::vector<std::string_view>{"--list", "--catalog"},
          std::vector<std::string_view>{"--list", "--json"}}) {
        ECLAIR_CHECK(eclair::smi::parse_options(arguments).error.has_value());
    }
    ECLAIR_CHECK(contains(eclair::smi::usage_text(), "--disable"));
}

void check_style() {
    using eclair::smi::StyleMode;
    eclair::smi::TerminalEnvironment terminal;
    terminal.stdout_is_terminal = true;
    terminal.term = "xterm-256color";
    const auto choose = [](StyleMode mode, ColorMode color,
                           const eclair::smi::TerminalEnvironment& environment) {
        return eclair::smi::choose_style(mode, color, environment);
    };

    // Plain is the nvidia-smi-like default: no color or symbols on any terminal.
    ECLAIR_CHECK(!choose(StyleMode::Plain, ColorMode::Auto, terminal).color);
    ECLAIR_CHECK(choose(StyleMode::Plain, ColorMode::Always, terminal).color);
    ECLAIR_CHECK(choose(StyleMode::Pretty, ColorMode::Auto, terminal).color);
    ECLAIR_CHECK(!choose(StyleMode::Pretty, ColorMode::Never, terminal).color);

    eclair::smi::TerminalEnvironment no_color = terminal;
    no_color.no_color = "1";
    ECLAIR_CHECK(!choose(StyleMode::Pretty, ColorMode::Auto, no_color).color);
    ECLAIR_CHECK(choose(StyleMode::Pretty, ColorMode::Always, no_color).color);
    no_color.no_color = "";
    ECLAIR_CHECK(choose(StyleMode::Pretty, ColorMode::Auto, no_color).color);

    eclair::smi::TerminalEnvironment pipe = terminal;
    pipe.stdout_is_terminal = false;
    ECLAIR_CHECK(!choose(StyleMode::Pretty, ColorMode::Auto, pipe).color);
    eclair::smi::TerminalEnvironment dumb = terminal;
    dumb.term = "dumb";
    ECLAIR_CHECK(!choose(StyleMode::Pretty, ColorMode::Auto, dumb).color);

    const eclair::smi::Style plain{};
    ECLAIR_CHECK(eclair::smi::paint(plain, eclair::smi::Tone::Good, "ok") == "ok");
    const eclair::smi::Style colored{.color = true};
    ECLAIR_CHECK(eclair::smi::paint(colored, eclair::smi::Tone::Good, "ok") == "\x1b[32mok\x1b[0m");
    ECLAIR_CHECK(eclair::smi::paint(colored, eclair::smi::Tone::Plain, "ok") == "ok");
}

void check_style_resolution() {
    using eclair::smi::StyleMode;
    eclair::smi::TerminalEnvironment environment;
    ECLAIR_CHECK(
        eclair::smi::resolve_style_mode(std::nullopt, environment, StyleMode::Plain).mode ==
        StyleMode::Plain);
    ECLAIR_CHECK(
        eclair::smi::resolve_style_mode(std::nullopt, environment, StyleMode::Pretty).mode ==
        StyleMode::Pretty);
    environment.style = "pretty";
    ECLAIR_CHECK(
        eclair::smi::resolve_style_mode(std::nullopt, environment, StyleMode::Plain).mode ==
        StyleMode::Pretty);
    // The command line beats the environment, which beats the build default.
    ECLAIR_CHECK(
        eclair::smi::resolve_style_mode(StyleMode::Plain, environment, StyleMode::Pretty).mode ==
        StyleMode::Plain);
    environment.style = "";
    ECLAIR_CHECK(
        !eclair::smi::resolve_style_mode(std::nullopt, environment, StyleMode::Plain).error);
    environment.style = "fancy";
    ECLAIR_CHECK(
        eclair::smi::resolve_style_mode(std::nullopt, environment, StyleMode::Plain).error);
    ECLAIR_CHECK(
        !eclair::smi::resolve_style_mode(StyleMode::Pretty, environment, StyleMode::Plain).error);

    ECLAIR_CHECK(parse({"--pretty"}).options.style == StyleMode::Pretty);
    ECLAIR_CHECK(parse({"--plain"}).options.style == StyleMode::Plain);
    ECLAIR_CHECK(parse({"--style=pretty"}).options.style == StyleMode::Pretty);
    ECLAIR_CHECK(!parse({}).options.style);
    ECLAIR_CHECK(parse({"--style", "loud"}).error.has_value());
    ECLAIR_CHECK(parse({"--pretty=yes"}).error.has_value());
}

void check_json_quote() {
    ECLAIR_CHECK(eclair::smi::json_quote("plain") == "\"plain\"");
    ECLAIR_CHECK(eclair::smi::json_quote("a\"b\\c") == "\"a\\\"b\\\\c\"");
    ECLAIR_CHECK(eclair::smi::json_quote("tab\there\n") == "\"tab\\there\\n\"");
    ECLAIR_CHECK(eclair::smi::json_quote(std::string_view("\x01", 1)) == "\"\\u0001\"");
    ECLAIR_CHECK(eclair::smi::json_quote("·") == "\"·\"");
}

void check_select_device() {
    eclair::smi::SystemReport report = sample_report();
    ECLAIR_CHECK(!eclair::smi::select_device("0", report));
    ECLAIR_CHECK(report.backends[0].devices.size() == 1U);
    ECLAIR_CHECK(report.backends[1].devices.empty());
    ECLAIR_CHECK(report.backends.size() == 2U);

    report = sample_report();
    ECLAIR_CHECK(!eclair::smi::select_device("host", report));
    ECLAIR_CHECK(report.backends[0].devices.empty());

    report = sample_report();
    ECLAIR_CHECK(eclair::smi::select_device("1", report).has_value());
    ECLAIR_CHECK(eclair::smi::select_device("cuda", report).has_value());
    ECLAIR_CHECK(eclair::smi::select_device("", report).has_value());
    ECLAIR_CHECK(report.backends[0].devices.size() == 1U);

    // A bare index that names devices in two backends is ambiguous.
    report = sample_report();
    report.backends[1].devices[0].id = "other:0";
    ECLAIR_CHECK(eclair::smi::select_device("0", report).has_value());
    ECLAIR_CHECK(!eclair::smi::select_device("other:0", report));
}

void check_policy_report() {
    eclair::smi::SystemReport report = sample_report();
    eclair::capability::FeaturePolicy policy;
    ECLAIR_CHECK(!policy.disable_list("fp8.e5m2.decode"));
    eclair::smi::apply_policy(policy, "--disable", report);
    ECLAIR_CHECK(report.disabled_keys.size() == 1U);
    ECLAIR_CHECK(report.disabled_keys[0] == "fp8.e5m2.decode");
    ECLAIR_CHECK(report.policy_source == "--disable");

    std::ostringstream text;
    eclair::smi::render_report(text, report, {});
    ECLAIR_CHECK(contains(text.str(), "Disabled by --disable: fp8.e5m2.decode"));
    ECLAIR_CHECK(row(text.str(), "FP8 E5M2 decode", {"Disabled"}));

    eclair::smi::SystemReport unchanged = sample_report();
    eclair::smi::apply_policy(eclair::capability::FeaturePolicy{}, "unused", unchanged);
    ECLAIR_CHECK(unchanged.disabled_keys.empty());
    ECLAIR_CHECK(unchanged.policy_source.empty());
}

void check_text_rendering() {
    const eclair::smi::SystemReport report = sample_report();

    std::ostringstream ascii;
    eclair::smi::render_report(ascii, report, {});
    const std::string plain = ascii.str();
    ECLAIR_CHECK(!contains(plain, "\x1b["));
    ECLAIR_CHECK(contains(plain, "Eclair 0.0.0-test"));
    ECLAIR_CHECK(contains(plain, "NVIDIA GPU 0  Tesla"));
    ECLAIR_CHECK(contains(plain, "Volta sm_70, 72 compute units, 31.7 GiB"));
    ECLAIR_CHECK(contains(plain, "CUDA driver API 13.0, runtime 12.9, kernels for sm 70"));
    ECLAIR_CHECK(contains(plain, "FP16 tensor cores present, FP8 tensor cores absent"));
    ECLAIR_CHECK(row(plain, "FP8 E4M3FN decode", {"Ready", "emulated on GPU"}));
    ECLAIR_CHECK(row(plain, "FP8 E5M2 encode", {"Ready"}));
    ECLAIR_CHECK(row(plain, "BF16 conversion", {"Ready"}));
    ECLAIR_CHECK(row(plain, "FP8-weight matrix multiplication", {"Ready", "PyTorch, ComfyUI"}));
    ECLAIR_CHECK(row(plain, "Paged int8 attention, prefill", {"Ready", "eclair-vllm"}));
    ECLAIR_CHECK(row(plain, "Paged int8 attention, decode", {"Ready", "emulated on GPU"}));
    ECLAIR_CHECK(row(plain, "Gated DeltaNet state pass", {"Ready", "native on GPU"}));
    ECLAIR_CHECK(!contains(plain, "planned"));
    ECLAIR_CHECK(!contains(plain, "Not implemented"));
    ECLAIR_CHECK(!contains(plain, "Ray-tracing emulation"));
    ECLAIR_CHECK(contains(plain, "CPU reference: 12 known-answer checks pass"));
    // The group tables share their columns, so "Status" starts at one offset.
    const std::string_view formats = line_with(plain, "Number formats");
    const std::string_view kernels = line_with(plain, "Model kernels");
    ECLAIR_CHECK(!formats.empty() && formats.find("Status") == kernels.find("Status"));
    ECLAIR_CHECK(formats.find("Used by") == kernels.find("Used by"));
    for (std::size_t begin = 0; begin < plain.size();) {
        const std::size_t end = std::min(plain.find('\n', begin), plain.size());
        ECLAIR_CHECK(end - begin <= 80U);
        begin = end + 1U;
    }

    std::ostringstream details;
    eclair::smi::render_report(details, report, {}, true);
    ECLAIR_CHECK(row(details.str(), "Ray-tracing emulation", {"Unavailable"}));
    // Details list the CPU reference device in full after the GPU.
    ECLAIR_CHECK(contains(details.str(), "\nCPU reference\n  Portable checks"));
    ECLAIR_CHECK(contains(details.str(), "Device ID: cuda:0"));
    ECLAIR_CHECK(contains(details.str(), "fp8.e4m3fn.decode"));
    ECLAIR_CHECK(contains(details.str(), "attention.paged_int8.decode: int8 cache widened"));
    ECLAIR_CHECK(!contains(details.str(), "planned"));
    ECLAIR_CHECK(!contains(details.str(), " \n"));
    ECLAIR_CHECK(contains(details.str(), "RT cores absent"));
    // The default report fits 80 columns. Details widen the status column for
    // "Not implemented", and wrapped lines such as the hardware list stay short.
    for (std::size_t begin = 0; begin < details.str().size();) {
        const std::size_t end = std::min(details.str().find('\n', begin), details.str().size());
        ECLAIR_CHECK(end - begin <= 90U);
        begin = end + 1U;
    }
    for (const char byte : plain) {
        ECLAIR_CHECK(static_cast<unsigned char>(byte) < 0x80U);
    }

    std::ostringstream fancy;
    eclair::smi::render_report(fancy, report, {.color = true});
    ECLAIR_CHECK(contains(fancy.str(), "\x1b[32m"));
    ECLAIR_CHECK(contains(fancy.str(), "FP16 tensor cores \x1b[32mpresent"));

    // Pretty uses the same ASCII status words in every locale.
    std::ostringstream ascii_pretty;
    eclair::smi::render_report(ascii_pretty, report, {.color = true});
    ECLAIR_CHECK(contains(ascii_pretty.str(), "\x1b[32mReady"));
    for (const char byte : ascii_pretty.str()) {
        ECLAIR_CHECK(static_cast<unsigned char>(byte) < 0x80U);
    }

    // Integrations call the GPU kernels, so CPU reference rows name no users.
    auto cpu_only = sample_report();
    ECLAIR_CHECK(!eclair::smi::select_device("host", cpu_only));
    std::ostringstream cpu_text;
    eclair::smi::render_report(cpu_text, cpu_only, {});
    ECLAIR_CHECK(
        row(cpu_text.str(), "FP8-weight matrix multiplication", {"Ready", "CPU reference"}));
    ECLAIR_CHECK(row(cpu_text.str(), "Gated DeltaNet state pass", {"Ready"}));
    ECLAIR_CHECK(!contains(cpu_text.str(), "eclair-vllm"));
    ECLAIR_CHECK(!contains(cpu_text.str(), "Used by"));
    ECLAIR_CHECK(contains(cpu_text.str(), "no GPU acceleration"));
    auto unavailable = sample_report();
    unavailable.backends[0].devices.clear();
    unavailable.backends[0].state = eclair::capability::BackendState::Unavailable;
    unavailable.backends[0].detail = "CUDA initialization failed";
    std::ostringstream unavailable_text;
    eclair::smi::render_report(unavailable_text, unavailable, {});
    ECLAIR_CHECK(contains(unavailable_text.str(), "NVIDIA GPU: unavailable"));
    ECLAIR_CHECK(contains(unavailable_text.str(), "CUDA initialization failed"));
    ECLAIR_CHECK(row(unavailable_text.str(), "BF16 conversion", {"CPU reference"}));

    std::ostringstream list;
    eclair::smi::render_list(list, report, {});
    ECLAIR_CHECK(contains(list.str(), "cuda:0"));
    ECLAIR_CHECK(contains(list.str(), "10 operations ready"));

    std::ostringstream catalog;
    eclair::smi::render_catalog(catalog, {});
    ECLAIR_CHECK(contains(catalog.str(), "rt.emulation"));
    ECLAIR_CHECK(contains(catalog.str(), "Ray-tracing emulation"));
    ECLAIR_CHECK(contains(catalog.str(), "Model kernels\n  Paged int8 attention, prefill"));
    ECLAIR_CHECK(contains(catalog.str(), "Used by eclair-vllm"));
}

void check_json_rendering() {
    std::ostringstream report_json;
    eclair::smi::render_json_report(report_json, sample_report());
    const std::string json = report_json.str();
    ECLAIR_CHECK(contains(json, "\"schema\": \"eclair-smi.report/2\""));
    ECLAIR_CHECK(contains(json, "\"name\": \"Tesla \\\"PG500\\\"\\t216\""));
    ECLAIR_CHECK(contains(json, "\"memory_bytes\": 34072559616"));
    ECLAIR_CHECK(contains(json, "\"fp16_tensor_cores\": \"present\""));
    ECLAIR_CHECK(contains(json, "\"key\": \"fp8.e4m3fn.decode\""));
    ECLAIR_CHECK(contains(json, "\"state\": \"enabled\""));
    // The host device has no memory size, which JSON reports as null.
    ECLAIR_CHECK(contains(json, "\"memory_bytes\": null"));
    ECLAIR_CHECK(contains(json, "\"hardware\": {}"));
    ECLAIR_CHECK(contains(json, "\"disabled\": []"));
    ECLAIR_CHECK(!contains(json, ",\n  }") && !contains(json, ",\n}"));

    std::ostringstream catalog_json;
    eclair::smi::render_json_catalog(catalog_json);
    ECLAIR_CHECK(contains(catalog_json.str(), "\"schema\": \"eclair-smi.catalog/1\""));
    ECLAIR_CHECK(contains(catalog_json.str(), "\"key\": \"fp8.gemm\""));
}

}  // namespace

int main() {
    check_options();
    check_style();
    check_style_resolution();
    check_json_quote();
    check_select_device();
    check_policy_report();
    check_text_rendering();
    check_json_rendering();
    return eclair::test::finish();
}
