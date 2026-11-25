/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

#include "system_report.hpp"

#include <ostream>
#include <string>
#include <string_view>

namespace eclair::smi {

// Schema identifiers. A field rename or meaning change needs a new number.
inline constexpr std::string_view kReportSchema = "eclair-smi.report/2";
inline constexpr std::string_view kCatalogSchema = "eclair-smi.catalog/1";

void render_json_report(std::ostream& out, const SystemReport& report);

void render_json_catalog(std::ostream& out);

// A JSON string literal, quotes included, with control characters escaped.
// Bytes of 0x80 and above pass through, so valid UTF-8 stays valid.
[[nodiscard]] std::string json_quote(std::string_view text);

}  // namespace eclair::smi
