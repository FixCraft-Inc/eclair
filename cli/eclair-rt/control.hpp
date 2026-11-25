/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "ipc.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>

namespace eclair::demo {
struct ControlCommand {
    std::string text;
    sockaddr_un peer{};
    socklen_t peer_size = 0;
};
class ControlServer {
public:
    ControlServer();
    ~ControlServer();
    ControlServer(const ControlServer&) = delete;
    ControlServer& operator=(const ControlServer&) = delete;
    [[nodiscard]] std::optional<ControlCommand> next();
    void reply(const ControlCommand& command, const std::string& response) const;
    [[nodiscard]] std::filesystem::path directory() const { return path_.parent_path(); }

private:
    std::filesystem::path path_;
    FileDescriptor lock_, socket_;
    bool bound_ = false;
};
[[nodiscard]] std::string control_request(const std::string& command);
}  // namespace eclair::demo
