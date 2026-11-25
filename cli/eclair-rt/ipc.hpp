/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once
#include "eclair/ops/ray_trace.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

#include <sys/types.h>

namespace eclair::demo {
class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) noexcept : fd_(fd) {}
    ~FileDescriptor();
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept;
    FileDescriptor& operator=(FileDescriptor&& other) noexcept;
    [[nodiscard]] int get() const noexcept { return fd_; }

private:
    int fd_;
};
class Mapping {
public:
    Mapping(int fd, std::size_t bytes);
    ~Mapping();
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    [[nodiscard]] std::uint8_t* data() const noexcept { return data_; }

private:
    std::uint8_t* data_ = nullptr;
    std::size_t bytes_ = 0;
};

inline constexpr int kWorkerControlFd = 198, kWorkerFramesFd = 199;
using Packet = std::array<std::uint8_t, 64>;
enum class Message : std::uint32_t { Hello = 1, Frame = 2, Quit = 3, Failure = 4 };
struct FrameMessage {
    Message kind = Message::Frame;
    std::uint32_t slot = 0, sequence = 0, seed = 1, samples = 1, bounces = 2;
    float seconds = 0;
    std::uint32_t scene = 0;  // Demo scene ID; protocol v2, bounded to 0..1.
    std::uint64_t render_ns = 0, copy_ns = 0;
};
[[nodiscard]] Packet encode(const FrameMessage& message);
[[nodiscard]] FrameMessage decode(const Packet& packet);
void send_packet(int fd, const FrameMessage& message);
[[nodiscard]] FrameMessage receive_packet(int fd, int timeout_ms);
[[nodiscard]] bool readable(int fd, int timeout_ms);
[[nodiscard]] std::size_t shared_frame_bytes(const ops::RayTraceShape& shape);
[[nodiscard]] FileDescriptor create_shared_frames(std::size_t bytes);
void validate_shared_frames(int fd, std::size_t bytes);

class CudaWorker {
public:
    CudaWorker(const std::filesystem::path& executable, const ops::RayTraceShape& shape,
               bool denoise = false);
    ~CudaWorker();
    CudaWorker(const CudaWorker&) = delete;
    CudaWorker& operator=(const CudaWorker&) = delete;
    void request(const FrameMessage& message);
    [[nodiscard]] bool ready() const;
    [[nodiscard]] FrameMessage receive();
    [[nodiscard]] std::span<const std::uint8_t> pixels(std::uint32_t slot) const;
    [[nodiscard]] pid_t pid() const noexcept { return pid_; }
    void stop();

private:
    ops::RayTraceShape shape_;
    std::size_t frame_bytes_ = 0;
    FileDescriptor frames_;
    Mapping mapping_;
    FileDescriptor control_;
    pid_t pid_ = -1;
    bool pending_ = false;
    std::uint32_t expected_slot_ = 0, expected_sequence_ = 0;
};
}  // namespace eclair::demo
