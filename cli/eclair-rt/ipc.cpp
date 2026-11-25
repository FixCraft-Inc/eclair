/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "ipc.hpp"

#include <bit>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace eclair::demo {
namespace {
[[noreturn]] void system_error(const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}
void put32(Packet& data, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) data[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
}
std::uint32_t get32(const Packet& data, std::size_t offset) {
    std::uint32_t result = 0;
    for (unsigned i = 0; i < 4; ++i)
        result |= static_cast<std::uint32_t>(data[offset + i]) << (i * 8);
    return result;
}
void put64(Packet& data, std::size_t offset, std::uint64_t value) {
    put32(data, offset, static_cast<std::uint32_t>(value));
    put32(data, offset + 4, static_cast<std::uint32_t>(value >> 32));
}
std::uint64_t get64(const Packet& data, std::size_t offset) {
    return get32(data, offset) | (static_cast<std::uint64_t>(get32(data, offset + 4)) << 32);
}
std::size_t frame_bytes(const ops::RayTraceShape& shape) {
    std::size_t result = 0;
    if (ops::ray_frame_bytes(shape, &result) != ops::Status::Success)
        throw std::invalid_argument("invalid frame dimensions");
    return result;
}
class SpawnActions {
public:
    SpawnActions() {
        const int e = posix_spawn_file_actions_init(&value);
        if (e) throw std::runtime_error(std::strerror(e));
    }
    ~SpawnActions() { posix_spawn_file_actions_destroy(&value); }
    posix_spawn_file_actions_t value{};
};
}  // namespace

FileDescriptor::~FileDescriptor() {
    if (fd_ >= 0) ::close(fd_);
}
FileDescriptor::FileDescriptor(FileDescriptor&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)) {}
FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
}
Mapping::Mapping(int fd, std::size_t bytes) : bytes_(bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) system_error("mmap frames");
    data_ = static_cast<std::uint8_t*>(p);
}
Mapping::~Mapping() {
    if (data_) munmap(data_, bytes_);
}
Packet encode(const FrameMessage& m) {
    Packet p{};
    put32(p, 0, 0x54524345U);
    put32(p, 4, 2);
    put32(p, 8, static_cast<std::uint32_t>(m.kind));
    put32(p, 12, m.slot);
    put32(p, 16, m.sequence);
    put32(p, 20, m.seed);
    put32(p, 24, m.samples);
    put32(p, 28, m.bounces);
    put32(p, 32, std::bit_cast<std::uint32_t>(m.seconds));
    put32(p, 36, m.scene);
    put64(p, 40, m.render_ns);
    put64(p, 48, m.copy_ns);
    return p;
}
FrameMessage decode(const Packet& p) {
    const auto kind = get32(p, 8);
    if (get32(p, 0) != 0x54524345U || get32(p, 4) != 2 || kind < 1 || kind > 4 ||
        get32(p, 36) > 1 || get64(p, 56) != 0)
        throw std::runtime_error("invalid renderer packet");
    FrameMessage m;
    m.kind = static_cast<Message>(kind);
    m.slot = get32(p, 12);
    m.sequence = get32(p, 16);
    m.seed = get32(p, 20);
    m.samples = get32(p, 24);
    m.bounces = get32(p, 28);
    m.seconds = std::bit_cast<float>(get32(p, 32));
    m.scene = get32(p, 36);
    m.render_ns = get64(p, 40);
    m.copy_ns = get64(p, 48);
    if (m.slot > 1 || m.samples == 0 || m.samples > 8 || m.bounces == 0 || m.bounces > 8 ||
        !render::finite(m.seconds) || ::fabsf(m.seconds) > 1.0e6F)
        throw std::runtime_error("out-of-range renderer packet");
    return m;
}
void send_packet(int fd, const FrameMessage& m) {
    const auto p = encode(m);
    ssize_t n;
    do {
        n = send(fd, p.data(), p.size(), MSG_NOSIGNAL);
    } while (n < 0 && errno == EINTR);
    if (n != static_cast<ssize_t>(p.size())) system_error("send renderer packet");
}
bool readable(int fd, int timeout_ms) {
    pollfd descriptor{fd, POLLIN, 0};
    int status;
    do {
        status = poll(&descriptor, 1, timeout_ms);
    } while (status < 0 && errno == EINTR);
    if (status < 0) system_error("poll renderer");
    return status > 0;
}
FrameMessage receive_packet(int fd, int timeout_ms) {
    if (!readable(fd, timeout_ms)) throw std::runtime_error("renderer response timeout");
    Packet p{};
    ssize_t n;
    do {
        n = recv(fd, p.data(), p.size(), MSG_TRUNC);
    } while (n < 0 && errno == EINTR);
    if (n != static_cast<ssize_t>(p.size()))
        throw std::runtime_error("renderer closed or sent a truncated packet");
    return decode(p);
}
std::size_t shared_frame_bytes(const ops::RayTraceShape& shape) {
    const auto bytes = frame_bytes(shape) * 2;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) throw std::runtime_error("cannot determine page size");
    const auto size = static_cast<std::size_t>(page);
    return ((bytes + size - 1) / size) * size;
}
FileDescriptor create_shared_frames(std::size_t bytes) {
    FileDescriptor fd(memfd_create("eclair-rt-frames", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (fd.get() < 0) system_error("memfd_create");
    if (bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max()) ||
        ftruncate(fd.get(), static_cast<off_t>(bytes)) != 0)
        system_error("size shared frames");
    if (fcntl(fd.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) < 0)
        system_error("seal shared frames");
    return fd;
}
void validate_shared_frames(int fd, std::size_t bytes) {
    struct stat st{};
    if (fstat(fd, &st) != 0) system_error("stat shared frames");
    const int seals = fcntl(fd, F_GET_SEALS);
    if (!S_ISREG(st.st_mode) || st.st_uid != getuid() || st.st_size < 0 ||
        static_cast<std::uint64_t>(st.st_size) != bytes || seals < 0 ||
        (seals & (F_SEAL_GROW | F_SEAL_SHRINK)) != (F_SEAL_GROW | F_SEAL_SHRINK))
        throw std::runtime_error("invalid shared frame allocation");
}
CudaWorker::CudaWorker(const std::filesystem::path& executable, const ops::RayTraceShape& shape,
                       bool denoise)
    : shape_(shape),
      frame_bytes_(frame_bytes(shape)),
      frames_(create_shared_frames(shared_frame_bytes(shape))),
      mapping_(frames_.get(), shared_frame_bytes(shape)) {
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair) != 0)
        system_error("socketpair");
    control_ = FileDescriptor(pair[0]);
    FileDescriptor child(pair[1]);
    SpawnActions actions;
    for (const auto& assignment :
         {std::pair{child.get(), kWorkerControlFd}, std::pair{frames_.get(), kWorkerFramesFd}}) {
        const int e =
            posix_spawn_file_actions_adddup2(&actions.value, assignment.first, assignment.second);
        if (e) throw std::runtime_error(std::strerror(e));
    }
    std::string path = executable.string(), width = std::to_string(shape.width),
                height = std::to_string(shape.height), parent = std::to_string(getpid()),
                filter = denoise ? "1" : "0";
    std::array<char*, 6> argv{path.data(),   width.data(),  height.data(),
                              parent.data(), filter.data(), nullptr};
    const int error =
        posix_spawn(&pid_, path.c_str(), &actions.value, nullptr, argv.data(), environ);
    if (error) {
        pid_ = -1;
        throw std::runtime_error("start CUDA worker: " + std::string(std::strerror(error)));
    }
    try {
        if (receive_packet(control_.get(), 15000).kind != Message::Hello)
            throw std::runtime_error("CUDA worker initialization failed");
    } catch (...) {
        stop();
        throw;
    }
}
CudaWorker::~CudaWorker() {
    try {
        stop();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "worker cleanup: %s\n", e.what());
    }
}
void CudaWorker::request(const FrameMessage& m) {
    if (pending_ || m.kind != Message::Frame || m.slot > 1)
        throw std::logic_error("invalid frame ownership transition");
    send_packet(control_.get(), m);
    pending_ = true;
    expected_slot_ = m.slot;
    expected_sequence_ = m.sequence;
}
bool CudaWorker::ready() const {
    return pending_ && readable(control_.get(), 0);
}
FrameMessage CudaWorker::receive() {
    if (!pending_) throw std::logic_error("no CUDA frame pending");
    const auto m = receive_packet(control_.get(), 15000);
    if (m.kind != Message::Frame || m.slot != expected_slot_ || m.sequence != expected_sequence_)
        throw std::runtime_error("CUDA worker failed or returned a stale frame");
    pending_ = false;
    return m;
}
std::span<const std::uint8_t> CudaWorker::pixels(std::uint32_t slot) const {
    if (slot > 1 || pending_) throw std::logic_error("CUDA frame is not CPU-owned");
    return {mapping_.data() + slot * frame_bytes_, frame_bytes_};
}
void CudaWorker::stop() {
    if (pid_ < 0) return;
    const auto pid = pid_;
    // The worker observes Quit after its bounded outstanding frame, then RAII
    // unregisters pinned memory. Reaping is the release acknowledgment.
    try {
        FrameMessage quit;
        quit.kind = Message::Quit;
        send_packet(control_.get(), quit);
    } catch (...) {
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            pid_ = -1;
            pending_ = false;
            return;
        }
        if (result < 0 && errno != EINTR) {
            if (errno == ECHILD) {
                pid_ = -1;
                return;
            }
            system_error("wait CUDA worker");
        }
        poll(nullptr, 0, 10);
    }
    if (kill(pid, SIGKILL) != 0 && errno != ESRCH) system_error("stop CUDA worker");
    pid_t result;
    do {
        result = waitpid(pid, &status, 0);
    } while (result < 0 && errno == EINTR);
    if (result < 0 && errno != ECHILD) system_error("reap CUDA worker");
    pid_ = -1;
    pending_ = false;
}
}  // namespace eclair::demo
