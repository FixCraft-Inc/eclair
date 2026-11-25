/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "control.hpp"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

namespace eclair::demo {
namespace {
std::filesystem::path runtime_directory() {
    std::filesystem::path base;
    if (const char* value = std::getenv("XDG_RUNTIME_DIR"))
        base = value;
    else
        base = "/run/user/" + std::to_string(getuid());
    struct stat st{};
    if (lstat(base.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != getuid() ||
        (st.st_mode & 077) != 0)
        throw std::runtime_error(
            "XDG_RUNTIME_DIR must be a private directory owned by the current user");
    const auto path = base / "eclair-rt";
    if (mkdir(path.c_str(), 0700) != 0 && errno != EEXIST)
        throw std::runtime_error("create preview runtime directory failed");
    if (lstat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != getuid() ||
        (st.st_mode & 077) != 0)
        throw std::runtime_error("unsafe preview runtime directory");
    return path;
}
sockaddr_un address(const std::filesystem::path& path) {
    sockaddr_un result{};
    result.sun_family = AF_UNIX;
    const auto text = path.string();
    if (text.size() >= sizeof(result.sun_path))
        throw std::invalid_argument("preview socket path too long");
    std::memcpy(result.sun_path, text.c_str(), text.size() + 1);
    return result;
}
}  // namespace
ControlServer::ControlServer() : path_(runtime_directory() / "control.sock") {
    lock_ = FileDescriptor(open((path_.parent_path() / "lock").c_str(),
                                O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600));
    struct stat st{};
    if (lock_.get() < 0 || fstat(lock_.get(), &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_uid != getuid() || flock(lock_.get(), LOCK_EX | LOCK_NB) != 0)
        throw std::runtime_error("preview already running or lock is unsafe");
    if (lstat(path_.c_str(), &st) == 0) {
        if (!S_ISSOCK(st.st_mode) || st.st_uid != getuid())
            throw std::runtime_error("unsafe stale preview socket");
        if (unlink(path_.c_str()) != 0)
            throw std::runtime_error("remove stale preview socket failed");
    } else if (errno != ENOENT)
        throw std::runtime_error("inspect preview socket failed");
    socket_ = FileDescriptor(socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
    int enabled = 1;
    if (socket_.get() < 0 ||
        setsockopt(socket_.get(), SOL_SOCKET, SO_PASSCRED, &enabled, sizeof(enabled)) != 0)
        throw std::runtime_error("create control socket failed");
    const auto addr = address(path_);
    if (bind(socket_.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0)
        throw std::runtime_error("bind control socket failed");
    bound_ = true;
    if (chmod(path_.c_str(), 0600) != 0) {
        unlink(path_.c_str());
        bound_ = false;
        throw std::runtime_error("protect control socket failed");
    }
}
ControlServer::~ControlServer() {
    if (bound_) unlink(path_.c_str());
}
std::optional<ControlCommand> ControlServer::next() {
    std::array<char, 128> data{};
    alignas(cmsghdr) std::array<char, CMSG_SPACE(sizeof(ucred))> credentials{};
    ControlCommand command;
    iovec buffer{data.data(), data.size()};
    msghdr message{};
    message.msg_name = &command.peer;
    message.msg_namelen = sizeof(command.peer);
    message.msg_iov = &buffer;
    message.msg_iovlen = 1;
    message.msg_control = credentials.data();
    message.msg_controllen = credentials.size();
    const ssize_t count = recvmsg(socket_.get(), &message, MSG_DONTWAIT | MSG_TRUNC);
    if (count < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return {};
        throw std::runtime_error("read control socket failed");
    }
    if (count == 0 || count >= static_cast<ssize_t>(data.size()) ||
        (message.msg_flags & MSG_CTRUNC) != 0)
        return {};
    bool authorized = false;
    for (cmsghdr* c = CMSG_FIRSTHDR(&message); c; c = CMSG_NXTHDR(&message, c)) {
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_CREDENTIALS &&
            c->cmsg_len == CMSG_LEN(sizeof(ucred))) {
            ucred peer{};
            std::memcpy(&peer, CMSG_DATA(c), sizeof(peer));
            authorized = peer.uid == getuid();
        }
    }
    if (!authorized) return {};
    command.peer_size = message.msg_namelen;
    command.text.assign(data.data(), static_cast<std::size_t>(count));
    return command;
}
void ControlServer::reply(const ControlCommand& command, const std::string& response) const {
    const auto ignored =
        sendto(socket_.get(), response.data(), response.size(), MSG_DONTWAIT | MSG_NOSIGNAL,
               reinterpret_cast<const sockaddr*>(&command.peer), command.peer_size);
    static_cast<void>(ignored);  // A disconnected command client must not stop the preview.
}
std::string control_request(const std::string& command) {
    if (command.empty() || command.size() > 100)
        throw std::invalid_argument("invalid control command size");
    FileDescriptor fd(socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0));
    if (fd.get() < 0) throw std::runtime_error("create command socket failed");
    sockaddr_un local{};
    local.sun_family = AF_UNIX;
    std::uint64_t nonce = 0;
    if (getrandom(&nonce, sizeof(nonce), 0) != static_cast<ssize_t>(sizeof(nonce)))
        throw std::runtime_error("command nonce unavailable");
    const auto name = "eclair-rtctl-" + std::to_string(getuid()) + "-" + std::to_string(getpid()) +
                      "-" + std::to_string(nonce);
    if (name.size() + 1 >= sizeof(local.sun_path))
        throw std::runtime_error("command address too long");
    std::memcpy(local.sun_path + 1, name.data(), name.size());
    const auto local_size =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 + name.size());
    if (bind(fd.get(), reinterpret_cast<const sockaddr*>(&local), local_size) != 0)
        throw std::runtime_error("bind command socket failed");
    const auto remote = address(runtime_directory() / "control.sock");
    if (connect(fd.get(), reinterpret_cast<const sockaddr*>(&remote), sizeof(remote)) != 0)
        throw std::runtime_error("preview is not running");
    if (send(fd.get(), command.data(), command.size(), MSG_NOSIGNAL) !=
        static_cast<ssize_t>(command.size()))
        throw std::runtime_error("send command failed");
    if (!readable(fd.get(), 20000)) throw std::runtime_error("preview command timed out");
    std::array<char, 2048> data{};
    const ssize_t count = recv(fd.get(), data.data(), data.size(), MSG_TRUNC);
    if (count <= 0 || count >= static_cast<ssize_t>(data.size()))
        throw std::runtime_error("invalid preview reply");
    return {data.data(), static_cast<std::size_t>(count)};
}
}  // namespace eclair::demo
