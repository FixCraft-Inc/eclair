/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "ipc.hpp"
#include "scene.hpp"
#include "eclair/capability/policy.hpp"
#include "eclair/cuda/ray_trace.hpp"
#include "eclair/cuda/ray_preview.hpp"

#include <cuda_runtime.h>
#include <charconv>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <optional>
#include <limits>

#include <sys/prctl.h>
#include <unistd.h>

namespace {
void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
std::uint32_t number(const char* value) {
    const std::string_view text(value);
    std::uint32_t result = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size() || result == 0)
        throw std::invalid_argument("invalid worker argument");
    return result;
}
class Registration {
public:
    Registration(void* pointer, std::size_t bytes) : pointer_(pointer) {
        checked(cudaHostRegister(pointer, bytes, cudaHostRegisterDefault), "register frame memory");
    }
    ~Registration() {
        if (cudaHostUnregister(pointer_) != cudaSuccess)
            std::fputs("unregister frame memory failed\n", stderr);
    }
    Registration(const Registration&) = delete;
    Registration& operator=(const Registration&) = delete;

private:
    void* pointer_;
};
template <class T>
class DeviceFrame {
public:
    explicit DeviceFrame(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("device frame too large");
        checked(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "allocate frame");
    }
    ~DeviceFrame() {
        if (data_ && cudaFree(data_) != cudaSuccess) std::fputs("free frame failed\n", stderr);
    }
    DeviceFrame(const DeviceFrame&) = delete;
    DeviceFrame& operator=(const DeviceFrame&) = delete;
    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};
struct PreviewStorage {
    explicit PreviewStorage(std::size_t count) : a(count), b(count), scratch(count * 2) {}
    DeviceFrame<eclair::render::PreviewPixel> a, b;
    DeviceFrame<eclair::render::Vec3> scratch;
};
struct LightingStorage {
    explicit LightingStorage(std::size_t size) : count(size), a(size), b(size), scratch(size * 2) {}
    std::size_t count;
    DeviceFrame<eclair::render::SurfaceLighting> a, b;
    DeviceFrame<eclair::render::Vec3> scratch;
};
class Stream {
public:
    Stream() { checked(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() {
        cudaStreamSynchronize(value);
        cudaStreamDestroy(value);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t value = nullptr;
};
class Event {
public:
    Event() { checked(cudaEventCreate(&value), "create event"); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t value = nullptr;
};
int run(int argc, char** argv) {
    using namespace eclair;
    if (argc != 5) throw std::invalid_argument("CUDA worker must be started by eclair-rt");
    ops::RayTraceShape shape{number(argv[1]), number(argv[2]), 1, 2, 1};
    const auto parent = number(argv[3]);
    const std::string_view filter_arg(argv[4]);
    if (filter_arg != "0" && filter_arg != "1") throw std::invalid_argument("invalid filter mode");
    const bool denoise = filter_arg == "1";
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || static_cast<std::uint32_t>(getppid()) != parent)
        throw std::runtime_error("renderer parent is gone");
    capability::FeaturePolicy policy;
    if (const char* disabled = std::getenv("ECLAIR_DISABLE")) {
        if (const auto error = policy.disable_list(disabled)) throw std::invalid_argument(*error);
    }
    if (policy.is_disabled(capability::Feature::RayTracingEmulation))
        throw std::runtime_error("rt.emulation disabled by policy");
    std::size_t bytes = 0;
    if (ops::ray_frame_bytes(shape, &bytes) != ops::Status::Success)
        throw std::invalid_argument("invalid frame dimensions");
    demo::FileDescriptor control(demo::kWorkerControlFd), frames(demo::kWorkerFramesFd);
    const std::size_t storage = demo::shared_frame_bytes(shape);
    demo::validate_shared_frames(frames.get(), storage);
    demo::Mapping mapping(frames.get(), storage);
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, 0), "device properties");
    if (properties.major != 7 || properties.minor != 0)
        throw std::runtime_error("Cornell CUDA worker is validated for sm_70 only");
    checked(cuda::check_ray_trace_image(0), "ray kernel image");
    int driver = 0, runtime = 0;
    checked(cudaDriverGetVersion(&driver), "driver API version");
    checked(cudaRuntimeGetVersion(&runtime), "runtime version");
    std::fprintf(stderr,
                 "Cornell CUDA device=%s sm_%d%d driver_api=%d runtime=%d frame_vram=%zu "
                 "pinned_ram=%zu\n",
                 properties.name, properties.major, properties.minor, driver, runtime,
                 bytes + (denoise ? bytes / 4 * render::kPreviewBytesPerPixel : 0), storage);
    Registration registration(mapping.data(), storage);
    DeviceFrame<std::uint8_t> device(bytes);
    std::optional<PreviewStorage> filtering;
    std::optional<LightingStorage> lighting;
    const std::size_t pixels = bytes / 4;
    if (denoise) filtering.emplace(pixels);
    bool valid_history = false, alternate = false;
    demo::FrameScene previous;
    float previous_seconds = 0;
    Stream stream;
    Event start, rendered, copied;
    demo::FrameMessage hello;
    hello.kind = demo::Message::Hello;
    demo::send_packet(control.get(), hello);
    while (true) {
        const auto request = demo::receive_packet(control.get(), -1);
        if (request.kind == demo::Message::Quit) break;
        if (request.kind != demo::Message::Frame)
            throw std::runtime_error("invalid worker request");
        shape.seed = request.seed;
        shape.samples = request.samples;
        shape.bounces = request.bounces;
        const auto scene =
            demo::make_scene(static_cast<demo::SceneKind>(request.scene), request.seconds);
        checked(cudaEventRecord(start.value, stream.value), "record frame start");
        if (filtering) {
            const auto layout = render::lighting_layout(scene.scene);
            const std::size_t lighting_count =
                static_cast<std::size_t>(layout.tiles) * render::kLightingTilePixels;
            // Even a fully specular scene uses a non-null cache descriptor.
            const std::size_t capacity = lighting_count > 0 ? lighting_count : 1;
            if (!lighting || lighting->count != capacity) {
                lighting.emplace(capacity);
                valid_history = false;
                std::fprintf(stderr, "surface_lighting_texels=%zu explicit_vram=%zu\n",
                             lighting_count,
                             bytes + pixels * render::kPreviewBytesPerPixel +
                                 capacity * render::kLightingBytesPerTexel);
            }
            auto* current = alternate ? filtering->b.data() : filtering->a.data();
            const auto* old = alternate ? filtering->a.data() : filtering->b.data();
            const bool reuse = valid_history && request.seconds >= previous_seconds &&
                               request.seconds - previous_seconds < 0.25F;
            checked(cuda::launch_ray_preview(
                        scene.scene, scene.camera, previous.scene, previous.camera, shape, reuse,
                        {current, pixels}, {old, pixels}, {filtering->scratch.data(), pixels * 2},
                        {device.data(), bytes}, stream.value,
                        {{alternate ? lighting->b.data() : lighting->a.data(), capacity},
                         {alternate ? lighting->a.data() : lighting->b.data(), capacity},
                         {lighting->scratch.data(), capacity * 2}}),
                    "filtered ray preview");
            previous = scene;
            previous_seconds = request.seconds;
            valid_history = true;
            alternate = !alternate;
        } else {
            checked(cuda::launch_ray_trace(scene.scene, scene.camera, shape, {device.data(), bytes},
                                           stream.value),
                    "ray trace");
        }
        checked(cudaEventRecord(rendered.value, stream.value), "record render complete");
        checked(cudaMemcpyAsync(mapping.data() + request.slot * bytes, device.data(), bytes,
                                cudaMemcpyDeviceToHost, stream.value),
                "copy frame to RAM");
        checked(cudaEventRecord(copied.value, stream.value), "record copy complete");
        checked(cudaEventSynchronize(copied.value), "wait frame copy");
        float render_ms = 0, copy_ms = 0;
        checked(cudaEventElapsedTime(&render_ms, start.value, rendered.value), "render time");
        checked(cudaEventElapsedTime(&copy_ms, rendered.value, copied.value), "copy time");
        auto response = request;
        response.render_ns = static_cast<std::uint64_t>(static_cast<double>(render_ms) * 1.0e6);
        response.copy_ns = static_cast<std::uint64_t>(static_cast<double>(copy_ms) * 1.0e6);
        demo::send_packet(control.get(), response);
    }
    checked(cudaStreamSynchronize(stream.value), "drain frame stream");
    return EXIT_SUCCESS;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "eclair-rt CUDA worker: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
