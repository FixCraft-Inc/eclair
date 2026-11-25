/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "control.hpp"
#include "ipc.hpp"
#include "presenter.hpp"
#include "scene.hpp"
#include "eclair/render/denoise.hpp"

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {
using namespace eclair;
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
extern "C" void stop_signal(int) {
    interrupted = 1;
}
enum class Mode : std::uint8_t { Raster, CpuRay, CudaRay, Pause };
std::string mode_name(Mode mode) {
    switch (mode) {
        case Mode::Raster:
            return "raster";
        case Mode::CpuRay:
            return "cpu-ray";
        case Mode::CudaRay:
            return "cuda-ray";
        case Mode::Pause:
            return "pause";
    }
    throw std::logic_error("invalid mode");
}
Mode parse_mode(std::string_view text) {
    if (text == "raster") return Mode::Raster;
    if (text == "cpu-ray") return Mode::CpuRay;
    if (text == "cuda-ray") return Mode::CudaRay;
    if (text == "pause") return Mode::Pause;
    throw std::invalid_argument("mode must be raster, cpu-ray, cuda-ray or pause");
}
std::uint32_t number(std::string_view text, std::uint32_t minimum, std::uint32_t maximum) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < minimum ||
        value > maximum)
        throw std::invalid_argument("numeric argument outside supported range");
    return value;
}
struct Options {
    ops::RayTraceShape shape{1280, 720, 1, 2, 1};
    Mode mode = Mode::CudaRay;
    demo::SceneKind scene = demo::SceneKind::Cornell;
    bool headless = false, listen = false, static_scene = false, denoise = false;
    std::uint32_t frames = 0, seconds = 0, threads = 8;
    std::filesystem::path worker, output;
    std::string control;
};
void usage() {
    std::puts(
        "eclair-rt [--mode raster|cpu-ray|cuda-ray|pause] [--width N --height N]\n"
        "          [--scene cornell|materials] [--samples 1..8 --bounces 1..8 --threads 1..8] "
        "[--static]\n"
        "          [--denoise] [--headless] [--listen] [--frames N] [--seconds N] [--output "
        "NEW.ppm]\n"
        "          [--cuda-worker /absolute/path/eclair-rt-cuda-worker]\n"
        "eclair-rt --control status|raster|cpu-ray|cuda-ray|still|animate|pause|snapshot|stop\n"
        "Keys: 1 raster, 2 CPU ray, 3 V100 ray, S still, A animate, Space pause, Esc quit.\n"
        "Pause releases the CUDA worker and CPU frame; stop also closes the display.");
}
Options options(int argc, char** argv) {
    Options o;
    o.worker = std::filesystem::canonical("/proc/self/exe").parent_path() / "eclair-rt-cuda-worker";
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help") {
            usage();
            std::exit(EXIT_SUCCESS);
        }
        if (key == "--headless") {
            o.headless = true;
            continue;
        }
        if (key == "--listen") {
            o.listen = true;
            continue;
        }
        if (key == "--denoise") {
            o.denoise = true;
            continue;
        }
        if (key == "--static") {
            o.static_scene = true;
            continue;
        }
        if (i + 1 == argc) throw std::invalid_argument("missing option value");
        const std::string_view value(argv[++i]);
        if (key == "--mode")
            o.mode = parse_mode(value);
        else if (key == "--scene") {
            if (value == "cornell")
                o.scene = demo::SceneKind::Cornell;
            else if (value == "materials")
                o.scene = demo::SceneKind::Materials;
            else
                throw std::invalid_argument("scene must be cornell or materials");
        } else if (key == "--width")
            o.shape.width = number(value, 16, 4096);
        else if (key == "--height")
            o.shape.height = number(value, 16, 4096);
        else if (key == "--samples")
            o.shape.samples = number(value, 1, 8);
        else if (key == "--bounces")
            o.shape.bounces = number(value, 1, 8);
        else if (key == "--threads")
            o.threads = number(value, 1, 8);
        else if (key == "--frames")
            o.frames = number(value, 1, 1000000);
        else if (key == "--seconds")
            o.seconds = number(value, 1, 86400);
        else if (key == "--cuda-worker")
            o.worker = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--control")
            o.control = value;
        else
            throw std::invalid_argument("unknown option: " + std::string(key));
    }
    if (o.headless && o.frames == 0 && o.seconds == 0 && !o.listen) o.frames = 120;
    if (!o.worker.is_absolute()) throw std::invalid_argument("CUDA worker path must be absolute");
    return o;
}
std::string json_string(std::string_view text) {
    std::string result = "\"";
    for (char character : text) {
        const auto c = static_cast<unsigned char>(character);
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c >= 32 && c < 127)
            result += static_cast<char>(c);
        else
            result += ' ';
    }
    return result + '"';
}
void save_ppm(const std::filesystem::path& path, const ops::RayTraceShape& shape,
              std::span<const std::uint8_t> rgba) {
    std::size_t bytes = 0;
    if (ops::ray_frame_bytes(shape, &bytes) != ops::Status::Success || rgba.size() < bytes)
        throw std::runtime_error("no completed CPU frame available");
    demo::FileDescriptor fd(open(
        path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0600));
    if (fd.get() < 0) throw std::runtime_error("cannot create new snapshot file");
    const auto write_all = [&](const void* pointer, std::size_t size) {
        const auto* data = static_cast<const std::uint8_t*>(pointer);
        while (size) {
            const ssize_t count = write(fd.get(), data, size);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("snapshot write failed");
            data += count;
            size -= static_cast<std::size_t>(count);
        }
    };
    const std::string header =
        "P6\n" + std::to_string(shape.width) + " " + std::to_string(shape.height) + "\n255\n";
    write_all(header.data(), header.size());
    std::vector<std::uint8_t> row(static_cast<std::size_t>(shape.width) * 3);
    for (std::uint32_t y = 0; y < shape.height; ++y) {
        for (std::uint32_t x = 0; x < shape.width; ++x) {
            const auto src = (static_cast<std::size_t>(y) * shape.width + x) * 4;
            for (std::size_t c = 0; c < 3; ++c)
                row[static_cast<std::size_t>(x) * 3 + c] = rgba[src + c];
        }
        write_all(row.data(), row.size());
    }
}
struct Statistics {
    std::uint64_t frames = 0;
    double render_ms = 0, copy_ms = 0;
    double last_render_ms = 0, last_copy_ms = 0, last_upload_ms = 0, last_present_ms = 0;
    std::vector<double> intervals;
    Clock::time_point last = Clock::now(), started = last;
    double fps = 0;
    void completed(double render, double copy) {
        const auto now = Clock::now();
        const double interval = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        ++frames;
        render_ms += render;
        copy_ms += copy;
        last_render_ms = render;
        last_copy_ms = copy;
        if (intervals.size() < 600)
            intervals.push_back(interval);
        else
            intervals[(frames - 1) % 600] = interval;
        if (interval > 0) fps = fps == 0 ? 1000 / interval : fps * 0.9 + 100 / interval;
    }
};
int run(const Options& o) {
    if (!o.control.empty()) {
        const auto reply = demo::control_request(o.control);
        std::cout << reply << '\n';
        return reply.find("\"error\":") == std::string::npos ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    std::signal(SIGINT, stop_signal);
    std::signal(SIGTERM, stop_signal);
    std::signal(SIGHUP, stop_signal);
    std::size_t bytes = 0;
    if (ops::ray_frame_bytes(o.shape, &bytes) != ops::Status::Success)
        throw std::invalid_argument("invalid dimensions");
    const auto lighting_texels = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               render::lighting_layout(demo::make_scene(o.scene, 0).scene).tiles) *
               render::kLightingTilePixels);
    const auto filtered_vram = bytes / 4 * render::kPreviewBytesPerPixel +
                               lighting_texels * render::kLightingBytesPerTexel;
    std::unique_ptr<demo::ControlServer> controls;
    if (!o.headless || o.listen) controls = std::make_unique<demo::ControlServer>();
    std::unique_ptr<demo::Presenter> presenter;
    if (!o.headless) presenter = std::make_unique<demo::Presenter>(o.shape);
    std::unique_ptr<demo::CudaWorker> worker;
    std::vector<std::uint8_t> rgba(bytes, 0);
    if (presenter) presenter->upload(rgba);
    Mode mode = Mode::Pause;
    bool running = true, pending = false;
    std::uint32_t sequence = 0, slot = 0;
    std::string last_error;
    Statistics stats;
    bool animated = !o.static_scene;
    float held_seconds = 0, animation_offset = 0;
    const auto elapsed_animation = [&](Clock::time_point now) {
        return o.headless ? static_cast<float>(stats.frames) / 60
                          : std::chrono::duration<float>(now - stats.started).count();
    };
    const auto scene_seconds = [&](Clock::time_point now) {
        return animated ? animation_offset + elapsed_animation(now) : held_seconds;
    };
    const auto set_mode = [&](Mode next) {
        if (mode == next && ((next == Mode::CudaRay) == static_cast<bool>(worker)) &&
            (next != Mode::Pause || rgba.empty()))
            return;
        if (worker) {
            worker->stop();
            worker.reset();
        }
        pending = false;
        mode = Mode::Pause;
        if (next == Mode::Pause) {
            std::vector<std::uint8_t>().swap(rgba);
            return;
        }
        if (rgba.empty()) rgba.resize(bytes);
        if (next == Mode::CudaRay) {
            try {
                worker = std::make_unique<demo::CudaWorker>(o.worker, o.shape, o.denoise);
            } catch (...) {
                std::vector<std::uint8_t>().swap(rgba);
                throw;
            }
        }
        mode = next;
        stats.fps = 0;
        stats.last_render_ms = stats.last_copy_ms = stats.last_upload_ms = stats.last_present_ms =
            0;
        stats.last = Clock::now();
        last_error.clear();
        std::fprintf(stderr, "Cornell mode=%s cuda_worker_pid=%d\n", mode_name(mode).c_str(),
                     worker ? worker->pid() : 0);
    };
    const auto status = [&]() {
        std::ostringstream out;
        out << std::fixed << std::setprecision(3);
        out << "{\"mode\":" << json_string(mode_name(mode)) << ",\"scene\":\""
            << (o.scene == demo::SceneKind::Materials ? "materials" : "cornell") << '"'
            << ",\"animated\":" << (animated ? "true" : "false")
            << ",\"cuda_worker_pid\":" << (worker ? worker->pid() : 0)
            << ",\"width\":" << o.shape.width << ",\"height\":" << o.shape.height
            << ",\"samples\":" << o.shape.samples << ",\"bounces\":" << o.shape.bounces
            << ",\"frames\":" << stats.frames
            << ",\"fps\":" << (mode == Mode::Pause ? 0 : stats.fps)
            << ",\"last_render_ms\":" << stats.last_render_ms
            << ",\"last_copy_ms\":" << stats.last_copy_ms
            << ",\"last_upload_ms\":" << stats.last_upload_ms
            << ",\"last_present_ms\":" << stats.last_present_ms
            << ",\"denoise\":" << (worker && o.denoise ? "true" : "false")
            << ",\"explicit_vram_bytes\":" << (worker ? bytes + (o.denoise ? filtered_vram : 0) : 0)
            << ",\"pinned_ram_bytes\":" << (worker ? demo::shared_frame_bytes(o.shape) : 0)
            << ",\"cpu_frame_bytes\":" << rgba.size()
            << ",\"display\":" << (presenter ? "true" : "false");
        if (!last_error.empty()) out << ",\"last_error\":" << json_string(last_error);
        out << '}';
        return out.str();
    };
    const auto command = [&](const std::string& text) {
        if (text == "status") return status();
        if (text == "still" || text == "animate") {
            if (text == "still" && animated) {
                held_seconds = scene_seconds(Clock::now());
                animated = false;
            } else if (text == "animate" && !animated) {
                animation_offset = held_seconds - elapsed_animation(Clock::now());
                animated = true;
            }
            return status();
        }
        if (text == "stop") {
            set_mode(Mode::Pause);
            running = false;
            return status();
        }
        if (text == "snapshot") {
            if (!controls) throw std::runtime_error("control server unavailable");
            const auto path = controls->directory() / ("frame-" + std::to_string(getpid()) + "-" +
                                                       std::to_string(stats.frames) + ".ppm");
            save_ppm(path, o.shape, rgba);
            return "{\"snapshot\":" + json_string(path.string()) + "}";
        }
        set_mode(parse_mode(text));
        return status();
    };
    set_mode(o.mode);
    while (running && !interrupted) {
        const auto loop_start = Clock::now();
        if (o.seconds && loop_start - stats.started >= std::chrono::seconds(o.seconds)) break;
        if (controls) {
            // Bound command work so an authorized client cannot starve display events.
            for (int i = 0; i < 16; ++i) {
                const auto request = controls->next();
                if (!request) break;
                try {
                    controls->reply(*request, command(request->text));
                } catch (const std::exception& e) {
                    controls->reply(*request, "{\"error\":" + json_string(e.what()) + "}");
                }
            }
        }
        if (presenter)
            for (const auto& event : presenter->events()) {
                try {
                    static_cast<void>(command(event));
                } catch (const std::exception& e) {
                    last_error = e.what();
                    std::fprintf(stderr, "Cornell: %s\n", e.what());
                }
            }
        if (!running) break;
        bool complete = false;
        double render_ms = 0, copy_ms = 0;
        auto shape = o.shape;
        shape.seed = sequence + 1;
        const float seconds = scene_seconds(loop_start);
        if (mode == Mode::CudaRay) {
            try {
                if (!pending) {
                    demo::FrameMessage request;
                    request.slot = slot;
                    request.sequence = sequence;
                    request.seed = shape.seed;
                    request.samples = shape.samples;
                    request.bounces = shape.bounces;
                    request.seconds = seconds;
                    request.scene = static_cast<std::uint32_t>(o.scene);
                    worker->request(request);
                    pending = true;
                }
                if (worker->ready() || (o.headless && !o.listen)) {
                    const auto reply = worker->receive();
                    const auto pixels = worker->pixels(reply.slot);
                    std::copy(pixels.begin(), pixels.end(), rgba.begin());
                    pending = false;
                    complete = true;
                    slot ^= 1;
                    render_ms = static_cast<double>(reply.render_ns) / 1.0e6;
                    copy_ms = static_cast<double>(reply.copy_ns) / 1.0e6;
                    // Let the next GPU frame run while Intel uploads/presents
                    // this owned CPU copy and waits for the next vblank.
                    if (presenter && !(o.frames && stats.frames + 1 >= o.frames)) {
                        demo::FrameMessage next;
                        next.slot = slot;
                        next.sequence = sequence + 1;
                        next.seed = shape.seed + 1;
                        next.samples = shape.samples;
                        next.bounces = shape.bounces;
                        next.scene = static_cast<std::uint32_t>(o.scene);
                        next.seconds = scene_seconds(Clock::now());
                        worker->request(next);
                        pending = true;
                    }
                }
            } catch (const std::exception& e) {
                const std::string reason = e.what();
                set_mode(Mode::CpuRay);
                last_error = reason;
                std::fprintf(stderr, "Cornell CUDA failed; CPU fallback: %s\n", reason.c_str());
            }
        } else if (mode == Mode::CpuRay || mode == Mode::Raster) {
            const auto start = Clock::now();
            const auto scene = demo::make_scene(o.scene, seconds);
            if (mode == Mode::CpuRay)
                demo::cpu_ray_frame(scene, shape, rgba, o.threads);
            else
                demo::raster_frame(scene, shape, rgba);
            render_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            complete = true;
        }
        if (complete) {
            ++sequence;
            stats.completed(render_ms, copy_ms);
            if (presenter) {
                const auto upload_start = Clock::now();
                presenter->upload(rgba);
                stats.last_upload_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - upload_start).count();
            }
            if (o.frames && stats.frames >= o.frames) break;
        }
        if (presenter) {
            const auto present_start = Clock::now();
            presenter->present(mode_name(mode) + (animated ? " MOTION" : " STILL") +
                                   (worker && o.denoise ? " FILTERED" : "") +
                                   (last_error.empty() ? "" : " FALLBACK"),
                               mode == Mode::Pause ? 0 : stats.fps);
            stats.last_present_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - present_start).count();
        }
        // SDL waits for vsync on the monitor. Poll at 1 ms only for headless work.
        if (!presenter && !complete)
            std::this_thread::sleep_for(std::chrono::milliseconds(mode == Mode::Pause ? 10 : 1));
    }
    if (!o.output.empty()) save_ppm(o.output, o.shape, rgba);
    const auto finished = Clock::now();
    if (worker) {
        worker->stop();
        worker.reset();
    }
    const double elapsed = std::chrono::duration<double>(finished - stats.started).count();
    auto intervals = stats.intervals;
    std::sort(intervals.begin(), intervals.end());
    const double p95 = intervals.empty() ? 0 : intervals[(intervals.size() - 1) * 95 / 100];
    std::cout << std::fixed << std::setprecision(4) << "{\"mode\":" << json_string(mode_name(mode))
              << ",\"scene\":\""
              << (o.scene == demo::SceneKind::Materials ? "materials" : "cornell") << '"'
              << ",\"width\":" << o.shape.width << ",\"height\":" << o.shape.height
              << ",\"samples\":" << o.shape.samples << ",\"bounces\":" << o.shape.bounces
              << ",\"denoise\":" << (o.denoise && mode == Mode::CudaRay ? "true" : "false")
              << ",\"static\":" << (animated ? "false" : "true")
              << ",\"headless\":" << (o.headless ? "true" : "false")
              << ",\"frames\":" << stats.frames << ",\"elapsed_seconds\":" << elapsed
              << ",\"wall_fps\":" << (elapsed > 0 ? static_cast<double>(stats.frames) / elapsed : 0)
              << ",\"mean_render_ms\":"
              << (stats.frames ? stats.render_ms / static_cast<double>(stats.frames) : 0)
              << ",\"mean_copy_ms\":"
              << (stats.frames ? stats.copy_ms / static_cast<double>(stats.frames) : 0)
              << ",\"p95_last600_frame_ms\":" << p95 << ",\"cuda_worker_reaped\":true}" << '\n';
    return EXIT_SUCCESS;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        return run(options(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "eclair-rt: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
