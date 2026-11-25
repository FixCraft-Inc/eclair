/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#include "presenter.hpp"
#include <SDL.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace eclair::demo {
namespace {
void checked(int status, const char* operation) {
    if (status < 0) throw std::runtime_error(std::string(operation) + ": " + SDL_GetError());
}
std::string intel_card() {
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm")) {
        const auto name = entry.path().filename().string();
        if (!name.starts_with("card") || name.find('-') != std::string::npos) continue;
        std::error_code error;
        const auto device = std::filesystem::canonical(entry.path() / "device", error);
        if (!error && device.filename() == "0000:00:02.0") {
            const auto driver = std::filesystem::canonical(entry.path() / "device/driver", error);
            if (error || driver.filename() != "i915")
                throw std::runtime_error("Intel display driver is not active");
            return name.substr(4);
        }
    }
    throw std::runtime_error("Intel motherboard display device not found");
}
// Independently drawn 5x7 glyphs for the small diagnostic overlay.
std::array<unsigned, 7> glyph(char c) {
    switch (c) {
        case 'A':
            return {14, 17, 17, 31, 17, 17, 17};
        case 'B':
            return {30, 17, 17, 30, 17, 17, 30};
        case 'C':
            return {14, 17, 16, 16, 16, 17, 14};
        case 'D':
            return {30, 17, 17, 17, 17, 17, 30};
        case 'E':
            return {31, 16, 16, 30, 16, 16, 31};
        case 'F':
            return {31, 16, 16, 30, 16, 16, 16};
        case 'G':
            return {14, 17, 16, 23, 17, 17, 14};
        case 'H':
            return {17, 17, 17, 31, 17, 17, 17};
        case 'I':
            return {31, 4, 4, 4, 4, 4, 31};
        case 'J':
            return {7, 2, 2, 2, 18, 18, 12};
        case 'K':
            return {17, 18, 20, 24, 20, 18, 17};
        case 'L':
            return {16, 16, 16, 16, 16, 16, 31};
        case 'M':
            return {17, 27, 21, 21, 17, 17, 17};
        case 'N':
            return {17, 25, 21, 19, 17, 17, 17};
        case 'O':
            return {14, 17, 17, 17, 17, 17, 14};
        case 'P':
            return {30, 17, 17, 30, 16, 16, 16};
        case 'Q':
            return {14, 17, 17, 17, 21, 18, 13};
        case 'R':
            return {30, 17, 17, 30, 20, 18, 17};
        case 'S':
            return {15, 16, 16, 14, 1, 1, 30};
        case 'T':
            return {31, 4, 4, 4, 4, 4, 4};
        case 'U':
            return {17, 17, 17, 17, 17, 17, 14};
        case 'V':
            return {17, 17, 17, 17, 17, 10, 4};
        case 'W':
            return {17, 17, 17, 21, 21, 27, 17};
        case 'X':
            return {17, 17, 10, 4, 10, 17, 17};
        case 'Y':
            return {17, 17, 10, 4, 4, 4, 4};
        case 'Z':
            return {31, 1, 2, 4, 8, 16, 31};
        case '0':
            return {14, 17, 19, 21, 25, 17, 14};
        case '1':
            return {4, 12, 4, 4, 4, 4, 14};
        case '2':
            return {14, 17, 1, 2, 4, 8, 31};
        case '3':
            return {30, 1, 1, 14, 1, 1, 30};
        case '4':
            return {2, 6, 10, 18, 31, 2, 2};
        case '5':
            return {31, 16, 16, 30, 1, 1, 30};
        case '6':
            return {14, 16, 16, 30, 17, 17, 14};
        case '7':
            return {31, 1, 2, 4, 8, 8, 8};
        case '8':
            return {14, 17, 17, 14, 17, 17, 14};
        case '9':
            return {14, 17, 17, 15, 1, 1, 14};
        case '-':
            return {0, 0, 0, 31, 0, 0, 0};
        case '.':
            return {0, 0, 0, 0, 0, 12, 12};
        default:
            return {};
    }
}
void text(SDL_Renderer* renderer, std::string_view value, int x, int y) {
    checked(SDL_SetRenderDrawColor(renderer, 225, 230, 232, 255), "text color");
    for (char c : value) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
        const auto rows = glyph(c);
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5; ++col)
                if ((rows[static_cast<std::size_t>(row)] &
                     (1U << static_cast<unsigned>(4 - col))) != 0) {
                    const SDL_Rect pixel{x + col * 2, y + row * 2, 2, 2};
                    checked(SDL_RenderFillRect(renderer, &pixel), "draw text");
                }
        x += 12;
    }
}
}  // namespace
struct Presenter::Impl {
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;
    ops::RayTraceShape shape;
    ~Impl() {
        if (texture) SDL_DestroyTexture(texture);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }
};
Presenter::Presenter(const ops::RayTraceShape& shape) : impl_(std::make_unique<Impl>()) {
    impl_->shape = shape;
    const char* video = std::getenv("SDL_VIDEODRIVER");
    if (video && std::string_view(video) == "KMSDRM") {
        const auto index = intel_card();
        if (SDL_SetHintWithPriority("SDL_KMSDRM_DEVICE_INDEX", index.c_str(), SDL_HINT_OVERRIDE) !=
            SDL_TRUE)
            throw std::runtime_error("cannot select Intel display");
    }
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
    checked(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS), "initialize SDL");
    const bool dummy = std::string_view(SDL_GetCurrentVideoDriver()) == "dummy";
    const int width = static_cast<int>(shape.width), height = static_cast<int>(shape.height);
    const Uint32 flags =
        (!video || std::string_view(video) != "KMSDRM") ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP;
    impl_->window = SDL_CreateWindow("Eclair Cornell preview", SDL_WINDOWPOS_CENTERED,
                                     SDL_WINDOWPOS_CENTERED, width, height + 56, flags);
    if (!impl_->window) throw std::runtime_error(SDL_GetError());
    impl_->renderer = SDL_CreateRenderer(
        impl_->window, -1,
        dummy ? SDL_RENDERER_SOFTWARE : SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!impl_->renderer) throw std::runtime_error(SDL_GetError());
    checked(SDL_RenderSetLogicalSize(impl_->renderer, width, height + 56), "logical display size");
    impl_->texture = SDL_CreateTexture(impl_->renderer, SDL_PIXELFORMAT_RGBA32,
                                       SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!impl_->texture) throw std::runtime_error(SDL_GetError());
    SDL_RendererInfo info{};
    checked(SDL_GetRendererInfo(impl_->renderer, &info), "renderer info");
    SDL_DisplayMode mode{};
    checked(SDL_GetCurrentDisplayMode(0, &mode), "display mode");
    std::fprintf(stderr, "Cornell display video=%s renderer=%s mode=%dx%d refresh=%d vsync=%s\n",
                 SDL_GetCurrentVideoDriver(), info.name, mode.w, mode.h, mode.refresh_rate,
                 (info.flags & SDL_RENDERER_PRESENTVSYNC) ? "yes" : "no");
    if (info.name && std::string_view(info.name).starts_with("opengl")) {
        using GetString = const unsigned char* (*)(unsigned);
        const auto get_string = reinterpret_cast<GetString>(SDL_GL_GetProcAddress("glGetString"));
        const auto* device = get_string ? get_string(0x1f01U) : nullptr;  // GL_RENDERER
        if (device)
            std::fprintf(stderr, "Cornell display GL device=%s\n",
                         reinterpret_cast<const char*>(device));
    }
    SDL_ShowCursor(SDL_DISABLE);
}
Presenter::~Presenter() = default;
std::vector<std::string> Presenter::events() {
    std::vector<std::string> commands;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) commands.emplace_back("stop");
        if (event.type != SDL_KEYDOWN || event.key.repeat) continue;
        switch (event.key.keysym.sym) {
            case SDLK_1:
                commands.emplace_back("raster");
                break;
            case SDLK_2:
                commands.emplace_back("cpu-ray");
                break;
            case SDLK_3:
                commands.emplace_back("cuda-ray");
                break;
            case SDLK_SPACE:
                commands.emplace_back("pause");
                break;
            case SDLK_s:
                commands.emplace_back("still");
                break;
            case SDLK_a:
                commands.emplace_back("animate");
                break;
            case SDLK_ESCAPE:
            case SDLK_q:
                commands.emplace_back("stop");
                break;
            default:
                break;
        }
    }
    return commands;
}
void Presenter::upload(std::span<const std::uint8_t> rgba) {
    std::size_t bytes = 0;
    if (ops::ray_frame_bytes(impl_->shape, &bytes) != ops::Status::Success || rgba.size() < bytes)
        throw std::invalid_argument("display frame too small");
    checked(SDL_UpdateTexture(impl_->texture, nullptr, rgba.data(),
                              static_cast<int>(impl_->shape.width * 4)),
            "upload display frame");
}
void Presenter::present(const std::string& mode, double fps) {
    auto* renderer = impl_->renderer;
    checked(SDL_SetRenderDrawColor(renderer, 12, 15, 18, 255), "background");
    checked(SDL_RenderClear(renderer), "clear");
    const SDL_Rect destination{0, 28, static_cast<int>(impl_->shape.width),
                               static_cast<int>(impl_->shape.height)};
    checked(SDL_RenderCopy(renderer, impl_->texture, nullptr, &destination), "present frame");
    char header[128];
    std::snprintf(header, sizeof(header), "ECLAIR CORNELL  %s  %.1F FPS", mode.c_str(), fps);
    text(renderer, header, 8, 7);
    text(renderer, "1 RASTER  2 CPU RAY  3 V100  SPACE PAUSE  ESC QUIT", 8,
         static_cast<int>(impl_->shape.height) + 35);
    SDL_RenderPresent(renderer);
}
}  // namespace eclair::demo
