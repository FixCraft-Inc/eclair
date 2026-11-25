/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// Shared pieces of the CUDA tests: an error check that abandons the test by
// throwing, RAII owners for streams, events and device memory, the evidence
// header line, and the benchmark timing method. C++17, for CUDA units.

#include "check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace eclair::test {

// Unexpected CUDA failures abandon the test. contain() turns them into a
// failed exit status.
inline void checked(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class Stream {
public:
    Stream() {
        checked(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking), "stream create");
    }
    ~Stream() {
        if (cudaStreamDestroy(value_) != cudaSuccess) {
            record_failure("stream destroy", __FILE__, __LINE__);
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return value_; }

private:
    cudaStream_t value_ = nullptr;
};

class Event {
public:
    Event() { checked(cudaEventCreate(&value_), "event create"); }
    ~Event() {
        if (cudaEventDestroy(value_) != cudaSuccess) {
            record_failure("event destroy", __FILE__, __LINE__);
        }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    [[nodiscard]] cudaEvent_t get() const noexcept { return value_; }

private:
    cudaEvent_t value_ = nullptr;
};

// Device memory for `count` elements of T. Tests offset into it to build
// views that miss a kernel's vector alignment.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error("allocation size overflow");
        }
        if (count != 0U) {
            checked(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "allocation");
        }
    }
    // A device copy of `values`, uploaded on `stream`.
    DeviceBuffer(const std::vector<T>& values, cudaStream_t stream) : DeviceBuffer(values.size()) {
        upload(values, stream);
    }
    ~DeviceBuffer() {
        if (data_ != nullptr && cudaFree(data_) != cudaSuccess) {
            record_failure("device free", __FILE__, __LINE__);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    // Enqueues the copy. CUDA stages pageable host memory, which every test
    // vector is, before returning, so `values` may be released right after.
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        if (values.size() != count_) {
            throw std::runtime_error("upload size mismatch");
        }
        if (count_ != 0U) {
            checked(cudaMemcpyAsync(data_, values.data(), count_ * sizeof(T),
                                    cudaMemcpyHostToDevice, stream),
                    "upload");
        }
    }
    // Copies everything back after the stream's earlier work completes.
    [[nodiscard]] std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> result(count_);
        if (count_ != 0U) {
            checked(cudaMemcpyAsync(result.data(), data_, count_ * sizeof(T),
                                    cudaMemcpyDeviceToHost, stream),
                    "download");
        }
        checked(cudaStreamSynchronize(stream), "stream synchronize");
        return result;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// The device, driver and toolchain line every GPU result starts with.
inline void print_device_header() {
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, 0), "device properties");
    int driver = 0;
    int runtime = 0;
    checked(cudaDriverGetVersion(&driver), "driver version");
    checked(cudaRuntimeGetVersion(&runtime), "runtime version");
    std::printf("device=%s sm_%d%d driver_api=%d runtime=%d nvcc=%d.%d.%d gcc=%d.%d\n",
                properties.name, properties.major, properties.minor, driver, runtime,
                __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__, __GNUC__,
                __GNUC_MINOR__);
}

// Nine CUDA-event samples of `launch` on `stream`, sorted, in milliseconds:
// [4] is the median and the last is the maximum. A bounded 100 ms warmup
// first, since short isolated launches otherwise measure idle clocks. GPU
// clocks are left alone.
template <typename Launch>
[[nodiscard]] std::array<float, 9> timings(cudaStream_t stream, Launch launch) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    do {
        launch();
        checked(cudaStreamSynchronize(stream), "warmup synchronize");
    } while (std::chrono::steady_clock::now() < deadline);
    Event start;
    Event stop;
    std::array<float, 9> samples{};
    for (float& sample : samples) {
        checked(cudaEventRecord(start.get(), stream), "start timing");
        launch();
        checked(cudaEventRecord(stop.get(), stream), "stop timing");
        checked(cudaEventSynchronize(stop.get()), "timing synchronize");
        checked(cudaEventElapsedTime(&sample, start.get(), stop.get()), "elapsed time");
    }
    std::sort(samples.begin(), samples.end());
    return samples;
}

// Runs a test body and returns main()'s status: failure for an abandoned
// test or any failed check.
template <typename Body>
[[nodiscard]] int contain(Body body) {
    try {
        body();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
    return finish();
}

}  // namespace eclair::test
