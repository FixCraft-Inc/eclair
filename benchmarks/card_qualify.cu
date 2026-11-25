/*
 * ECLAIR - Extensible Compute Layer for Architecture-Independent Runtime
 * Copyright (C) 2026 FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 * Independent hardware qualification tooling. No Eclair runtime dependency.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
volatile std::sig_atomic_t stopped = 0;
bool cleanup_failed = false;
void on_signal(int) { stopped = 1; }
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(status));
}
void cleanup(cudaError_t status) noexcept {
    if (status != cudaSuccess) {
        cleanup_failed = true;
        std::fputs("CUDA cleanup failure\n", stderr);
    }
}
void cancelled() {
    if (stopped) throw std::runtime_error("qualification cancelled");
}
std::size_t number(std::string_view input, std::size_t low, std::size_t high) {
    std::size_t result = 0;
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size()
        || result < low || result > high) throw std::invalid_argument("argument out of range");
    return result;
}

class Stream {
public:
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cleanup(cudaStreamDestroy(value)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t value{};
    void wait() const { check(cudaStreamSynchronize(value)); }
};
template<class T> class Buffer {
public:
    explicit Buffer(std::size_t count) : size(count) {
        if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("allocation size out of range");
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Buffer() { cleanup(cudaFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data{};
    const std::size_t size;
};
class Event {
public:
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cleanup(cudaEventDestroy(value)); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t value{};
};
class Blas {
public:
    explicit Blas(cudaStream_t stream) {
        check(cublasCreate(&value));
        const auto result = cublasSetStream(value, stream);
        if (result != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(value);
            check(result);
        }
    }
    ~Blas() {
        if (cublasDestroy(value) != CUBLAS_STATUS_SUCCESS) {
            cleanup_failed = true;
            std::fputs("cuBLAS cleanup failure\n", stderr);
        }
    }
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    cublasHandle_t value{};
};

__device__ std::uint64_t device_pattern(std::size_t i, int pattern, int pass) {
    if (pattern == 0) return 0;
    if (pattern == 1) return ~std::uint64_t{0};
    if (pattern == 2) return 0xaaaaaaaaaaaaaaaaULL;
    if (pattern == 3) return 0x5555555555555555ULL;
    if (pattern == 4) return static_cast<std::uint64_t>(i) ^ 0x97f04a7c15ULL;
    if (pattern == 5) return ~(static_cast<std::uint64_t>(i) ^ 0x97f04a7c15ULL);
    if (pattern == 6) {
        auto x = static_cast<std::uint64_t>(i) + 0x9e3779b97f4a7c15ULL
                 + static_cast<std::uint64_t>(pass);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    const auto bit = std::uint64_t{1} << ((pattern - 7) % 64);
    return pattern < 71 ? bit : ~bit;
}
__global__ void fill_pattern(std::uint64_t* data, std::size_t size, int pattern, int pass) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) data[i] = device_pattern(i, pattern, pass);
}
__global__ void verify_pattern(const std::uint64_t* data, std::size_t size,
                               int pattern, int pass, unsigned long long* errors) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        if (data[i] != device_pattern(i, pattern, pass)) atomicAdd(errors, 1ULL);
    }
}
std::uint64_t host_pattern(std::size_t i, int pattern, int pass) {
    // Independent CPU checks make pattern generation observable as well as reads.
    switch (pattern) {
    case 0: return 0;
    case 1: return UINT64_MAX;
    case 2: return 0xaaaaaaaaaaaaaaaaULL;
    case 3: return 0x5555555555555555ULL;
    case 4: return i ^ 0x97f04a7c15ULL;
    case 5: return UINT64_MAX ^ (i ^ 0x97f04a7c15ULL);
    case 6: {
        std::uint64_t value = i + std::uint64_t{0x9e3779b97f4a7c15ULL} + pass;
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }
    default:
        if (pattern < 71) return std::uint64_t{1} << (pattern - 7);
        return UINT64_MAX ^ (std::uint64_t{1} << (pattern - 71));
    }
}
unsigned long long verify(const Buffer<std::uint64_t>& data,
                           Buffer<unsigned long long>& errors, const Stream& stream,
                           int pattern, int pass) {
    check(cudaMemsetAsync(errors.data, 0, sizeof(unsigned long long), stream.value));
    verify_pattern<<<4096, 256, 0, stream.value>>>(data.data, data.size, pattern, pass, errors.data);
    check(cudaGetLastError());
    stream.wait();
    unsigned long long count = 0;
    check(cudaMemcpy(&count, errors.data, sizeof(count), cudaMemcpyDeviceToHost));
    return count;
}
void memory(int passes, bool self_test) {
    Stream stream;
    std::size_t available = 0, total = 0;
    check(cudaMemGetInfo(&available, &total));
    constexpr std::size_t reserve = std::size_t{1} << 30;
    if (available <= reserve) throw std::runtime_error("insufficient free VRAM");
    const std::size_t bytes = self_test ? 8192 : ((available - reserve) / (64U << 20)) * (64U << 20);
    Buffer<std::uint64_t> data(bytes / sizeof(std::uint64_t));
    Buffer<unsigned long long> errors(1);
    std::printf("{\"event\":\"memory_begin\",\"bytes\":%zu,\"available\":%zu,\"total\":%zu}\n",
                bytes, available, total);
    if (self_test) {
        fill_pattern<<<4, 256, 0, stream.value>>>(data.data, data.size, 1, 0);
        check(cudaGetLastError());
        check(cudaMemsetAsync(data.data + 17, 0, sizeof(std::uint64_t), stream.value));
        if (verify(data, errors, stream, 1, 0) != 1)
            throw std::runtime_error("memory scanner failed its injected-error control");
        return;
    }
    for (int pass = 0; pass < passes; ++pass) {
        for (int pattern = 0; pattern < 135; ++pattern) {
            cancelled();
            fill_pattern<<<4096, 256, 0, stream.value>>>(data.data, data.size, pattern, pass);
            check(cudaGetLastError());
            stream.wait();
            if (pattern == 6) std::this_thread::sleep_for(std::chrono::seconds(2));
            const auto count = verify(data, errors, stream, pattern, pass);
            if (count != 0) throw std::runtime_error("VRAM mismatches: " + std::to_string(count));
            for (std::size_t sample = 0; sample < 16; ++sample) {
                const std::size_t i = sample * (data.size - 1) / 15;
                std::uint64_t actual = 0;
                check(cudaMemcpy(&actual, data.data + i, sizeof(actual), cudaMemcpyDeviceToHost));
                if (actual != host_pattern(i, pattern, pass))
                    throw std::runtime_error("VRAM CPU-reference mismatch");
            }
            std::printf("{\"event\":\"memory_pattern\",\"pass\":%d,\"pattern\":%d,\"errors\":0}\n",
                        pass, pattern);
        }
    }
}
void timing_report(const char* name, std::vector<float> times, double work) {
    if (times.empty()) throw std::runtime_error("no timing samples");
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double p95 = times[(times.size() * 95 + 99) / 100 - 1];
    if (!(median > 0)) throw std::runtime_error("invalid event timing");
    std::printf("{\"event\":\"benchmark\",\"baseline\":\"%s\",\"samples\":%zu,"
                "\"median_ms\":%.6f,\"p95_ms\":%.6f,\"work_units\":%.0f,\"units_per_second\":%.3f}\n",
                name, times.size(), median, p95, work, work / (median / 1000.0));
}
void bandwidth() {
    constexpr std::size_t bytes = std::size_t{256} << 20;
    Stream stream;
    Buffer<std::uint8_t> first(bytes), second(bytes);
    struct Pinned {
        void* data{};
        Pinned() { check(cudaMallocHost(&data, bytes)); }
        ~Pinned() { cleanup(cudaFreeHost(data)); }
        Pinned(const Pinned&) = delete;
        Pinned& operator=(const Pinned&) = delete;
    } host;
    std::memset(host.data, 0xa5, bytes);
    Event start, end;
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<float> times;
        for (int run = -5; run < 30; ++run) {
            cancelled();
            check(cudaEventRecord(start.value, stream.value));
            if (mode == 0) check(cudaMemcpyAsync(first.data, host.data, bytes, cudaMemcpyHostToDevice, stream.value));
            if (mode == 1) check(cudaMemcpyAsync(host.data, first.data, bytes, cudaMemcpyDeviceToHost, stream.value));
            if (mode == 2) check(cudaMemcpyAsync(second.data, first.data, bytes, cudaMemcpyDeviceToDevice, stream.value));
            check(cudaEventRecord(end.value, stream.value));
            check(cudaEventSynchronize(end.value));
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start.value, end.value));
            if (run >= 0) times.push_back(ms);
        }
        if (mode == 2) check(cudaMemcpy(host.data, second.data, bytes, cudaMemcpyDeviceToHost));
        const auto* ptr = static_cast<const std::uint8_t*>(host.data);
        for (std::size_t i = 0; i < bytes; ++i)
            if (ptr[i] != 0xa5) throw std::runtime_error("transfer correctness mismatch");
        timing_report(mode == 0 ? "CUDA_H2D_pinned" : mode == 1 ? "CUDA_D2H_pinned" : "CUDA_D2D",
                      std::move(times), static_cast<double>(bytes) * (mode == 2 ? 2.0 : 1.0));
    }
}
float input_value(std::size_t i, std::uint32_t seed) {
    std::uint32_t x = static_cast<std::uint32_t>(i) ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    return static_cast<float>(static_cast<int>(x % 17) - 8) / 16.0F;
}
class MixedMemory {
    static std::size_t words() {
        std::size_t available = 0, total = 0;
        check(cudaMemGetInfo(&available, &total));
        constexpr std::size_t reserve = std::size_t{1} << 30;
        if (available <= reserve + (64U << 20))
            throw std::runtime_error("insufficient VRAM for mixed stress");
        return ((available - reserve) / (64U << 20)) * (64U << 20) / sizeof(std::uint64_t);
    }
public:
    MixedMemory() : data_(words()), errors_(1) {
        std::printf("{\"event\":\"mixed_memory_begin\",\"bytes\":%zu}\n",
                    data_.size * sizeof(std::uint64_t));
        start();
    }
    ~MixedMemory() {
        if (pending_) cleanup(cudaStreamSynchronize(stream_.value));
    }
    MixedMemory(const MixedMemory&) = delete;
    MixedMemory& operator=(const MixedMemory&) = delete;
    void start() {
        if (pending_) throw std::logic_error("mixed scan already pending");
        check(cudaMemsetAsync(errors_.data, 0, sizeof(unsigned long long), stream_.value));
        const int pattern = scans_ % 135, pass = scans_ / 135;
        fill_pattern<<<4096, 256, 0, stream_.value>>>(data_.data, data_.size, pattern, pass);
        check(cudaGetLastError());
        verify_pattern<<<4096, 256, 0, stream_.value>>>(data_.data, data_.size, pattern, pass, errors_.data);
        check(cudaGetLastError());
        check(cudaEventRecord(finished_.value, stream_.value));
        pending_ = true;
    }
    bool collect(bool wait) {
        if (!pending_) return true;
        const auto state = wait ? cudaEventSynchronize(finished_.value)
                                : cudaEventQuery(finished_.value);
        if (state == cudaErrorNotReady) return false;
        check(state);
        unsigned long long errors = 0;
        check(cudaMemcpy(&errors, errors_.data, sizeof(errors), cudaMemcpyDeviceToHost));
        if (errors != 0) throw std::runtime_error("mixed VRAM mismatches: " + std::to_string(errors));
        for (std::size_t point = 0; point < 16; ++point) {
            const auto index = point * (data_.size - 1) / 15;
            std::uint64_t actual = 0;
            check(cudaMemcpy(&actual, data_.data + index, sizeof(actual), cudaMemcpyDeviceToHost));
            if (actual != host_pattern(index, scans_ % 135, scans_ / 135))
                throw std::runtime_error("mixed VRAM CPU-reference mismatch");
        }
        ++scans_;
        pending_ = false;
        return true;
    }
    void finish() {
        collect(true);
        if (scans_ < 135) throw std::runtime_error("incomplete mixed memory pattern coverage");
        std::printf("{\"event\":\"mixed_memory_result\",\"scans\":%d,\"errors\":0}\n", scans_);
    }
private:
    Stream stream_;
    Event finished_;
    Buffer<std::uint64_t> data_;
    Buffer<unsigned long long> errors_;
    bool pending_ = false;
    int scans_ = 0;
};

void gemm(bool half, int seconds, int duty_percent, int n, bool mixed = false) {
    Stream stream;
    Blas handle(stream.value);
    check(cublasSetMathMode(handle.value, half ? CUBLAS_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
    const std::size_t count = static_cast<std::size_t>(n) * n;
    std::vector<float> a(count), b(count), output(count);
    std::vector<__half> ah(count), bh(count);
    for (std::size_t i = 0; i < count; ++i) {
        a[i] = input_value(i, 0x1234U); b[i] = input_value(i, 0x7139U);
        ah[i] = __float2half_rn(a[i]); bh[i] = __float2half_rn(b[i]);
    }
    Buffer<float> da(count), db(count), dc(count);
    Buffer<__half> dah(count), dbh(count);
    check(cudaMemcpy(da.data, a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(db.data, b.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dah.data, ah.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dbh.data, bh.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
    const float alpha = 1, beta = 0;
    auto launch = [&] {
        if (half) check(cublasGemmEx(handle.value, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            &alpha, dah.data, CUDA_R_16F, n, dbh.data, CUDA_R_16F, n,
            &beta, dc.data, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        else check(cublasSgemm(handle.value, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            &alpha, da.data, n, db.data, n, &beta, dc.data, n));
    };
    auto validate = [&] {
        stream.wait();
        check(cudaMemcpy(output.data(), dc.data, count * sizeof(float), cudaMemcpyDeviceToHost));
        const int points = n <= 128 ? n : 16;
        for (int y = 0; y < points; ++y) for (int x = 0; x < points; ++x) {
            const int row = x * (n - 1) / (points - 1), col = y * (n - 1) / (points - 1);
            double expected = 0;
            for (int k = 0; k < n; ++k)
                expected += static_cast<double>(a[static_cast<std::size_t>(row) + static_cast<std::size_t>(k) * n])
                            * b[static_cast<std::size_t>(k) + static_cast<std::size_t>(col) * n];
            if (static_cast<double>(output[static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * n]) != expected)
                throw std::runtime_error("GEMM CPU-reference mismatch");
        }
    };
    for (int warm = 0; warm < 5; ++warm) launch();
    validate();
    std::unique_ptr<MixedMemory> traffic;
    if (mixed) traffic = std::make_unique<MixedMemory>();
    Event start, end;
    std::vector<float> times;
    const auto began = std::chrono::steady_clock::now();
    auto last_check = began;
    do {
        cancelled();
        check(cudaEventRecord(start.value, stream.value));
        for (int batch = 0; batch < 8; ++batch) launch();
        check(cudaEventRecord(end.value, stream.value));
        check(cudaEventSynchronize(end.value));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start.value, end.value));
        if (times.size() < 100000) times.push_back(ms / 8.0F);
        else throw std::runtime_error("timing sample bound exceeded");
        if (std::chrono::steady_clock::now() - last_check >= std::chrono::seconds(10)) {
            validate(); last_check = std::chrono::steady_clock::now();
        }
        if (duty_percent < 100)
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms * (100.0 / duty_percent - 1.0)));
        if (traffic && traffic->collect(false)) traffic->start();
    } while (std::chrono::steady_clock::now() - began < std::chrono::seconds(seconds));
    if (traffic) traffic->finish();
    validate();
    std::printf("{\"event\":\"gemm_correctness\",\"n\":%d,\"precision\":\"%s\","
                "\"absolute_error\":0,\"checked_points\":%d,\"seconds\":%d,\"duty_percent\":%d}\n",
                n, half ? "fp16_fp32_accum" : "fp32", n <= 128 ? n * n : 256, seconds, duty_percent);
    timing_report(half ? "cuBLAS_GemmEx_FP16_FP32_TensorOp_requested" : "cuBLAS_SGEMM_pedantic",
                  std::move(times), 2.0 * n * n * n);
}
} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
    try {
        int device_count = 0;
        check(cudaGetDeviceCount(&device_count));
        if (device_count != 1) throw std::runtime_error("expected exactly one GPU");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        if (properties.major != 7 || properties.minor != 0)
            throw std::runtime_error("qualification is restricted to the Volta target");
        if (argc == 2 && std::string_view(argv[1]) == "self-test") {
            memory(1, true); gemm(false, 0, 100, 128); gemm(true, 0, 100, 128);
        } else if (argc == 3 && std::string_view(argv[1]) == "memory") {
            memory(static_cast<int>(number(argv[2], 1, 3)), false);
        } else if (argc == 2 && std::string_view(argv[1]) == "bandwidth") {
            bandwidth();
        } else if (argc == 3 && std::string_view(argv[1]) == "mixed") {
            gemm(true, static_cast<int>(number(argv[2], 30, 600)), 100, 4096, true);
        } else if (argc == 5 && std::string_view(argv[1]) == "gemm") {
            const std::string_view precision(argv[2]);
            if (precision != "fp16" && precision != "fp32") throw std::invalid_argument("invalid precision");
            gemm(precision == "fp16", static_cast<int>(number(argv[3], 1, 600)),
                 static_cast<int>(number(argv[4], 10, 100)), 4096);
        } else throw std::invalid_argument("usage: card_qualify self-test | memory PASSES | bandwidth | mixed SECONDS | gemm fp16|fp32 SECONDS DUTY_PERCENT");
        if (cleanup_failed) return 1;
        std::puts("{\"event\":\"complete\",\"passed\":true}");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "qualification failed: %s\n", error.what());
        return 1;
    }
}
