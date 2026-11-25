/*
 * ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
 * Copyright (C) 2026  FixCraft Inc.
 * Licensed under the GNU Affero General Public License v3.0 or later.
 */

#pragma once

// cuBLAS baselines for CUDA tests: a status check that abandons the test and
// an RAII handle. Only tests that link CUDA::cublas include this. C++17.

#include "check.hpp"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace eclair::test {

inline void checked_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": status " + std::to_string(status));
    }
}

class Blas {
public:
    Blas() { checked_blas(cublasCreate(&value_), "cuBLAS create"); }
    ~Blas() {
        if (cublasDestroy(value_) != CUBLAS_STATUS_SUCCESS) {
            record_failure("cuBLAS destroy", __FILE__, __LINE__);
        }
    }
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;
    [[nodiscard]] cublasHandle_t get() const noexcept { return value_; }

private:
    cublasHandle_t value_ = nullptr;
};

}  // namespace eclair::test
