# Contributing to Eclair

Use supported CUDA interfaces and keep NVIDIA's driver and the reported GPU
identity intact. Tie capability and performance claims to tests. Keep network
access and downloads explicit. Use AGPL-3.0-or-later for original Eclair code
and identify third-party code and its license. Google's
[C++](https://google.github.io/styleguide/cppguide.html),
[Python](https://google.github.io/styleguide/pyguide.html) and
[shell](https://google.github.io/styleguide/shellguide.html) guides inform the
rules below.

## Layers

Each layer is its own CMake library, and dependencies only point down. A new
dependency needs a reason in the review, never a transitive include or link.

| Library | Owns | May depend on |
| --- | --- | --- |
| `eclair_core` | Numerical formats and portable reference conversion (`formats/`); shared render vector, material, BSDF and geometry math | Standard library |
| `eclair_ops` | Operation shapes, buffer extents, validation and numerical contracts (`ops/`) | core |
| `eclair_capability` | Feature catalog, report types, enable/disable policy (`capability/`) | Standard library |
| `eclair_runtime` | Device-independent operation selection (`runtime/`) | ops, capability |
| `eclair_backend_host` | Host reference backend and its known-answer probes (`host/`) | core, ops, capability |
| `eclair_cuda_architecture` | NVIDIA compute-capability rules, host-only and testable without CUDA | capability |
| `eclair_cuda` | CUDA kernels, device queries, the CUDA backend probe | core, ops, capability, CUDA architecture rules, CUDA runtime |
| `eclair_cuda_blas` | CUDA operations that finish in a cuBLAS call Eclair configures per call, currently the large-M W4A16 linear | cuda, cuBLAS |
| `eclair_smi_core` | `eclair-smi` options, terminal style, text and JSON rendering (`cli/eclair-smi/`) | capability |
| `eclair_render_scene` | Demo Cornell/material scenes, threaded CPU tracing and raster baseline (`cli/eclair-rt/`) | ops, shared render math, threads |
| `eclair_render_ipc` | Bounded Linux frame storage and worker ownership; no CUDA | ops, Linux system interfaces |
| `eclair-rt` (executable) | Controls, scheduling, measurements, optional SDL2/Intel presentation | render_scene, render_ipc, optional SDL2 |
| `eclair-rt-cuda-worker` (executable) | Demo-owned CUDA allocations, events, frame copies and preview history | render_scene, render_ipc, cuda |
| `eclair_kernels` (shared) | C interface for loaders such as Python's ctypes (`include/eclair/c/`, `src/eclair/c/`). Exports only `eclair_*` C functions, contains every exception, links the CUDA runtime and cuBLAS dynamically | cuda, cuda_blas |

Rules that follow from the table:

- Lower layers never depend on a CLI, framework adapter, benchmark policy or
  presentation. `capability/` names no vendor. Only `cli/eclair-smi/backends_*.cpp`
  choose which backends a build contains.
- A backend reports what it can run. It never claims a feature without an
  implementation and a probe: the host backend runs known-answer checks, and
  the CUDA backend checks that a loadable kernel image exists for the device.
- A new architecture or vendor is a new backend behind the same report types,
  with its rules in a host-testable library like `eclair_cuda_architecture`.
  A non-target GPU used for validation does not become a backend or change
  V100 dispatch.
- Adding a feature means one catalog entry with a stable key, a state from every
  backend, tests for each state, and a `docs/ECLAIR_SMI.md` update. Keys are
  never reused for a different operation.
- Operation contracts share `Status` and `BufferView` from
  `include/eclair/ops/buffer.hpp`, and linear plans share one capability and
  policy decision in `src/eclair/runtime/linear_selection.hpp`.

## C++ and CUDA

| Area | Eclair decision |
| --- | --- |
| Standard | C++20 for host code. CUDA translation units use C++17 (the CUDA 12 and CMake 3.25 envelope in `docs/DEBIAN_V100.md`), so headers a `.cu` file includes stay C++17. Avoid library features GCC 12 lacks, such as `<format>` |
| Formatting | `.clang-format` (Google base, 4-space indent, 100 columns, left pointers, include order preserved). Format the lines you change. Use clang-format with the checked-in configuration |
| Names | Types and enumerators `PascalCase`, functions and variables `snake_case`, constants `kPascalCase`, macros `ECLAIR_ALL_CAPS`, private members a trailing underscore. Google differs for functions (`PascalCase`) and enumerators (`kName`) |
| Headers | `.hpp` with `#pragma once`, self-contained, own header first. Include what you use. Public headers live in `include/eclair/<layer>/` |
| Integers | `std::size_t` for sizes and counts, fixed-width types for formats. Check narrowing and subtraction before they wrap |
| Errors | Expected outcomes are return values the caller must check: `cudaError_t` at CUDA seams, `std::optional<std::string>` for validation messages, report states for capability. Exceptions only abandon an operation and never cross a C boundary, kernel launch wrapper or `noexcept` function. `main` contains them. Google forbids exceptions. Eclair allows them only with that containment |
| Warnings | First-party host targets use `eclair_target_warnings()`: `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion`. New code builds without warnings on GCC 12 and 14 |
| Statics | Namespace-scope objects are `constexpr` or trivially destructible, as Google requires. File-local helpers go in an unnamed namespace |
| Enums | `enum class` with a fixed underlying type. Switch without `default` so `-Wswitch` flags a new enumerator, then handle the impossible value after the switch |

CUDA specifics:

- The caller owns device buffers and the stream. A launch wrapper validates
  pointers and sizes, never allocates or synchronizes, and returns
  `cudaGetLastError()`.
- A probe that may fail on purpose clears the per-thread error with
  `cudaGetLastError()`, so a later launch cannot report it.
  `check_kernel_images` in `src/eclair/cuda/kernel_image.hpp` does this for
  kernel-image probes.
- A function that changes the current device restores it before returning.
- Shared host/device numerical-format math lives in a `formats/` header with functions marked
  `ECLAIR_HOST_DEVICE` (`include/eclair/formats/host_device.hpp`), so the host
  reference and the kernel cannot drift apart. Elementwise conversions reuse
  the kernel and validating launcher in `src/eclair/cuda/convert_kernel.hpp`.
- Shared vector/material/BSDF semantics live in `render/vector.hpp`,
  `render/material.hpp` and `render/bsdf.hpp`; analytic intersections live in
  `render/geometry.hpp`, below operation validation. `render/shading.hpp`, `render/surface_lighting.hpp`,
  `render/preview_shading.hpp` and `render/denoise.hpp` consume ray operation
  contracts; they own shared scalar shading/cache/filter semantics, not allocations,
  policy or presentation. CUDA kernels and portable tests call those helpers.
- Tensor-core linear operations reuse the tile loop in
  `src/eclair/cuda/wmma_tile.hpp`. A new operation supplies a weight policy
  (load raw values, store binary16 into the stage tile, +0 outside the matrix)
  and an epilogue, rather than another copy of the loop.
- Compile only for architectures selected explicitly with
  `ECLAIR_CUDA_ARCHITECTURES`. Volta needs CUDA 12.x and `70`.

## Python and shell tooling

Tool failures need a clear error. Exit
non-zero on every unmet expectation, set timeouts on subprocesses, build argv
lists and quote anything that crosses a shell, keep importable logic apart
from `main()`, and use the standard library first. Shell scripts quote every
expansion, check every status, and send errors to stderr. Never put credentials
or private evidence in logs or Git.

## Source headers

New files start with the Eclair AGPL header, naming the project as
"ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and
Rendering".

## Tests

- Every numerical primitive gets edge-case and representation tests first:
  NaN, infinity, signed zero, subnormals, overflow, rounding and unsupported
  formats. Expected values come from the specification, not from the code
  under test.
- New C++ tests use `tests/check.hpp`, whose `ECLAIR_CHECK` stays active in
  Release builds, unlike `assert`.
- CUDA tests use `tests/cuda_support.hpp` for RAII streams, events and device
  buffers, `checked` and `contain` for failures, the device header line and
  the benchmark timing method. Register them with `eclair_add_cuda_test` in
  `CMakeLists.txt`, which applies C++17 and the selected architectures. cuBLAS
  baselines use `tests/cublas_support.hpp`.
- Capability rules are tested on the host for several architectures. GPU tests
  state the device, driver, toolkit, flags, shapes, precision, synchronization
  and tolerance. A probe-dependent state needs a negative control, such as a
  build without a loadable kernel image.
- A regression test fails for the intended reason when its fix is removed.

```sh
cmake -S . -B build -DECLAIR_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
```

For GPU changes, build and run the matching tests using the
[V100 toolchain](docs/DEBIAN_V100.md). Report the device and toolchain for each
run. For performance changes, follow the [benchmark method](docs/BENCHMARKS.md).

## Review

Review correctness first, then reliability and bounds, then dependency
direction, then readability. Report a finding with its location, trigger,
consequence and evidence. Line counts, file counts and formatter output locate
code to read. They are not findings by themselves. Comments explain invariants,
units, ownership and surprising tradeoffs, not what the next line does.
Remove dead code only after tracing its callers.
