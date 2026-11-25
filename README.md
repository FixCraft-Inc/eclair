# Eclair

Eclair is a CUDA compatibility library for running inference and software
rendering on GPUs such as the NVIDIA Tesla V100 (Volta, `sm_70`). It adds
selected FP8, BF16 and quantized inference operations in user space through
the standard NVIDIA driver and CUDA runtime.

## Try it

Build the CPU reference and run the capability CLI. This needs a C++20
compiler and CMake 3.24 or newer.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
build/eclair-smi --details
```

For a V100, install a compatible NVIDIA driver and CUDA 12.x toolkit,
then select architecture 70. See the [Debian V100 guide](docs/DEBIAN_V100.md)
for the tested toolchain.

```sh
cmake -S . -B build-v100 -DCMAKE_BUILD_TYPE=Release \
  -DECLAIR_ENABLE_CUDA=ON -DECLAIR_CUDA_ARCHITECTURES=70
cmake --build build-v100 --parallel
ctest --test-dir build-v100 --output-on-failure
build-v100/eclair-smi --json
```

`eclair-smi` reports what the build and device support. See [CLI options and
report fields](docs/ECLAIR_SMI.md).

## Supported operations

- [FP8 and BF16 conversion](docs/FORMATS.md), [FP8-weight linear](docs/FP8_LINEAR.md)
  and [BF16 linear](docs/BF16_LINEAR.md). The FP8 linear kernel decodes E4M3FN
  or E5M2 weights for FP16 operands and FP32 accumulation on Volta.
- [Packed int4 linear](docs/W4A16_LINEAR.md), [paged int8 attention](docs/PAGED_ATTENTION.md)
  and [Gated DeltaNet](docs/GATED_DELTA.md), with CUDA kernels and a C interface.
- [PyTorch](docs/PYTORCH.md) supports a checked subset of `torch._scaled_mm`
  with PyTorch 2.7.1+cu126 and also exposes explicit Eclair operators. The
  [scaled-matmul example](examples/pytorch_scaled_mm.py) multiplies two FP8
  tensors and checks the result.
- The [vLLM plugin](integrations/vllm/README.md) targets a specific build of
  vLLM 0.32.0 for Volta. It selects kernels by operation, shape and layout.
- The [software ray-tracing demo](docs/REALTIME_RENDERING.md) can run headless
  or display an interactive preview with SDL2.

## VRAM use and measured results

Fused FP8-weight linear keeps weights in one byte per element and decodes
them inside the kernel. A 4096 x 4096 matrix uses 16 MiB for FP8 weights or
32 MiB for FP16, plus scale storage when applicable. The CUDA benchmark on
a Tesla V100 measured 32.77 µs at one input row with fused FP8 and 48.13 µs
with resident FP16 cuBLAS. At 16 rows it measured 137.25 µs and 63.49 µs,
respectively. [Workload and reproduction command](docs/FP8_LINEAR.md#kernel-revision-2-performance-2026-10-24-utc)

The PyTorch adapter may expand weights to a temporary FP16 matrix for
more than four rows when the matrix fits the configured workspace budget.
With a 1024 x 1024 weight matrix and 16 rows, the fused path took 271.36 µs
and added 98,816 bytes at the PyTorch allocated-tensor peak. The budgeted
path took 34.82 µs and added 2,195,456 bytes. These deltas measure one
operator call, excluding the resident inputs, allocator reserves and
cuBLAS workspace. [PyTorch benchmark and memory policy](docs/PYTORCH.md#execution-and-memory-policy)

The vLLM attention kernel reads the per-token-head int8 KV cache directly.
At head size 256, K, V and their two FP32 scales occupy 520 bytes per token
per KV head, compared with 1,024 bytes for FP16 K and V. This is cache
payload size. End-to-end VRAM also includes weights, activations, scratch
buffers and framework allocations. [Cache layout and supported shapes](docs/PAGED_ATTENTION.md#semantics)

[Benchmark methodology](docs/BENCHMARKS.md) lists the measurement commands and
baselines. See the [architecture](docs/ARCHITECTURE.md) and
[contributor guide](CONTRIBUTING.md) for library boundaries and tests.

## The name

In French, *éclair* means “lightning.” It is also the name of a pastry with a
shell, filling, and glaze. Those layers fit a compatibility library. The
lightning nods to speed and to Nikola Tesla, the namesake of Eclair's first
GPU target, the NVIDIA Tesla V100.

## License

Eclair is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE).
