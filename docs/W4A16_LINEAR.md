# Experimental W4A16

The int4-weight linear primitive has an FP64 reference, CUDA kernels and a
versioned C interface. The [vLLM plugin](../integrations/vllm/README.md)
can select the skinny kernel for captured decode with two to four rows.
The larger-batch kernel follows the same weight rounding contract and uses
a caller-provided weight buffer and cuBLAS workspace.

## Storage and numerical contract

`ops/w4a16_linear.hpp` owns the contract and buffer validation:

- Binary16 activations `[M,K]`, uint32 packed weights `[K/8,N]`, binary16
  scales `[K/128,N]`, binary16 output `[M,N]`.
- Each packed word's low-to-high nibbles correspond to logical input
  offsets **0,2,4,6,1,3,5,7**, with unsigned zero point 8. There is no
  activation permutation, implicit zero-point tensor or bias.
- A dequantized weight is `binary16((nibble - 8) * scale)`. Weight rounding
  happens before multiplication by the activation, and every product is exact
  in FP32. The dot accumulates in FP32 in an unspecified order, then rounds to
  binary16. Accumulation may truncate terms about 23 bits below the largest
  one, as the V100 tensor cores do, and a binary16 subnormal operand may align
  as if its exponent were -14 ([BF16_LINEAR.md](BF16_LINEAR.md#tensor-core-accumulation)).
  Zero signs and NaN payloads are not promised. Subnormals are retained.
- K must be a positive multiple of 128 for nonempty output. Caller-owned
  FP32 scratch `[splits,M,N]` cannot alias inputs or output. Validators check
  extents, alignment, address/size overflow and overlap without reading values.

This is the post-loading layout of pinned vLLM 0.32's Exllama path for the
symmetric group-128 compressed-tensors weights. The serialized
checkpoint instead packs sequential nibbles along K in `[N,K/8]` and has
scales `[N,K/128]`. The packed words and scales were verified against
that representation. The native kernel consumes the Exllama layout directly.
Ignoring alignment, the packed values and FP16 scales take 0.516 bytes per
weight, compared with 2 bytes for an FP16 weight tensor. This storage saving
comes from the layout that both kernels share. The skinny kernel also needs
FP32 split scratch. The large-row route adds a full FP16 expansion and cuBLAS
workspace.

`host/w4a16_linear.cpp` sums independently unpacked, rounded weights in FP64.
`cuda/w4a16_linear.cu` accepts M=1..4 and at most 128 splits, with N/K within
signed-int range and 4-byte-aligned activations. Each warp owns 32 adjacent
output channels as the four quad pairs of PTX `mma.m8n8k4` (Volta `HMMA.884`,
FP32 accumulation), so the `[K/8,N]` pack stays coalesced. A lane dequantizes
its channel's sixteen packed words per scale group directly into tensor-core
A fragments. One `LOP3` places a nibble pair in binary16 1024's mantissa, an
exact `HADD2` or `HFMA2` recovers q - 8, and `HMUL2` by the FP16 scale is the
only rounding. Activations are staged as binary16 in padded shared rows that
feed the B fragments. B columns beyond M duplicate row 0, and their results are
never stored. The next group's activations are read before the current
group's products and stored after them, with one barrier per group. Two FP32
accumulator chains halve the dependent mma sequence. Splits own whole scale
groups. A second kernel reduces the FP32 partials in split order and writes
binary16, avoiding binary16 partial atomics.

`c/w4a16_linear.h` exports ABI 1 through `eclair_w4a16_linear` and its image
probe. The caller owns device buffers and stream until completion. Launches
allocate and synchronize nothing, preserve the current device, and reject
unsupported arguments before enqueuing work. Graph capture is supported.
The vLLM plugin can fall back to the existing framework operation.

## Verification, 2026-10-27

V100/Tesla PG500-216, driver 580.196.07, CUDA 12.9.86, GCC 12.2, explicit
sm_70. CMake Debug, CUDA C++17 with `-g` and no device `-G` or fast math.
Focused host/CUDA tests first passed at source `c42244ea...`; the tensor-core
source `8a0148c6...` passes them plus the last item below. The CUDA test uses
a nondefault stream and synchronizes before downloads:

- All 65536 binary16 scale patterns, all 16 unsigned nibbles and every M=1..4.
- Exact dyadic FP64 comparisons, irregular N=33/131, K=384/6144 and splits
  1/4/16/128, including empty splits and protected buffer sentinels.
- NaN/infinite activations and scales, signed zero, subnormal inputs/results,
  overflow and exact cancellation after intermediate weight rounding.
- Rejected extents/overlaps and graph replay with changed activation values.
- Since the tensor-core partials: `1 + x - 1` with `x = 1.5 * 2^-j`, inside
  one mma step and through the FP32 accumulator across steps, must recover x
  at least through j = 22. A random K=6144 test (M=1..4, splits 1/16)
  compares every output with an independent FP64 dot of FP16-rounded weights
  under the BF16 bound `(K + splits + 4) * 2^-22` times the aligned product
  magnitudes, plus half a binary16 ulp.

Removing weight rounding from a temporary host reference fails the exact
cancellation regression. The prior shared-activation candidate also passed
the complete V100 CTest suite, 30/30; later kernel changes ran focused checks.
Host Release passes 19/19 with GCC 14.2. All ten new C/C++/CUDA files pass clang-format-19. The C header compiles as
C11 with warnings as errors, and the existing vLLM Python suite passes 28/28.

Run the host and CUDA suites from the V100 build with
`ctest --test-dir build-v100 -R w4a16 --output-on-failure`.

## Large-M implementation, 2026-10-28

A second implementation of the same contract serves calls with many rows.
It never changes the numerical contract above: the weights are
`binary16((q - 8) * scale)` and the dot accumulates in FP32.

- `ops::validate_w4a16_dequantize` and `ops::validate_w4a16_gemm` own the
  buffers: a caller-owned `[K,N]` binary16 scratch, the output and a byte
  workspace for the GEMM. The three written buffers must be disjoint from
  each other and from the inputs. `shape.splits` must be 1.
- `cuda::launch_w4a16_dequantize` (in `w4a16_linear.cu`, so it reuses the
  linear kernel's exact LOP3/HADD2/HFMA2 dequantization and single HMUL2
  rounding) writes the weights row-major. One thread covers a packed word row
  and four aligned channels, so a warp stores 256 contiguous bytes per K row.
  Odd N or a misaligned buffer takes a one-channel path.
- `cuda::launch_w4a16_gemm` in the new `eclair_cuda_blas` library runs the
  reconstruction, then `cublasGemmEx` with FP16 operands, `CUBLAS_COMPUTE_32F`,
  `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` and atomics disallowed.
  It sets the stream, then the caller's workspace (setting the stream resets
  it), then the math mode on an Eclair-owned `BlasHandle` for every call.
  PyTorch's process-global reduced-precision flag plays no part.
- Only `eclair_cuda_blas`, its test and `eclair_kernels` link cuBLAS.
  `eclair_cuda` and `eclair-smi` stay free of it. Inside vLLM the C library
  binds the process's already loaded `libcublas.so.12` (the PyTorch 2.13
  wheel's 12.6 copy), as it already does with the CUDA runtime.
- `c/w4a16_gemm.h` ABI 1 exports `eclair_w4a16_gemm_create`/`_destroy` for a
  per-device context, `eclair_w4a16_gemm` and the reconstruction image probe.
  Creation allocates cuBLAS's default pool. Calls allocate and synchronize
  nothing and reject bad arguments before enqueuing work.

The plugin's opt-in `ECLAIR_NATIVE_W4A16_PREFILL=1` first routed every eager
call of 51 rows or more. The current path routes only each shape's
measured row range. Captures, biases,
other layouts and rows outside the range keep Exllama. Each call allocates the `[K,N]` temporary the
installed path also allocates, plus a 32 MiB GEMM workspace, from PyTorch's
allocator on the current stream.

Tests (`tests/cuda_w4a16_gemm_tests.cu`, host `tests/w4a16_linear_tests.cpp`):
all 65536 binary16 scale bit patterns times all 16 nibbles bit-identical to
the specification on both column paths (NaN only as NaN), odd N and more than
65535 word rows, exact dyadic FP64 fixtures for M = 51, 64 and 130 with and
without a workspace, bit-identical repeats, the cancellation regression, an
accumulation regression (every 32nd activation is ±2048, the rest 1/4, exact
result 248), special activations and the random FP64 bound with up to 128
split partials. Replacing `CUBLAS_COMPUTE_32F` with `CUBLAS_COMPUTE_16F` fails
34492 exact-fixture checks and all 816 accumulation checks on an sm_86 GPU
build, while the random bound alone still passes. On the V100 all 31 CTest
tests passed (Tesla PG500-216, driver 580.196.07, CUDA 12.9.86, GCC 12.2).
