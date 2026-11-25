# FP8-weight linear operation

The experimental W8A16 contract computes `output = A * W^T + bias` for contiguous
row-major A[M,K] (binary16 bits), W[N,K] (E4M3FN or E5M2 bytes), FP32 scales and
FP32 output[M,N]. FP8 is storage; computation uses V100 FP16/FP32 instructions.
This contract is independent of a framework's scaled-matmul schema.

## Numerical and buffer contract

For each W[n,k], multiply the decoded FP8 value by its FP32 scale, then round to
binary16, nearest even. Scales are row-major
`[ceil(N/block_n), ceil(K/block_k)]` and multiply rather than divide weights.
Tensor, channel and arbitrary block scaling share this rule. A is binary16.
The GPU accumulates in FP32 and adds optional FP32 bias[N]. The portable oracle
sums products in FP64, rounds the sum to FP32, then adds the bias. GPU reduction
order is allowed to differ within the stated test tolerance.

Binary16 conversion has gradual underflow and overflow to infinity. NaNs and
infinities propagate through arithmetic; NaN payload equality and reduction
zero signs are not promised. Moving scales across conversion changes rounding
and overflow and is not a legal optimization of this contract.

`BufferView.size` counts elements, not bytes. Validation checks format, positive
scale blocks, arithmetic overflow, missing/short/misaligned/wrapping buffers and
output overlap with used input ranges. It cannot establish allocation sizes or
memory residency from raw pointers. Inputs may alias each other; output may not
alias an input. Zero M or N is a validated no-op; nonempty output requires K>0.
CUDA nonempty dimensions are limited to INT_MAX before narrowing.

## Ownership and selection

- `ops::fp8_linear_sizes` and `validate_fp8_linear` own common validation.
- `host::fp8_linear_reference` accepts host-accessible storage and allocates nothing.
- `runtime::plan_fp8_linear` consumes a DeviceReport and FeaturePolicy. Unknown,
  missing, duplicate, disabled or unimplemented capabilities fail closed. An
  explicit CPU report selects the CPU reference; it never consumes GPU pointers.
- `cuda::launch_fp8_linear` accepts device buffers and a caller-owned stream,
  allocates/copies/synchronizes nothing, and returns `cudaError_t`. Success means
  enqueued; the caller checks completion. Validation failure enqueues nothing.
  The caller matches the report and current device and enforces policy before
  this low-level call. Buffers remain alive until completion.

M<=4 uses CUDA cores. One warp owns an output channel, decodes each weight once
and applies it to every row with FP32 FMA, then reduces across the warp. Lanes
read 16 weight bytes at a time when K is a multiple of 16 and the pointers are
aligned, otherwise 4 bytes or single bytes. Larger M uses 128-thread blocks that
compute 64x64 output tiles in 32-deep K stages. Each warp owns 32x32 outputs as
2x2 FP16 WMMA fragments with FP32 accumulation, and the next stage's global
loads are issued before the current stage's MMAs. Edge padding is zero and is
never multiplied by a scale, so a NaN or infinite scale reaches only the outputs
whose dot products use it.

Both kernels decode two FP8 values at a time with 32-bit and half2 bit
operations instead of the per-value reference decode. A unit tensor or channel
scale skips the multiply and rounding. That is exact because every FP8 value is
a binary16 value. GPU tests compare all 256 patterns of both formats through
both kernels and every vector-load position, with unit and non-unit scales,
against the host oracle. The weight format and tensor/channel/block scale layout
specialize the kernels. Workspace is zero, and weights stay in FP8 storage.
Image probes check both formats and all six format/scale combinations for both
kernel families, restore the prior device and clear expected lookup errors.
Capability is enabled on sm_70 only.

`ops::LinearEpilogueShape` and `cuda::launch_linear_epilogue` define a separate
FP16 output stage for FP32 accumulators: multiply by one FP32 scale, add an
optional FP32 bias, each rounded to FP32 without a fused multiply-add, then
round to binary16 nearest-even. It replaces three full passes (multiply, add,
conversion) with one and is bit-identical to them. `host::linear_epilogue_reference`
is its oracle. The same ownership rules apply: caller-owned buffers and stream,
validation before enqueue, no allocation or synchronization.

The PyTorch adapter has a separate scalar-scaled W8A8 contract with scales
applied *after* accumulation. It can expand one weight matrix into a bounded
FP16 temporary and call cuBLAS for larger batches; see [PYTORCH.md](PYTORCH.md).
That optional temporary is not hidden inside this low-level API.

## Correctness evidence

Host tests exhaust all 65,536 binary16 representations (NaNs checked by class),
all 256 patterns of both FP8 formats, conversion ties, underflow, overflow,
signed zero, infinity and NaN. Linear tests cover known matrices, scales, bias,
invalid extents, aliasing, alignment and size/address overflow. Runtime tests
cover explicit fallback, policy, unavailable images and dimension narrowing.

V100 numerical tests use nonblocking streams and stream synchronization before
readback; compare FP32 output with the FP64 host oracle and cuBLAS GemmEx using
resident rounded FP16 weights. Finite tolerance is
`abs(error) <= 2e-4 + 2e-4 * abs(reference)`, with explicit NaN/infinity checks.
Both FP8 formats cover:

| M | N | K | Scale blocks |
| --- | --- | --- | --- |
| 17 | 19 | 33 | 7 x 13 |
| 1 | 257 | 129 | 128 x 128 |
| 4 | 19 | 33 | 7 x 13 |
| 32 | 48 | 64 | channel |
| 16 | 32 | 48 | tensor |

Additional cases exercise all representations, scalar exceptional values,
output guards and rejection without output changes.

Kernel revision 2 adds, for both formats: the CUDA-core path with 16-byte,
4-byte and misaligned loads (M 1-4, K 20-1024), block scales shared per
four-column group and per column, tensor-core tiles from M=5 to M=257 with
several K stages, partial last stages and partial edge tiles, misaligned A and
W views, every byte pattern at every vector-load position with unit, non-unit
and mixed channel scales, and a NaN block scale that must poison exactly the
outputs whose dot products use it. Removing the E4M3FN NaN fix-up or swapping
the decode byte order makes the suite fail (10 and 60 failed checks).

The epilogue's host test pins its contract to binary16 bit patterns derived
from the specification: ties to even, gradual underflow, overflow at 65520,
signed zero with and without bias, NaN and infinity. One vector,
`0x1.000002p-3 * 0x1.000004p+0 - 0x1.000002p-3`, rounds to 0x0000 with two FP32
roundings and to 0x0001 with a fused multiply-add. The GPU test requires bit
equality with the host oracle on random and exceptional values, the vector and
scalar paths, more than 65,535 rows, misaligned views and rejection without
writes.

## Kernel revision 2 performance (2026-10-24 UTC)

Tesla PG500-216 V100, 32,768 MiB, `sm_70`, driver 580.196.07. The Release
build used CUDA 12.9.86, GCC 12.2 and cuBLAS 12.9.2. The benchmark used
E4M3FN weights, tensor scale 1, FP16 activation and FP32 output with
deterministic dyadic inputs and no bias. It checks exact agreement with cuBLAS
before timing. Each path runs on one nonblocking stream with CUDA graph capture
off, 100 ms of warmup and nine synchronized CUDA-event samples. P95 is the
largest of nine samples. The fused timing includes FP8 decoding and scaling.
The baseline measures computation with FP16 weights resident on the GPU.

Build a Release configuration for `sm_70` and run
`build-v100/eclair_cuda_fp8_linear_tests --benchmark`.

| M | N = K | Fused median / p95 (µs) | Resident FP16 cuBLAS median (µs) |
| --- | --- | --- | --- |
| 1 | 1024 | 8.160 / 8.192 | 11.264 |
| 16 | 1024 | 36.864 / 37.888 | 12.288 |
| 128 | 1024 | 39.936 / 39.936 | 19.456 |
| 1 | 4096 | 32.768 / 33.792 | 48.128 |
| 4 | 4096 | 58.368 / 61.440 | 63.488 |
| 16 | 4096 | 137.248 / 138.240 | 63.488 |
| 128 | 4096 | 210.944 / 212.992 | 111.616 |
| 1024 | 4096 | 1251.328 / 1252.352 | 444.416 |

At the measured one- and four-row shapes, fused decode is faster than resident
FP16 cuBLAS and reads half the weight bytes. The fused kernel takes more time
at the larger measured row counts. At M=1024 and N=K=4096, the effective rates
are 27.5 TFLOP/s for Eclair and 77.3 TFLOP/s for cuBLAS. This run did not
measure device utilization or full-model memory.
