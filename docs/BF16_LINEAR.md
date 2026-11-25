# BF16 linear operation

The experimental `bf16.gemm` contract computes `output = A * W^T + bias` for
contiguous row-major BF16 A[M,K] and W[N,K], optional FP32 bias[N] and FP32
output[M,N]. The V100 implementation uses CUDA cores for M<=4 and FP16 tensor
cores for larger batches. The V100 has no BF16 arithmetic, and
binary16's range ends at 65504 where BF16's reaches about 3.4e38, so a plain
cast to FP16 overflows. Eclair scales each row by a power of two instead.

## Row scaling

`ops::Bf16RowScalingShape` turns a BF16 matrix into binary16 bits plus one
int32 shift per row:

- The shift puts the row's largest finite magnitude in [2^15, 2^16)
  (`formats::bf16_fp16_shift`). The largest BF16 significand there is 65280,
  below binary16's 65504. A row without a finite nonzero value gets shift 0.
- Each element becomes `formats::bf16_to_fp16_scaled_bits(bits, shift)`:
  binary16 nearest-even of value * 2^shift. A power of two changes only the
  exponent, and BF16 has 8 significant bits to binary16's 11, so every element
  whose exponent is at most 32 below the row's largest stays exact. Smaller
  elements round to binary16 subnormals or zero.
- Infinity and NaN do not affect the shift. Infinity stays infinite, NaN
  becomes the quiet NaN with its sign, and zero keeps its sign.

Weights are row-scaled once, at load. Their binary16 copy is the same size as
the BF16 original. Activations are row-scaled on every call into
caller-provided workspace: M*K binary16 values and M shifts.

## Numerical contract

```text
output[m,n] = fp32(sum_k A'[m,k] * W'[n,k] * 2^-(a_shift[m] + w_shift[n])) + bias[n]
```

A' and W' are the row-scaled binary16 values. Shifts are clamped to +-160
(`formats::clamp_bf16_fp16_shift`), which changes no result: beyond that
bound every finite nonzero BF16 value is already 0 or infinity in binary16.
So any int32 shift is defined.

Each product of two binary16 values is exact in FP32. The GPU accumulates in
FP32 in an unspecified order, then applies the power of
two with `scalbnf`, which is exact unless the result leaves FP32's normal
range. The host oracle sums the same products in FP64, applies the power in
FP64 and rounds once to FP32, overflowing to infinity. The bias is added in
FP32. NaN and infinity propagate as in IEEE arithmetic, and zero signs are not
promised.

When every element lies within its row's exact window, the result equals the
BF16 definition (FP64 sums of the BF16 products rounded once) up to FP32
accumulation. The host test checks that equality bit for bit on rows centered
between 2^-45 and 2^45. The one documented difference is the flush: in
`(2^40, -2^40, 1) . (1, 1, 1)` the 1 rounds away before the cancellation, so
Eclair returns 0 where the exact result is 1. Operands spanning more than 32
binades within one row and cancelling to that depth are the case to watch.

## Tensor-core accumulation

For M>=5, the tensor cores add exact products after aligning them to the largest
exponent and truncating what falls below roughly 23 bits. They do not
normalize binary16 subnormal inputs first, so a subnormal operand aligns as if
its exponent were -14 and narrows the window for the other products in its
group. Measured through the public launcher on both the V100 and an
RTX 3050 Ti:

- With normal operands, `1 + 1.5 * 2^-j` stays exact through j = 22, so the
  window holds bit 23 below the largest product.
- In a constructed case where the largest product is `2^-23 * 2^15` (a
  subnormal times a normal, about 2^-8 but aligned as 2^1), the other product
  is truncated to multiples of 2^-21 in row-scaled units. The relative error
  is 2.79e-5. A random test found this case first.

Row scaling creates binary16 subnormals only from elements more than 29
binades below their row's largest, so this needs a row's large elements to
meet zeros or tiny values in the other operand. The same hardware behavior
applies to any FP16 GEMM on these tensor cores. The GPU tests bound the error
by `(K + 4) * 2^-22 * sum_k |A'W'| + 2^-22 * |reference| + 2^-148` after
rescaling, where each nonzero subnormal operand weighs 2^-14.

## Ownership and selection

- `ops::validate_bf16_row_scaling`, `bf16_linear_sizes` and
  `validate_bf16_linear` own validation: missing, short, misaligned, wrapping
  or overlapping buffers and size overflow. Zero M or N is a no-op. Nonempty
  output needs K > 0. Inputs may alias each other, output may not.
- `host::bf16_row_scaling_reference` and `host::bf16_linear_reference` take
  host storage and allocate nothing.
- `runtime::plan_bf16_linear` applies the FP8 plan's selection rules to
  `bf16.gemm`. Both plans now share one selection function.
- `cuda::launch_bf16_row_scaling` runs one warp per row: a reduction of the
  largest finite magnitude, then the conversion, with 16-byte accesses when
  aligned. For M<=4, `cuda::launch_bf16_linear` streams one output channel per
  warp on CUDA cores and shares each weight across the activation rows. Larger
  batches run the shared tensor-core tile loop (`src/eclair/cuda/wmma_tile.hpp`,
  also used by the FP8 linear). Both use binary16 operands and an epilogue
  that applies both shifts and the bias. The wrappers
  validate, allocate and synchronize nothing, and return `cudaError_t`.
  Success means enqueued.
- `cuda::check_bf16_linear_image` probes every scaling/linear variant. The capability is
  enabled on sm_70 only, like `fp8.gemm`.

## Correctness evidence

Host: the scaled conversion matches an exact double-precision oracle for all
65,536 BF16 patterns at every shift from -170 to 170. The row shift puts
every finite magnitude in [2^15, 2^16) exactly. The linear pipeline matches the
BF16 definition bit for bit, and range, overflow, FP32 subnormal results,
infinity, NaN, extreme shifts and validation are pinned. Breaking the
tie-to-even rule, the shift by one, the epilogue power or the row maximum
made the tests fail.

GPU (`eclair.cuda_bf16_linear`): row scaling is bit-exact with the host for
every BF16 pattern, on vector and scalar paths, misaligned views and more rows
than resident warps. The linear operation is checked against the FP64 oracle
within the bound above for M from 1 to 257, N up to 300 and K up to 1024,
including partial K stages, partial edge tiles, misaligned views, wide
exponent spreads with flushed elements, infinity, NaN, overflow and FP32
subnormal results, with sentinels and rejection checks. An epilogue off by
one power and a missing warp reduction made it fail (29 and 93 failed checks).

On 2026-10-24 Debug and Release CTest passed 21/21 on the V100 (Tesla
PG500-216, sm_70, driver 580.196.07, CUDA 12.9.86, GCC 12.2 in the Debian 12
build root). An RTX 3050 Ti (sm_86,
CUDA 13.4) passed the same suite as a generic check.

## Performance on the V100 (2026-10-24)

`eclair_cuda_bf16_linear_tests --benchmark`, Release, same device and source,
cuBLAS 120902. N = K = 4096, no bias, FP32 output. BF16 inputs with exponents
from -3 to 3, so every baseline stays finite. CUDA events on one nonblocking
stream, a 100 ms warmup per path, nine samples, medians in microseconds.
Correctness was checked before timing: 256 sampled outputs against the FP64
oracle, and the largest difference from SGEMM was below 6e-6 of the largest
output.

| M | Eclair (A scaling + linear) | of which linear | cuBLAS BF16 | cuBLAS FP16 | SGEMM |
| --- | --- | --- | --- | --- | --- |
| 1 | 156.7 | 130.0 | 48.1 | 48.1 | 77.8 |
| 16 | 164.9 | 133.1 | 675.8 | 52.2 | 147.5 |
| 128 | 207.9 | 178.2 | 733.2 | 111.6 | 409.6 |
| 1024 | 1198.1 | 1148.9 | 4017.2 | 471.0 | 3125.2 |
| 4096 | 4601.9 | 4478.0 | 14663.7 | 1808.4 | 11998.2 |

- cuBLAS BF16 is `cublasGemmEx` with `CUDA_R_16BF` inputs, the call a
  framework makes for a BF16 matmul on this card. PyTorch itself was not
  measured. From M=16, Eclair is 3.2-4.1 times faster than it (29.9 versus
  9.4 TFLOPS at M=4096) and 2.6 times faster than SGEMM at M=4096.
- At M=1, cuBLAS BF16 takes a fast GEMV path and Eclair is 3.3 times slower.
  The old tile loop wasted 63 of 64 rows there; the skinny path below replaces it.
- Resident FP16 cuBLAS is 2.5 times faster than Eclair from M=1024, but it is
  a plain cast with no range protection. Eclair's linear runs the same tile
  loop as the FP8 linear and reaches about the same rate (27.6 TFLOPS for FP8
  at M=1024), so tile tuning would help both.
- Row-scaling 32 MiB of weights takes 129-138 us once. Activation scaling is
  28-130 us per call.

Utilization was not profiled. The run was not repeated.

## Skinny FP16/BF16 path, 2026-10-27

The contract already accepts binary16 operands with zero shifts, so resident
FP16 weights need no conversion. M=1..4 now uses one warp per output channel,
16-byte loads when aligned, FP32 FMA and a warp reduction. Other layouts take
scalar loads. M>=5 keeps WMMA. No numerical contract or BF16 row-scaling
rule changes. The tensor-core alignment probe explicitly uses five rows.

V100 focused CTest passed 4/4: host/CUDA BF16 linear
and Gated DeltaNet output. Skinny coverage adds all 65536 binary16 patterns
on both load paths for every M=1..4, zero shifts, sentinels and irregular
shapes, in addition to the existing FP64 tests with nonzero shifts.

`eclair_cuda_skinny_linear_bench` uses resident FP16 input/weights, zero
shifts and an FP16 output pass, against cuBLAS 120902 GemmEx
`COMPUTE_32F/DEFAULT_TENSOR_OP`. Release sm_70, CUDA 12.9.86, GCC 12.2,
driver 580.196.07. Nine CUDA-event samples after 100 ms warmup per path;
sampled FP64 sums agree exactly on the dyadic benchmark inputs. These are
microbenchmarks, not engine speedups.

| M | N / K | Eclair ms | cuBLAS ms | Eclair weight GB/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | 5120 / 6144 | 0.080 | 0.091 | 788 |
| 3 | 5120 / 6144 | 0.079 | 0.118 | 798 |
| 1 | 248320 / 5120 | 2.501 | 3.878 | 1017 |
| 2 | 248320 / 5120 | 2.569 | 3.926 | 990 |
| 3 | 248320 / 5120 | 2.615 | 3.953 | 972 |
| 4 | 248320 / 5120 | 3.152 | 3.994 | 807 |

The C interface `eclair/c/bf16_linear.h` (ABI 1) exposes the existing
operation. The vLLM consumer is opt-in with `ECLAIR_NATIVE_FP16_LINEAR=1`
and applies only to an unquantized `lm_head` in the Eclair compressed-tensors
config. Biases, larger batches and unsupported layouts keep upstream code.
Real RedHatAI weights on 128 seeded synthetic hidden-state rows give relative
Frobenius error about 0.000358 against PyTorch FP16 linear, all finite, top-1
agreement 127/128. In the differing row both native logits round correctly to
the same binary16 value under an FP64 dot check; cuBLAS rounds one down, so
the tie is broken differently. Graph replay, fallback and wrong-ABI checks
pass. The original strict top-1 gate failed and its record is retained.
The revised numerical gate allows only a native tie whose two values match
FP64 dots rounded independently with Python's binary16 RNE conversion; it
still reports raw agreement and rejects unexplained or nonfinite differences.
Reassessment of the original GPU record passes that gate. The framework path requires explicit opt-in.

## Next steps

1. Large batches through cuBLAS. Row-scaled operands are ordinary binary16
   matrices, so an adapter can multiply them with cuBLAS FP16 GEMM (1808 us at
   M=4096) and then apply the shifts and bias in one epilogue pass, as the FP8
   adapter does with its expansion. That needs a scaled epilogue primitive and
   its own evidence. `eclair_cuda` itself stays free of cuBLAS.
2. A BF16 output epilogue, so callers do not need a separate conversion pass.
3. A framework consumer (PyTorch or a ComfyUI loader) with model-level tests.
