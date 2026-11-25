# Native PyTorch adapter

The opt-in adapter implements a checked subset of `aten::_scaled_mm` on the
V100 using native C++/CUDA. An explicit loader installs the CUDA registration
before a script runs. Python only performs startup and library loading; it does
not replace arithmetic functions or edit the application.

This is an experimental, process-wide override of an internal PyTorch operator,
pinned to **torch 2.7.1+cu126**, Linux x86_64. It is not a universal model plugin.
PyTorch prints its normal operator-override warning on load. Restart without
loading Eclair to restore native behavior. Other CUDA devices fail closed in
this process. The adapter uses the device's reported compute capability 7.0.

## Supported call surface

A[M,K] and B[K,N] are E4M3FN or E5M2 (including mixed formats). A is contiguous;
B is the transpose of contiguous weights[N,K]. Both scales are contiguous FP32
CUDA tensors with one element, on the same device. Matrices are nonempty and
dimensions fit signed int. The operation widens FP8 exactly to FP16, accumulates
in FP32, then multiplies by scale_a and scale_b in order. Optional bias[N] is
FP32 or the output dtype. An explicit output dtype is required: FP32, FP16 or
BF16. FP16/BF16 output uses PyTorch's standard conversion.

Training, deterministic-algorithm mode, CUDA graph capture, `scale_result`,
implicit output dtype and `use_fast_accum=True` are rejected. Row/block-scaled
`_scaled_mm`, its `out` overload, torch.compile/export, quantizer gates, custom
framework kernels and subprocess startup are not covered. A model may reject
the device before calling this operator; that requires its own adapter before
the loader's eligibility check.

The separate native `torch.ops.eclair.fp8_linear(A, weights, scales, block_n,
block_k, bias=None)` accepts FP16 A, FP8 weights and FP32 block scales/bias,
and returns FP32. It follows [FP8_LINEAR.md](FP8_LINEAR.md), where scales round
into FP16 weights before multiplication. These two numerical contracts must
remain distinct.

## Execution and memory policy

M<=4 uses the zero-weight-workspace Eclair kernel. Larger batches use an FP16
copy of the current weight matrix and cuBLAS if it fits
`ECLAIR_TORCH_WORKSPACE_BYTES` (default 67108864, 64 MiB); otherwise they use the
fused kernel. Zero forces the fused path. The byte count must be a nonnegative
integer; invalid configuration fails. No allocation failure is silently retried.

This budget covers **expanded weights only**. It excludes the FP16 activation
temporary, FP32 result, PyTorch allocator caching and cuBLAS's own workspace.
Persistent checkpoint weights stay FP8. Expansion repeats for each operation;
there is no persistent FP16 weight cache. The allocator and current CUDA stream
own temporary lifetime. The adapter changes neither stream nor cuBLAS handle
configuration. Calls are asynchronous; callers check CUDA completion normally.
`ECLAIR_DISABLE` is validated and enforced before work.

## Explicit installation and build

Use an isolated Python environment. The tested interpreter was Python
3.13.5. Install the pinned PyTorch package explicitly.

```sh
python -m pip --isolated install --no-input --no-cache-dir --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1+cu126
```

The wheel includes sm_70. Newer wheel/toolkit combinations may omit it; the
loader rejects other torch versions. NVIDIA CUDA packages remain separately
licensed dependencies. PyTorch is a BSD-licensed external dependency.

After building Release Eclair archives for `sm_70` in `build-v100`:

```sh
python integrations/pytorch/build.py --eclair-build build-v100 \
  --output "$PWD/build-v100/eclair_torch.so"
```

The standalone adapter compiler uses installed native g++14.2, C++20, `-O2`,
PIC/shared, torch's ABI1 and includes/libraries. Kernels remain the isolated
GCC 12.2/CUDA 12.9 Release archives. `build.py` never overwrites an existing output.
Use a new output path for each reviewed rebuild.

Run an unchanged example:

```sh
PYTHONPATH="$PWD/integrations/pytorch" python \
  -m eclair_torch --library "$PWD/build-v100/eclair_torch.so" \
  examples/pytorch_scaled_mm.py
```

Applications may instead call `eclair_torch.load(path)` explicitly at startup.
Install/load in every process that needs it; no global system-Python changes
are made. The contract 2 artifact recorded below is `eclair_torch_v3.so` (v2 was
used for its benchmark). The contract 3 artifact is `eclair_torch_v5.so`.

## Evidence on 2026-10-24 UTC

Tesla PG500-216 sm_70, R580.196.07. PyTorch 2.7.1+cu126, bundled CUDA 12.6 runtime /
cuBLAS 12.6.4.1; Eclair kernels built with CUDA 12.9.86. Native scaled_mm failed:

```text
torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+
Exception raised from _scaled_mm_out_cuda at /pytorch/aten/src/ATen/native/cuda/Blas.cpp:1125
```

Eclair made the same call succeed without application edits. Six tests passed
with both 0 and 64 MiB budgets: block-scaled native linear; three format pairs;
M1,4,17,32 with irregular dimensions; scalar scales 0.7/1.3 and bias; all 256 FP8
patterns including NaNs/infinities/signed zero/subnormals; nondefault stream;
FP32/FP16/BF16 outputs; invalid arguments, policy, invalid workspace configuration and deterministic-mode rejection. CPU reference uses FP64
products/sum rounded to FP32 before post-scales/bias. FP32 atol/rtol2e-4,
FP16 2e-3, BF16 2e-2; nonfinite comparisons are explicit. Stream synchronization
precedes readback.

```sh
python tests/pytorch_adapter_tests.py --library /absolute/path/to/adapter.so
ECLAIR_TORCH_WORKSPACE_BYTES=0 python tests/pytorch_adapter_tests.py \
  --library /absolute/path/to/adapter.so
python benchmarks/pytorch_scaled_mm.py --library /absolute/path/to/adapter.so
```

The test first requires the native failure, then loads the native registration.
It is opt-in rather than a default host CTest dependency. NumPy is absent in the
isolated environment; torch emits a NumPy initialization warning, but these
tests use no NumPy. No model was downloaded.

The bounded adapter benchmark used N=K=1024, E4M3FN, scalar scales 1, FP32 output,
seed 42, graph capture off, one nonblocking stream, 100 ms warmup per path and nine
CUDA-event samples with event synchronization. CPU reference matched exactly.
All conversion, allocation and post-scale work is included; uploads and CPU
reference are excluded. The comparison is between Eclair adapter policies,
not against a successful native FP8 kernel (none exists on V100).

| M | Fused median / p95 (us) | 64 MiB budget median / p95 (us) | Allocated tensor peak delta, fused / budgeted (bytes) |
| --- | --- | --- | --- |
| 1 | 32.768 /35.840 | 32.768 /35.840 | 6656 /6656 |
| 16 | 271.360 /273.408 | 34.816 /35.840 | 98816 /2195456 |
| 128 | 284.672 /285.728 | 43.008 /44.032 | 786944 /2883584 |

With the 64 MiB budget, the effective throughput was 0.964 TFLOP/s at
M=16 and 6.242 TFLOP/s at M=128. Peak deltas count live PyTorch tensor
allocations above the resident inputs. Allocator reserves and cuBLAS
workspaces are outside that metric. P95 is the maximum of nine samples.
The original registration's benchmark ran on an earlier development revision.

## Operators-only application integration

`build.py --operators-only` defines `ECLAIR_TORCH_ATEN_OVERRIDE=0`, retaining
the custom Eclair operators while omitting the process-wide aten registration.
`torch.ops.eclair.adapter_info()` reports `eclair.pytorch/3;operators-only`,
and the default build reports `eclair.pytorch/3;aten-override`. Contract 3 adds
`out_dtype` to `w8a16_mm`. Use a fresh process when changing adapter variants. Loading both variants
in one process registers the same Eclair operators twice.

`torch.ops.eclair.w8a16_mm(A, weights, scale, bias=None,
workspace_bytes=67108864, out_dtype=None)` takes contiguous CUDA FP16 A[M,K],
E4M3FN/E5M2 weights[N,K], one contiguous FP32 CUDA scale and optional FP32
bias[N], all on the same sm_70 device. It returns FP32 when `out_dtype` is None
or `torch.float32`. With `torch.float16` it returns that FP32 value rounded once
to binary16, bit for bit, through Eclair's one-pass linear epilogue instead of
separate multiply, add and conversion passes. Other dtypes are rejected.
Weights widen exactly to FP16;
accumulation is FP32, followed by scalar multiplication and bias. This differs
from the block-scaled operator's weight rounding contract. It shares inference,
dimension, policy, graph-capture and deterministic-mode restrictions with the
existing adapter. Nonfinite values follow the existing IEEE/reference behavior.

M>4 uses expanded-weight cuBLAS when the FP16 weight matrix fits the nonnegative
per-call budget; other calls use the existing fused kernel with unit scale,
then apply the post-scale. The per-call budget does not read or modify the
script launcher's ECLAIR_TORCH_WORKSPACE_BYTES setting. It excludes activations,
FP32 output, allocator caches and library workspace. No persistent weight cache
or OOM retry is introduced. FP16 output still materializes the FP32 accumulators
for the epilogue to read, so its peak temporary memory equals the FP32 path
followed by a conversion.

`tests/pytorch_w8a16_tests.py` checks both formats, irregular shapes, independent
FP64 references, all byte representations, nondefault streams, zero/64 MiB
budgets, invalid inputs/policy and the negative control that native `_scaled_mm`
still rejects sm_70 in the isolated build. Its FP16-output tests require bit
equality with the FP32 result converted by `.half()` across both formats, both
kernel paths, both budgets, with and without bias, an FP16-overflowing scale,
NaN and infinite activations, bias and scale values, plus an independent FP64
reference.

### Contract 3 evidence (2026-10-24 UTC)

Tesla PG500-216, R580.196.07, torch 2.7.1+cu126, Eclair Release archives
(CUDA 12.9.86, GCC 12.2), adapter compiled with g++ 14.2. The W8A16 suite passed all six tests. The scaled-matmul suite passed all
six tests with both 64 MiB and zero weight workspace.

`benchmarks/pytorch_w8a16.py` compares four paths at the Wan2.2 TI2V-5B linear
shapes, with FP16 activations and results, E4M3FN weights with one FP32 scale,
the 256 MiB expansion budget, CUDA events on one nonblocking stream, 100 ms
warmup and nine samples. The FP16 epilogue was bit-identical to FP32 followed by
`.half()` at all twelve shapes before timing. Medians in microseconds:

| M | N x K | Eclair FP32 then half | Eclair FP16 epilogue | Upstream FP8 cast | FP16 resident |
| --- | --- | --- | --- | --- | --- |
| 1 | 3072 x 3072 | 37.9 | 31.7 | 120.8 | 33.8 |
| 1 | 14336 x 3072 | 92.2 | 80.9 | 572.4 | 196.6 |
| 1 | 3072 x 14336 | 105.5 | 97.3 | 505.8 | 116.7 |
| 16 | 14336 x 3072 | 589.8 | 581.6 | 575.5 | 200.7 |
| 512 | 14336 x 3072 | 1205.2 | 1075.2 | 1025.0 | 662.5 |
| 10140 | 3072 x 3072 | 3277.8 | 2812.9 | 2615.3 | 2594.8 |
| 10140 | 14336 x 3072 | 15086.6 | 12807.2 | 11975.7 | 11903.0 |
| 10140 | 3072 x 14336 | 12582.9 | 12006.4 | 11735.0 | 11694.1 |

"Upstream FP8 cast" is `F.linear(a, w8.to(float16), fp16 bias)`, the per-call
cast ComfyUI 0.42.1 uses for FP8-stored weights on GPUs without FP8 compute. It
applies no weight scale, so its values differ. It is a speed and memory
baseline only. Peak allocated-tensor deltas at M=10140, 14336 x 3072 were 833,
833, 363 and 279 MiB respectively: the Eclair paths hold an FP32 accumulator
matrix that the others do not.

The FP16 epilogue reduced tested Eclair call times by 1.4–16%. At one row,
fused Eclair took 1.07–2.4 times less time than resident FP16 and 3.8–7.1 times
less than the per-call FP8 cast. At the larger tested row counts the benchmark
budget allowed weight expansion, and the FP32 accumulator increased Eclair's
temporary memory. These timings cover individual calls. Reproduce them with
`python benchmarks/pytorch_w8a16.py --library build-v100/eclair_torch.so`.
