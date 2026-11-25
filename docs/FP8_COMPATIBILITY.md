# FP8 compatibility on V100

Eclair now executes FP8-weight linear operations on V100 through supported
CUDA interfaces. A version-pinned native PyTorch adapter also replaces the
previously failing `torch._scaled_mm` call for a tested subset of its contract.
No kernel module, driver modification or false capability is involved. See
[the operation contract](FP8_LINEAR.md) and [PyTorch integration](PYTORCH.md).
This establishes operator compatibility; model loaders and full models need
separate validation.

## Where Eclair should sit

Eclair can be the middleman above the installed NVIDIA stack:

```text
model loader / framework
        |
Eclair adapter: validate checkpoint, select supported operation
        |
Eclair dispatch: real sm_70 capabilities, numerical and workspace policy
        |
FP8 storage -> software decode and scales -> FP16 computation
        |
supported CUDA runtime / user-space driver (libcuda)
        |
installed R580 kernel modules -> V100
```

An adapter must participate before a framework rejects the device or chooses
an incompatible kernel. A wrapper around `libcuda` sits below Python model
configuration checks and cannot catch an operation that never reaches it.
Wrapping an API that receives a typed GEMM request could replace that specific
operation, but does not cover custom kernels, model loading or framework gates.

The closed driver has user-space and kernel components. Eclair can call its
supported interfaces as an external dependency. It cannot embed a kernel
module inside a user-space library: the module still executes in the kernel
and owns the hardware. A container can package selected user-space libraries
but still uses the host kernel driver. Packaging does not change instruction
support or move framework checks. Any redistribution needs separate review of
NVIDIA's license terms and Eclair's AGPL obligations; do not assume NVIDIA
binaries can be absorbed into Eclair's source or relicensed.

The claim that the driver "never sees data types" needs qualification:
user-space CUDA libraries and typed math APIs do know types. Kernel submission
is not a documented FP8 operator interface with shapes, scales and an emulation
callback. A generic launch does not supply enough mathematical meaning to
substitute an FP8 GEMM safely.

Translation is possible in principle. A PTX translator could lower a supported
subset of documented PTX to Volta operations before compilation. That would
need its own instruction, synchronization, memory and numerical contracts;
PTX may be absent when an application ships only architecture-specific cubins.
Rewriting arbitrary compiled programs is a much larger project than supplying
the operations a supported framework actually needs. Neither this review nor
an assembler rejection proves all hypothetical translation impossible.
Eclair's NVIDIA backend preserves the documented [driver boundary](ARCHITECTURE.md#driver-boundary).

## CUDA kernels and a Linux kernel module

A CUDA compute kernel executes GPU work; a Linux kernel module executes host
kernel code. Optimizing Eclair's CUDA kernels can reduce conversion, memory and
matmul costs. A Linux module does not add FP8 instructions, HBM capacity or
model-loader support. No measured host-kernel bottleneck or documented module
hook requiring an Eclair module has been established for the current workload.

A separate module could be justified by a specific supported host/control or
memory-integration requirement. Define the interface, ownership, expected gain,
benchmark and maintenance/security responsibility before implementation. Keep
the NVIDIA driver intact; arbitrary instruction rewriting through its private
interfaces is outside this backend's contract. A userspace plugin or managed
service is the first implementation choice for automatic selection, startup,
configuration and diagnostics. Neither this preference nor the existing
operator tests rule out every possible future kernel-mode backend.

The vLLM engine test is done and needed no
kernel module. Its remaining costs are GPU kernels launched through the
supported driver.

## Implementation gates

1. Define the status, buffer, stream, workspace and fallback contracts, then a
   scaled FP8-weight linear operation with a portable reference. State format,
   scale direction and granularity, layouts, shapes, bias, accumulation and
   output types, rounding, nonfinite behavior and tolerances. W8A16 is the
   first target; W8A8 requires decoding the quantized activations as well.
2. Establish correctness with software decode and an FP16 GEMM baseline.
   Bound the workspace and report its size. Full weight expansion consumes
   extra VRAM; it is a correctness baseline, not the memory-efficient endpoint.
   Specify where scales are applied because rounding or overflow can change
   when scaling moves across a conversion or accumulation.
3. Add a fused Volta implementation that keeps weights in FP8 storage and
   decodes tiles for FP16 math, with a measured choice of GEMV/GEMM by shape.
   Compare accuracy, latency, throughput and memory against named baselines.
   FP8 storage alone does not guarantee a speedup or native FP8 numerical parity.
4. Integrate one pinned framework version and one explicit checkpoint scheme.
   Select Eclair before the framework's native FP8 eligibility check. Verify
   plugin ordering and all worker processes; keep the device identity `sm_70`.
   Prefer supported extension points. If they cannot replace native quantizer
   selection early enough, use an explicit Eclair launcher/adapter or a reviewed
   upstream change, rather than promising installation alone will suffice.
5. Prove the whole path with synthetic tensors first, then a small model. Other scale layouts, MoE, attention, KV cache, graph capture and
   compiled execution need their own compatibility gates.

For PyTorch, a version-tested custom operation is a useful initial boundary.
Replacing `aten::_scaled_mm` is a separate compatibility experiment: existing
CUDA registrations, overloads, return schema, scales, layouts, bias, output
dtype, streams and compiled execution must be checked. `torch.library` does
not imply universal transparent replacement of framework or custom operators.
Checkpoint conversion is another bounded option, subject to architecture
support, conversion semantics and available RAM/VRAM; it cannot always work.

## Native framework failure and replacement, 2026-10-24 UTC

The test used PyTorch 2.7.1+cu126 in an isolated Python 3.13 environment. Its architecture list includes `sm_70`. Native
`torch._scaled_mm` rejected the V100; `TORCH_SHOW_CPP_STACKTRACES=1` identified
`_scaled_mm_out_cuda` at `/pytorch/aten/src/ATen/native/cuda/Blas.cpp:1125`.
The exact exception was:

```text
torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+
```

Loading Eclair's native C++ CUDA registration made the unchanged call succeed.
Six integration tests passed with both zero and 64 MiB weight-workspace
budgets. The loader and launcher perform registration; numerical replacement
runs in C++/CUDA through the supported NVIDIA runtime and driver.
