# Architecture

Eclair's architecture separates stable contracts from device-specific execution. The direction supports a seamless integration experience while respecting the boundary owned by the vendor driver.

## Layers

1. **Public API and integrations** expose explicit runtime selection to applications and adapters for frameworks such as PyTorch or vLLM. An adapter may provide a drop-in integration for a supported call surface; arbitrary CUDA compatibility is not assumed.
2. **Capability and dispatch** identifies devices, driver/runtime versions, supported formats, available kernels, workspace limits, and policy. Dispatch must fail closed when a capability is unknown.
3. **Numerical semantics** defines FP8, BF16, integer formats, conversion, saturation, NaN/infinity behavior, rounding, and tolerances independently of any GPU kernel.
4. **Portable primitives** define contracts for GEMM, attention, conversion, quantization, reductions, and memory movement. A host reference implementation provides an oracle where practical.
5. **Execution backends** implement those contracts for Volta (`sm_70`) and later architectures. CUDA kernels are one backend; future backends must not leak into public types.
6. **Runtime and memory** owns streams, events, allocations, workspaces, synchronization, and lifetime. Ownership and failure behavior must be explicit at every boundary.
7. **Diagnostics and measurement** provide structured errors, capability reports, correctness checks, profiling hooks, and reproducible benchmarks without telemetry or implicit network access.

## Driver boundary

On NVIDIA hardware, the supported NVIDIA kernel driver and CUDA runtime remain responsible for device access, scheduling, memory mappings, and command submission. Eclair can submit its own kernels and provide compatibility or dispatch behavior in user space. It cannot honestly promise to turn an unsupported hardware instruction into native hardware support or to replace the proprietary kernel driver without a separately supported driver stack.

This still permits a seamless product experience: an application selects a documented Eclair runtime or integration, and Eclair routes each operation to an optimized kernel, a portable implementation, or an explicit fallback. The boundary and any performance or numerical tradeoff must remain visible in diagnostics and documentation.

For FP8 models, an adapter must select Eclair before the framework's native
FP8 eligibility check. Merely wrapping the installed driver cannot intercept
checks that reject the model before any driver operation. The recommended
first target is FP8-resident weights with software decode and FP16 computation
on Volta, retaining the real `sm_70` identity. See
[FP8 compatibility](FP8_COMPATIBILITY.md) for the implementation gates,
compiler evidence, and limits of the framework and driver claims.

## Dependency direction

```text
applications / framework adapters
            |
       public runtime
            |
 capability + dispatch ---- diagnostics / benchmarks
            |
 numerical semantics + portable primitives
            |
 CUDA / Volta backend, future hardware backends
            |
 supported vendor runtime and driver
```

Lower layers never depend on the CLI or a framework. The exact dependency
rules live in `CONTRIBUTING.md` and are reflected in CMake targets. Implemented
contracts include `formats/` (FP8, binary16 and bfloat16, see
[FORMATS.md](FORMATS.md)), host and CUDA format conversions, `ops/` shape/buffer validation, the explicit
`runtime/` linear plans, host oracles and CUDA conversion/linear operations,
plus the FP32-accumulator-to-FP16 linear epilogue that adapters use for
binary16 results. The FP8 and BF16 linear operations share one tensor-core
tile loop and differ only in how weights reach binary16 and in their epilogue
([FP8_LINEAR.md](FP8_LINEAR.md), [BF16_LINEAR.md](BF16_LINEAR.md)).
The experimental [skinny W4A16 operation](W4A16_LINEAR.md) has separate
`ops/` validation, an FP64 host reference, a CUDA implementation and C ABI.
It consumes explicitly shuffled group-128 int4 weights, rounds weights to
binary16 and multiplies them on tensor cores with FP32 accumulation. It has no
capability-catalog entry. Its V100 numerical gate passes and it beats
Exllama for M=2..4, but every M=1 shape is still slower.
A large-M implementation of the same contract reconstructs the binary16
weights into caller-owned scratch and finishes in one cuBLAS GEMM whose FP32
compute and reduction precision it sets on its own handle for every call.
It lives in `eclair_cuda_blas`, the only library besides `eclair_kernels`
that links cuBLAS, so `eclair_cuda` and `eclair-smi` stay free of it.
The source vLLM adapter has an explicit experimental M=2..4-only opt-in,
with image/layout/capture checks and Exllama fallback, and a separate eager opt-in for the large-M path
within measured per-shape row ranges. Frameworks own scratch
allocation. A separate draft model subclass can project vocabulary views
without changing the shared target head.
The plan selects an implementation without owning buffers or launching work.
Callers match the report to the current device and provide memory in its domain.
CPU fallback requires a separate CPU plan and host-accessible buffers; device
pointers never silently move to the CPU.

The optional `integrations/pytorch/` adapter owns framework-specific validation,
allocation and startup ordering. Its native operator calls the CUDA primitive
or, for its scalar-scaled matmul contract, uses a bounded weight temporary with
cuBLAS. The generic runtime has no PyTorch dependency. Raw CUDA launch functions
do not read process policy: direct callers enforce policy through the plan.
The adapter independently enforces the same `FeaturePolicy` before execution.

The [vLLM integration](../integrations/vllm/README.md) reaches Eclair kernels
through the versioned `eclair_kernels` C interface. It supplies
[paged attention](PAGED_ATTENTION.md) and
[Gated DeltaNet state and output passes](GATED_DELTA.md) without introducing
a model or engine dependency into the core libraries.

## Rendering consumer

The experimental [rendering preview](REALTIME_RENDERING.md) keeps vector math,
material validation, BSDF sampling/evaluation, intersections and path integration
in separate shared host/device headers, scene/shape validation in
`ops/`, the scalar oracle in `host/`, and tracing/filter kernels in `cuda/`.
The `rt.emulation` capability requires a loadable sm_70 image; RT hardware
facts remain unchanged. Other architectures retain their existing states.

`Material` supplies Lambert diffuse, GGX conductors (perceptual roughness) and
smooth dielectric interfaces (IOR, Fresnel and Snell refraction). Scene validation
rejects rough glass, dielectric sheets and overlapping dielectric bounds. The
path integrator combines one area-light sample and BSDF sampling with MIS.
The optional preview keeps object-space diffuse lighting/cache semantics in
`surface_lighting.hpp`, split-Fresnel cached shading in `preview_shading.hpp`,
and image reconstruction in `denoise.hpp`. Direct lighting updates each frame;
only indirect atlas lighting accumulates. The raw oracle remains independent.
Static specular pixel history requires identity of the entire scene and camera;
primary-surface reprojection alone cannot track a reflected/refracted object.

`cli/eclair-rt/` owns demo scene construction, raster comparison, process
lifecycle and presentation. A CUDA worker owns its context, VRAM, pinned
frame mappings and completion events. A separate CUDA-free process presents
completed RGBA8 frames on Intel using upstream SDL2 KMSDRM. Mode switches
reap the worker before reporting release. Filtering history remains in VRAM;
only final pixels cross PCIe. This is one bounded experimental consumer, not
a general graphics API interception layer. `CONTRIBUTING.md` names its targets.

## Capability reporting

The capability layer (`include/eclair/capability/`) defines Eclair's feature catalog, the report a backend returns, and the enable/disable policy. It names no vendor. Each backend owns the rules for its devices and returns a `BackendReport`:

- The host backend reports the portable references. It runs known-answer checks before reporting a conversion as enabled.
- The CUDA backend queries devices through the runtime API, checks that a loadable kernel image exists for each device, and passes the results to `eclair_cuda_architecture`. That library maps compute capabilities to hardware units and feature states in host-only C++, so the rules for `sm_70`, `sm_80`, `sm_89` and others are tested without a GPU.

A feature is `enabled` only with an implementation and a passing probe.
`not-implemented` has no execution path; `unsupported` records device/build
restrictions; `n/a` records an operation that is not applicable. Backend IDs
(`host`, `cuda`) remain machine identifiers. Friendly labels (`CPU reference`,
`NVIDIA GPU`) and an explicit reference role belong to the backend report.
`eclair-smi` renders them without inventing capability facts.

CUDA probes establish image availability, not correctness or speed. The linear
probe checks both formats, all scale specializations and GEMV/WMMA images.
Known-answer host probes and independent numerical GPU tests supply different
evidence. The generic runtime consumes the feature state and policy and rejects
missing, duplicate, disabled or unrecognized implementations. See
[ECLAIR_SMI.md](ECLAIR_SMI.md) and [FP8_LINEAR.md](FP8_LINEAR.md).

## Repository layout

- `include/eclair/`: public headers and contracts, one directory per layer (`formats/`, `ops/`, `runtime/`, `capability/`, `host/`, `cuda/`, `render/`).
- `cli/eclair-smi/`: the capability report tool, split into options, terminal style, text and JSON rendering, and build-selected backend lists.
- `cli/eclair-rt/`: Cornell/material scenes, CPU comparison, bounded CUDA worker IPC and optional SDL presenter.
- `src/eclair/`: implementation of portable runtime and numerical components.
- `backends/`: reserved for independently implemented future backends; current CUDA code lives in `src/eclair/cuda/`.
- `kernels/`: reserved for additional kernels; current `.cu` files live with the CUDA backend.
- `runtime/`: reserved integration area; current selection code lives in `src/eclair/runtime/`.
- `integrations/`: framework or application adapters.
- `tests/`: host and device correctness tests.
- `benchmarks/`: reproducible benchmark programs and result metadata.
- `docs/`: design, research, roadmap, and measurement contracts.
