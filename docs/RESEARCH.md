# Research questions

This file records questions to answer with primary sources, experiments, and code review. It is not a claim that the answers have already been validated.

## Hardware and toolchains

- Which CUDA 12.x toolkit releases compile and run `sm_70` kernels, and which host compiler combinations are supported?
- Which Volta instructions, Tensor Core operand layouts, shared-memory limits, and asynchronous-copy behaviors are usable through documented CUDA/PTX interfaces?
- What driver, firmware, MIG, persistence, and PCIe conditions affect a V100 PCIe 32 GB deployment?

Primary references to investigate include the NVIDIA Tesla V100 architecture whitepaper, CUDA C++ Programming Guide, CUDA Math API, PTX ISA documentation, and the CUDA compiler release notes.

The current setup recommendation is recorded in [DEBIAN_V100.md](DEBIAN_V100.md). It intentionally favors a conservative CUDA 12.x and proprietary-driver envelope until the workhorse is installed and tested.

## Numerical compatibility

- Which FP8 interchange semantics should Eclair expose: E4M3FN, E5M2, or additional variants?
- How should saturation, NaN payloads, signed zero, subnormals, accumulation type, stochastic rounding, and error tolerances be specified?
- Which BF16 and integer paths can be emulated on Volta without misleading users about native support?

Reference material includes the IEEE floating-point standard, the OCP 8-bit floating-point specification, NVIDIA format documentation, and independent numerical test vectors.

## Runtime and integration

- Which explicit runtime API gives applications seamless dispatch without intercepting unsupported or arbitrary CUDA calls?
- Which PyTorch, vLLM, and other framework extension points are stable enough to support, and how will version drift be tested?
- Which allocations, streams, events, and graph-capture rules must the memory layer preserve?
- What is the smallest capability report that lets a caller understand why a fallback was chosen?

## Originality and provenance

The project may later study public behavior and algorithms, including the Volta FP8 compatibility project named in the kickoff. That work must remain research only until explicitly authorized, and any later analysis must distinguish public ideas from protected source implementation. No source from that project is included in this repository.

## Initial local reference review (2026-10-11)

After the foundation was drafted, the local checkout at `~/vllm-fp8-w8a16-sm70` was reviewed at the documentation, history, and file-layout level. This was an observational review; no files were copied into Eclair and no third-party implementation was treated as an Eclair contract.

The reference confirms several architectural constraints that Eclair should keep explicit:

- V100 has no native FP8 Tensor Core path. Its useful FP8 strategy is FP8-resident storage with conversion and FP16 computation, so memory residency and fused conversion must be measured separately from native FP8 arithmetic.
- The implementation separates a CUDA-core fused dequantization fallback from a Volta WMMA/Tensor Core path, with backend eligibility based on scale layout and matrix alignment. Eclair should model this as dispatch over a common numerical contract rather than as one universal kernel.
- Block scale, channel scale, tensor scale, MoE routing, prefill, and decode have different constraints. A format label alone is insufficient for dispatch.
- The reference documents CUDA 12.x and source builds targeting `sm_70` as deployment constraints. CUDA 13 and newer native FP8 assumptions must remain outside the initial V100 support envelope until independently verified.
- Its benchmark and test artifacts show why Eclair needs per-shape correctness gates, explicit baselines, CUDA-graph state, and memory measurements. Those results belong to that project and are not Eclair benchmark claims.

The reference contains deliberately vendored third-party TurboMind/LMDeploy material with Apache-2.0 provenance. Eclair will not import that source during the kickoff. If a later task authorizes comparative implementation research, record exact versions, licenses, file provenance, and independent design decisions before any dependency is considered.

As an engineering example, the reference is strong in measured kernel experiments, per-format/backend gates, fallback coverage, exactness checks, and benchmark records. Its implementation is not a suitable Eclair base: it is tied to vLLM release internals, Python monkey patches, environment-variable policy, JIT extension loading, and deliberately vendored third-party code. Eclair should reuse the evidence and contract ideas while keeping its C++/CUDA runtime independent.
