# Benchmark methodology

Eclair benchmark results are meaningful only when another engineer can reproduce the same workload and distinguish correctness from speed.

Each result must record:

- Eclair revision and build type;
- GPU model, memory size, driver, CUDA toolkit, and `sm_` target;
- host CPU, operating system, and compiler;
- operation, tensor shapes, batch/sequence dimensions, layouts, and alignment;
- input and accumulation formats, output tolerance, and reference implementation;
- warm-up count, measured iterations, synchronization method, streams, and graph-capture state;
- latency distribution, throughput, peak VRAM/workspace, and relevant utilization counters;
- baseline name, version, and build flags.

Correctness must be checked on the same inputs as the timed operation or on a separately identified deterministic vector. Report absolute and relative error, NaN/infinity behavior, and any intentionally relaxed numerical mode. Never hide failed or unsupported cases by silently substituting a different workload.

The host and CUDA unit tests verify correctness. Use the capability probe
to record device properties, and measure throughput in a separate run.

## Format conversion bandwidth

FP8 decode and encode bandwidth on the V100, against a device-to-device copy
of the same bytes, is in [FORMATS.md](FORMATS.md). One unrepeated run, with
input coverage and timing limits stated there.

## BF16 linear

The BF16 linear operation against cuBLAS BF16, FP16 and SGEMM on the V100 is in
[BF16_LINEAR.md](BF16_LINEAR.md). One unrepeated run, kernel timings only, no
model or framework.

## FP8 linear and native adapter measurements

Build a Release configuration for `sm_70` and run
`build-v100/eclair_cuda_fp8_linear_tests --benchmark` to compare fused
FP8-weight linear with resident-FP16 cuBLAS. The recorded V100 results cover
both low-row-count decode and larger matrices. [FP8 linear results](FP8_LINEAR.md)

The native PyTorch adapter can expand weights per call when the workspace
budget allows it. Its benchmark records latency and PyTorch allocated-tensor
peak deltas with and without expansion. The explicit W8A16 benchmark also
compares a per-call FP8 cast and resident FP16 at Wan2.2 linear shapes.
[PyTorch measurements and commands](PYTORCH.md#evidence-on-2026-10-24-utc)

## Experimental W4A16

[W4A16_LINEAR.md](W4A16_LINEAR.md) gives the numerical contract, shape
constraints and per-call workspace of the skinny and large-batch kernels.

## Attention and Gated DeltaNet

The [attention](PAGED_ATTENTION.md) and [Gated DeltaNet](GATED_DELTA.md)
guides give V100 kernel measurements and their reference checks. Keep
isolated timings separate from measurements of a complete model step.
