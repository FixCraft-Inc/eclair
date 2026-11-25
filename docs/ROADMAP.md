# Roadmap

The current target is the V100. The checked-in contracts cover
[format conversions](FORMATS.md), [FP8](FP8_LINEAR.md), [BF16](BF16_LINEAR.md)
and [int4-weight](W4A16_LINEAR.md) linear operations, [paged attention](PAGED_ATTENTION.md)
and [Gated DeltaNet](GATED_DELTA.md).

## Framework validation

Test each new scale layout, data type and operator shape against the scalar
reference before adding it to a framework adapter. The
[PyTorch adapter](PYTORCH.md) and [vLLM plugin](../integrations/vllm/README.md)
record their current version and dispatch constraints.

Before changing a dispatch threshold, measure correctness and performance on
both sides of the threshold. Include weight expansion, peak workspace and
frame or token timing where those costs apply. Use the
[benchmark methodology](BENCHMARKS.md) to record the comparison.

## Backend expansion

Add new architectures through the capability and dispatch interfaces. Each
backend needs a host-testable capability rule and a device probe for the
feature it implements. See [architecture](ARCHITECTURE.md).
