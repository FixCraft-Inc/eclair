#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""V100 checks for the vLLM sm_70 environment and the eclair-ct-sm70 W4A16 path.

Run inside the isolated vLLM venv on an idle V100. Real checkpoint layers go
through the plugin's scheme and vLLM's selected kernel. The reference
dequantizes the compressed-tensors pack-quantized format independently:
eight unsigned 4-bit values per int32, least significant nibble first along
the input dimension, offset 8, times one scale per output row and group.
Products are summed in FP64. The record states shapes, dtypes, stream and
tolerances. It is evidence for these layers and row counts only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

# Thresholds on the relative Frobenius error and cosine similarity against FP64 sums.
REL_FRO_LIMIT = 5e-3
COSINE_LIMIT = 0.9999
ROWS = (1, 3, 64, 257)


def unpack_int4(packed, rows: int, columns: int):
    import torch

    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    return nibbles.reshape(rows, -1)[:, :columns]


def reference_weight(packed, scale, shape):
    """FP64 weight [out, in] from pack-quantized symmetric int4 with group scales."""
    import torch

    rows, columns = (int(v) for v in shape.tolist())
    group = columns // scale.shape[1]
    if group * scale.shape[1] != columns or packed.shape != (rows, math.ceil(columns / 8)):
        raise ValueError("unexpected pack-quantized shapes")
    values = unpack_int4(packed.to(torch.int32), rows, columns).to(torch.float64) - 8.0
    return values * scale.to(torch.float64).repeat_interleave(group, dim=1)


def compare(actual, expected) -> dict:
    import torch

    actual, expected = actual.to(torch.float64), expected.to(torch.float64)
    finite = bool(torch.isfinite(actual).all())
    difference = actual - expected
    norm = float(torch.linalg.vector_norm(expected))
    cosine = float(torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    return {"finite": finite, "rel_fro": float(torch.linalg.vector_norm(difference)) / norm,
            "max_abs": float(difference.abs().max()), "ref_max_abs": float(expected.abs().max()),
            "cosine": cosine}


def layer_check(scheme_type, tensors: dict, prefix: str, generator, stream) -> list[dict]:
    import torch

    packed = tensors[prefix + ".weight_packed"]
    scale = tensors[prefix + ".weight_scale"]
    shape = tensors[prefix + ".weight_shape"]
    out_features, in_features = (int(v) for v in shape.tolist())
    group = in_features // scale.shape[1]
    scheme = scheme_type(strategy="group", num_bits=4, group_size=group, symmetric=True,
                         layer_name=prefix)
    layer = torch.nn.Module()
    # Engine workers allocate parameters on the GPU, so this check does the same.
    with torch.device("cuda"):
        scheme.create_weights(layer, output_size=out_features, input_size=in_features,
                              output_partition_sizes=[out_features],
                              input_size_per_partition=in_features, params_dtype=torch.float16,
                              weight_loader=lambda *a, **k: None)
    with torch.no_grad():
        layer.weight_packed.data.copy_(packed)
        layer.weight_scale.data.copy_(scale.to(torch.float16))
        layer.weight_shape.data.copy_(shape)
    scheme.process_weights_after_loading(layer)
    weight = reference_weight(packed, scale.to(torch.float16), shape)
    results = []
    for rows in ROWS:
        x = torch.randn(rows, in_features, generator=generator, dtype=torch.float32).to(torch.float16)
        expected = x.to(torch.float64) @ weight.T
        with torch.cuda.stream(stream):
            device_x = x.cuda(non_blocking=False)
            torch.cuda.current_stream().synchronize()
            started = time.monotonic()
            actual = scheme.apply_weights(layer, device_x, None)
        stream.synchronize()
        elapsed = time.monotonic() - started
        metrics = compare(actual.cpu(), expected)
        metrics.update(layer=prefix, rows=rows, out_features=out_features, in_features=in_features,
                       group=group, kernel=type(scheme.kernel).__name__,
                       dtype=str(actual.dtype), first_call_seconds=elapsed)
        metrics["passed"] = (metrics["finite"] and metrics["rel_fro"] <= REL_FRO_LIMIT
                             and metrics["cosine"] >= COSINE_LIMIT and actual.dtype == torch.float16
                             and tuple(actual.shape) == (rows, out_features))
        results.append(metrics)
    # Steady-state latency after the correctness calls above have compiled every kernel.
    for rows, repeats in ((1, 200), (8192, 5)):
        x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
        with torch.cuda.stream(stream):
            scheme.apply_weights(layer, x, None)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                scheme.apply_weights(layer, x, None)
            end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1000.0 / repeats
        results.append({"layer": prefix, "timing_rows": rows, "kernel": type(scheme.kernel).__name__,
                        "seconds": seconds,
                        "tflops": 2.0 * rows * in_features * out_features / seconds / 1e12,
                        "weight_gbps": packed.numel() * 4 / seconds / 1e9})
    # Large activations can overflow binary16 even when the FP64 sum is finite.
    big = torch.full((1, in_features), 64.0, dtype=torch.float16)
    expected = big.to(torch.float64) @ weight.T
    with torch.cuda.stream(stream):
        actual = scheme.apply_weights(layer, big.cuda(), None)
    stream.synchronize()
    results.append({"layer": prefix, "probe": "constant 64 activations",
                    "kernel_output_finite": bool(torch.isfinite(actual).all()),
                    "fp64_max_abs": float(expected.abs().max()), "binary16_max": 65504.0})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", action="append",
                        default=["model.language_model.layers.3.self_attn.q_proj",
                                 "model.language_model.layers.3.mlp.down_proj",
                                 "model.language_model.layers.0.linear_attn.in_proj_qkv"])
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    import triton
    from safetensors import safe_open
    import vllm
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization import get_quantization_config

    record: dict = {"schema": "eclair.vllm-v100-checks/1", "passed": False,
                    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "torch_arch_list": torch.cuda.get_arch_list(), "triton": triton.__version__,
                    "vllm": vllm.__version__, "device": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "env_disabled_kernels": os.environ.get("VLLM_DISABLED_KERNELS", ""),
                    "tolerance": {"rel_fro": REL_FRO_LIMIT, "cosine": COSINE_LIMIT},
                    "reference": "FP64 sums of independently dequantized checkpoint weights",
                    "synchronization": "stream.synchronize after each call on a nondefault stream"}
    try:
        import vllm._C_stable_libtorch  # noqa: F401  extension must load on this device
        from vllm.plugins import load_general_plugins

        # Engine processes load the entry point themselves. This check does it explicitly.
        load_general_plugins()
        from eclair_vllm.wna16_sm70 import EclairCompressedTensorsSm70Config, Sm70WNA16

        record["plugin_registered"] = get_quantization_config("eclair-ct-sm70") is EclairCompressedTensorsSm70Config
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        tensors = {}
        for prefix in args.layer:
            for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
                name = prefix + suffix
                with safe_open(args.checkpoint / index[name], framework="pt") as source:
                    tensors[name] = source.get_tensor(name)
        generator = torch.Generator().manual_seed(1234)
        stream = torch.cuda.Stream()
        record["results"] = []
        with set_current_vllm_config(VllmConfig()):
            # Linear layers query the tensor-parallel group, as in a one-GPU engine worker.
            from vllm.distributed import init_distributed_environment, initialize_model_parallel

            torch.cuda.set_device(0)
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                         distributed_init_method=f"file://{args.output}.rendezvous")
            initialize_model_parallel(1, 1)
            for prefix in args.layer:
                record["results"] += layer_check(Sm70WNA16, tensors, prefix, generator, stream)
        checked = [r for r in record["results"] if "passed" in r]
        record["passed"] = bool(record["plugin_registered"] and checked
                                and all(r["passed"] for r in checked))
    except Exception as error:  # the record must capture any setup failure verbatim
        record["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("passed", "error", "vllm", "capability")}))
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
