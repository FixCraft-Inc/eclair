#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Compare W8A16 linear paths at diffusion-transformer shapes, temporaries included.

Paths, all with FP16 activations and FP16 results:

- eclair_fp32_then_half: w8a16_mm FP32 result, then .half(). The ComfyUI-Eclair
  v1 composition (FP32 multiply, add and conversion passes).
- eclair_fp16_epilogue: w8a16_mm(out_dtype=float16), one native epilogue pass.
  Must be bit-identical to the previous path before anything is timed.
- upstream_fp8_cast: F.linear(a, w8.to(float16), fp16 bias). Upstream ComfyUI's
  per-call cast for FP8-stored weights on GPUs without FP8 compute. It applies
  no weight scale, so its values differ. Speed and memory only.
- fp16_resident: F.linear with resident FP16 weights, the upstream default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/pytorch"))
from eclair_torch import load

# (N, K) of the Wan2.2 TI2V-5B transformer's linears: dim 3072, FFN 14336.
LINEAR_SHAPES = ((3072, 3072), (14336, 3072), (3072, 14336))
# Decode, small batch, cross-attention-sized and one CFG-batched 832x480x49
# video request (2 x 5070 tokens under the model's 4x16x16 VAE and 1x2x2 patches).
ROWS = (1, 16, 512, 10140)
SAMPLES = 9
WARMUP_SECONDS = 0.1


def time_path(stream: torch.cuda.Stream, launch: Callable[[], torch.Tensor]) -> dict:
    """CUDA-event median/max of SAMPLES launches and the allocated-tensor peak delta."""
    deadline = time.monotonic() + WARMUP_SECONDS
    while time.monotonic() < deadline:
        launch()
        stream.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    samples = []
    for _ in range(SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        launch()
        stop.record(stream)
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000.0)
    return {"median_us": statistics.median(samples), "max_us": max(samples),
            "tensor_peak_delta_bytes": torch.cuda.max_memory_allocated() - resident}


def same_half(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    nan = torch.isnan(actual)
    return bool(torch.equal(nan, torch.isnan(expected))) and bool(
        torch.equal(actual[~nan].view(torch.int16), expected[~nan].view(torch.int16)))


def run(workspace_bytes: int) -> int:
    stream = torch.cuda.Stream()
    failures = 0
    with torch.inference_mode(), torch.cuda.stream(stream):
        for n, k in LINEAR_SHAPES:
            generator = torch.Generator(device="cuda").manual_seed(n * 31 + k)
            w16 = torch.randn(n, k, device="cuda", generator=generator).mul_(0.02).half()
            # ComfyUI-Eclair's load-time quantization: one FP32 scale per tensor.
            scale = (w16.float().abs().amax() / 448.0).reshape(())
            w8 = (w16.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            bias32 = torch.randn(n, device="cuda", generator=generator).mul_(0.1)
            bias16 = bias32.half()
            for m in ROWS:
                a = torch.randn(m, k, device="cuda", generator=generator).half()
                paths = {
                    "eclair_fp32_then_half": lambda: torch.ops.eclair.w8a16_mm(
                        a, w8, scale, bias32, workspace_bytes).half(),
                    "eclair_fp16_epilogue": lambda: torch.ops.eclair.w8a16_mm(
                        a, w8, scale, bias32, workspace_bytes, out_dtype=torch.float16),
                    "upstream_fp8_cast": lambda: torch.nn.functional.linear(
                        a, w8.to(torch.float16), bias16),
                    "fp16_resident": lambda: torch.nn.functional.linear(a, w16, bias16),
                }
                previous = paths["eclair_fp32_then_half"]()
                fused = paths["eclair_fp16_epilogue"]()
                stream.synchronize()
                identical = same_half(fused, previous)
                if not identical:
                    failures += 1
                del previous, fused
                record = {"m": m, "n": n, "k": k, "workspace_bytes": workspace_bytes,
                          "fp16_epilogue_bit_identical": identical,
                          "fp8_weight_bytes": w8.numel(), "fp16_weight_bytes": 2 * w16.numel()}
                if identical:
                    for name, launch in paths.items():
                        record[name] = time_path(stream, launch)
                print(json.dumps(record), flush=True)
                del a
            del w16, w8, bias32, bias16
            torch.cuda.empty_cache()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=256,
                        help="expanded-weight budget per call (ComfyUI-Eclair default 256)")
    args = parser.parse_args()
    if not 0 <= args.workspace_mib <= 1024:
        parser.error("--workspace-mib must be 0..1024")
    load(args.library)
    print(json.dumps({
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "adapter": torch.ops.eclair.adapter_info(),
        "samples": SAMPLES, "warmup_s_per_path": WARMUP_SECONDS,
        "timer": "CUDA events on one nonblocking stream", "synchronization": "event",
        "precision": "FP16 activations and results; E4M3FN weights with one FP32 scale",
        "includes": "weight expansion, FP32 temporaries, scale, bias and FP16 conversion",
        "excludes": "uploads, model loading, allocator warm-up outside the warmup window",
        "memory_metric": "PyTorch allocated-tensor peak delta; excludes cuBLAS workspace",
        "utilization": "not measured"}), flush=True)
    failures = run(args.workspace_mib * 1024 * 1024)
    if failures:
        print(f"FAIL: {failures} shape(s) where the FP16 epilogue differed from FP32 then half",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
