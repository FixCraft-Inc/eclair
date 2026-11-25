#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Bounded comparison of two Eclair adapter policies, including temporary work."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/pytorch"))
from eclair_torch import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    load(args.library)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "device": torch.cuda.get_device_name(),
                      "capability": torch.cuda.get_device_capability(),
                      "samples": 9, "warmup_ms_per_path": 100,
                      "timer": "CUDA events", "synchronization": "event",
                      "baseline": "Eclair zero-weight-workspace scaled_mm adapter",
                      "precision": "FP8 widening to FP16, FP32 accumulation/output",
                      "includes": "A conversion, W conversion when selected, allocation, scaling",
                      "excludes": "uploads, CPU reference, model loader",
                      "memory_metric": "PyTorch allocated-tensor peak delta; excludes library workspaces",
                      "utilization": "not measured"}), flush=True)
    stream = torch.cuda.Stream()
    original = os.environ.get("ECLAIR_TORCH_WORKSPACE_BYTES")
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            for m in (1, 16, 128):
                a_cpu = (torch.randn(m, 1024) / 4).to(torch.float8_e4m3fn)
                w_cpu = (torch.randn(1024, 1024) / 4).to(torch.float8_e4m3fn)
                expected = (a_cpu.double() @ w_cpu.double().t()).float()
                a, b = a_cpu.cuda(), w_cpu.cuda().t()
                scale = torch.ones((), device="cuda")
                stream.synchronize()
                for budget in (0, 64 * 1024 * 1024):
                    os.environ["ECLAIR_TORCH_WORKSPACE_BYTES"] = str(budget)

                    def launch() -> torch.Tensor:
                        return torch._scaled_mm(a, b, scale, scale, out_dtype=torch.float32)

                    actual = launch()
                    stream.synchronize()
                    torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
                    error = (actual.cpu() - expected).abs().max().item()
                    del actual
                    deadline = time.monotonic() + 0.1
                    while time.monotonic() < deadline:
                        launch()
                        stream.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    resident = torch.cuda.memory_allocated()
                    samples = []
                    for _ in range(9):
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        launch()
                        stop.record(stream)
                        stop.synchronize()
                        samples.append(start.elapsed_time(stop) * 1000)
                    median = statistics.median(samples)
                    print(json.dumps({"m": m, "n": 1024, "k": 1024, "format": "E4M3FN",
                                      "scale": 1.0, "weight_workspace_budget": budget,
                                      "fp8_weight_bytes": w_cpu.numel(),
                                      "median_us": median, "p95_us": max(samples),
                                      "effective_tflops": 2 * m * 1024 * 1024 / (median * 1e6),
                                      "tensor_peak_delta_bytes": torch.cuda.max_memory_allocated() - resident,
                                      "max_absolute_error": error}), flush=True)
    finally:
        if original is None:
            os.environ.pop("ECLAIR_TORCH_WORKSPACE_BYTES", None)
        else:
            os.environ["ECLAIR_TORCH_WORKSPACE_BYTES"] = original


if __name__ == "__main__":
    main()
