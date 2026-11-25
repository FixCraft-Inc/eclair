#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Summarize complete MTP steps in a pinned vLLM 0.32 W4A16 decode trace.

Each target CUDA graph contains W4A16 calls. Its first device event starts a
step, which includes draft graphs and sampling up to the next target graph.
The final incomplete interval is excluded. GPU idle is the complement of the
union of kernels/copies/sets, not the sum of their durations. An idle interval
does not by itself identify a CPU bottleneck. The profiler adds overhead.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

W4A16 = re.compile(r"gemm_half_q_half_gptq_4bit_kernel<(?:true|false), (\d+)>")
NATIVE_W4A16 = re.compile(r"w4a16_partial<\s*(?:\(unsigned int\)\s*)?(\d+)(?:u)?>")
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def w4a16_match(name: str) -> re.Match | None:
    """M is explicit in either pinned partial kernel's template."""
    return W4A16.search(name) or (NATIVE_W4A16.search(name) if "eclair::cuda::" in name else None)


def intervals(events: list[dict], start: float, end: float) -> list[tuple[float, float]]:
    """Merge device activity, clipped to one half-open step interval (microseconds)."""
    merged: list[tuple[float, float]] = []
    for event in sorted(events, key=lambda e: e["ts"]):
        a, b = max(start, event["ts"]), min(end, event["ts"] + event["dur"])
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(b, merged[-1][1])
        else:
            merged.append((a, b))
    return merged


def category(name: str) -> str:
    if w4a16_match(name) or ("eclair::cuda::" in name and "w4a16_reduce" in name):
        return "w4a16"
    if "eclair::cuda::" in name and "decode" in name:
        return "paged_decode"
    if "skinny" in name and "eclair::cuda::" in name:
        return "fp16_linear"
    return "other"


def summarize(events: list[dict]) -> dict:
    device = sorted((e for e in events if e.get("cat") in DEVICE_CATEGORIES
                     and e.get("dur", 0) > 0), key=lambda e: e["ts"])
    weights = [e for e in device if w4a16_match(e["name"])]
    if not weights:
        raise ValueError("trace has no pinned W4A16 kernels")
    targets = {e["args"]["correlation"] for e in weights}
    if any(not e["args"].get("graph id") for e in weights):
        raise ValueError("expected graph-captured target W4A16 calls")
    first: dict[int, float] = {}
    for event in device:
        correlation = event.get("args", {}).get("correlation")
        if correlation in targets:
            first.setdefault(correlation, event["ts"])
    starts = sorted(first.values())
    if len(starts) < 2:
        raise ValueError("trace needs at least two target graph replays")
    rows, calls = [], []
    cursor = 0
    for index, (start, end) in enumerate(zip(starts, starts[1:])):
        while cursor < len(device) and device[cursor]["ts"] + device[cursor]["dur"] <= start:
            cursor += 1
        stop = cursor
        while stop < len(device) and device[stop]["ts"] < end:
            stop += 1
        active = device[cursor:stop]
        merged = intervals(active, start, end)
        busy = sum(b - a for a, b in merged)
        gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
        tail = end - merged[-1][1] if merged else end - start
        totals: dict[str, float] = defaultdict(float)
        for event in active:
            duration = min(end, event["ts"] + event["dur"]) - max(start, event["ts"])
            if event["cat"] == "kernel":
                totals[category(event["name"])] += duration / 1000
            match = w4a16_match(event["name"])
            if match:
                grid = event["args"]["grid"]
                calls.append({"step": index, "m": int(match[1]), "grid": grid,
                              "implementation": "exllama" if W4A16.search(event["name"]) else "eclair",
                              "start_us": event["ts"] - starts[0], "duration_us": event["dur"]})
        rows.append({"step": index, "start_us": start - starts[0],
                     "window_ms": (end - start) / 1000, "busy_ms": busy / 1000,
                     "idle_ms": (end - start - busy) / 1000, "tail_idle_ms": tail / 1000,
                     "largest_idle_ms": max([tail, *gaps], default=0) / 1000,
                     "kernel_ms": dict(totals)})
    shapes = Counter((c["m"], tuple(c["grid"])) for c in calls)
    means = {key: statistics.mean(r[key] for r in rows) for key in (
        "window_ms", "busy_ms", "idle_ms", "tail_idle_ms", "largest_idle_ms")}
    means["kernel_ms"] = {key: statistics.mean(r["kernel_ms"].get(key, 0) for r in rows)
                          for key in sorted({k for r in rows for k in r["kernel_ms"]})}
    return {"schema": "eclair.vllm-decode-steps/1", "complete_steps": len(rows),
            "target_graphs": len(starts), "excluded_final_interval": True,
            "mean_step": means, "steps": rows, "w4a16_calls": calls,
            "w4a16_shapes": [{"m": m, "grid": list(grid), "calls": n,
                                "calls_per_step": n / len(rows)}
                               for (m, grid), n in sorted(shapes.items())],
            "limitations": ["Profiled device gaps do not establish their host cause.",
                            "Kernel sums can overlap; busy time uses the device interval union.",
                            "M comes from the pinned kernel templates, not grid dimensions."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(json.loads(args.trace.read_text())["traceEvents"])
        with args.trace.open("rb") as trace:
            result["trace_sha256"] = hashlib.file_digest(trace, "sha256").hexdigest()
        with args.output.open("x") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"decode_trace: {error}\n")
    print(json.dumps({key: result[key] for key in ("complete_steps", "mean_step", "w4a16_shapes")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
