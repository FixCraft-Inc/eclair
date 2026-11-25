#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Time candidate prefill attention paths on the V100 at Qwen3.8 27B shapes.

One chunk of queries attends causally (lower-right aligned) to a cached prefix
plus itself: 24 query heads, 4 KV heads, head dimension 256, FP16. Each path is
timed with CUDA events after one warmup and checked against an FP32 reference
on a small slice. This is an operator measurement for deciding where engineering
effort goes, not an end-to-end model result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

HEADS, KV_HEADS, DIM = 24, 4, 256


def flops(chunk: int, prefix: int) -> float:
    # QK^T and PV over the causal region: each query row sees prefix + its own position.
    return 4.0 * DIM * HEADS * chunk * (prefix + (chunk + 1) / 2)


def time_call(torch, function, repeats: int = 3) -> float:
    function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / repeats


def reference(torch, q, k, v, prefix: int, rows: int):
    """FP32 attention for the first `rows` query rows of head 0."""
    group = HEADS // KV_HEADS
    scale = 1.0 / math.sqrt(DIM)
    qs = q[0, 0, :rows].float()
    ks, vs = k[0, 0 // group].float(), v[0, 0 // group].float()
    scores = qs @ ks.T * scale
    positions = torch.arange(ks.shape[0], device=q.device)
    allowed = positions[None, :] <= (prefix + torch.arange(rows, device=q.device))[:, None]
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ vs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--prefix", type=int, action="append")
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.bias import causal_lower_right

    prefixes = args.prefix or [0, 32768, 131072, 196608]
    record: dict = {"schema": "eclair.attention-probe/1", "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0), "shape": {
                        "q_heads": HEADS, "kv_heads": KV_HEADS, "head_dim": DIM,
                        "chunk": args.chunk, "dtype": "float16"},
                    "timing": "CUDA events, one warmup, mean of 3 calls", "results": []}
    generator = torch.Generator(device="cuda").manual_seed(7)
    for prefix in prefixes:
        total = prefix + args.chunk
        q = torch.randn(1, HEADS, args.chunk, DIM, device="cuda", dtype=torch.float16,
                        generator=generator)
        k = torch.randn(1, KV_HEADS, total, DIM, device="cuda", dtype=torch.float16,
                        generator=generator)
        v = torch.randn(1, KV_HEADS, total, DIM, device="cuda", dtype=torch.float16,
                        generator=generator)
        mask = causal_lower_right(args.chunk, total)
        item = {"prefix": prefix, "tflop": flops(args.chunk, prefix) / 1e12}
        group = HEADS // KV_HEADS
        # The fused kernels need equal head counts. Folding the KV-head groups into the
        # batch dimension and broadcasting each KV head with a zero stride copies nothing.
        layouts = {
            "efficient_grouped": (SDPBackend.EFFICIENT_ATTENTION, lambda: torch.nn.functional
                                  .scaled_dot_product_attention(
                                      q.view(KV_HEADS, group, args.chunk, DIM),
                                      k.view(KV_HEADS, 1, total, DIM).expand(KV_HEADS, group, total, DIM),
                                      v.view(KV_HEADS, 1, total, DIM).expand(KV_HEADS, group, total, DIM),
                                      attn_mask=mask).reshape(1, HEADS, args.chunk, DIM)),
            "efficient_repeated": (SDPBackend.EFFICIENT_ATTENTION, lambda: torch.nn.functional
                                   .scaled_dot_product_attention(
                                       q, k.repeat_interleave(group, dim=1),
                                       v.repeat_interleave(group, dim=1), attn_mask=mask)),
            "math": (SDPBackend.MATH, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True)),
        }
        for name, (backend, call) in layouts.items():
            if name == "math" and total > 40000:
                continue  # The math path materializes the full score matrix.
            try:
                with sdpa_kernel(backend):
                    seconds = time_call(torch, call)
                    out = call()
                expected = reference(torch, q, k, v, prefix, 64)
                error = (out[0, 0, :64].float() - expected).abs().max().item()
                item[name] = {"seconds": seconds, "tflops": item["tflop"] / seconds,
                              "max_abs_error_vs_fp32": error}
            except RuntimeError as failure:  # unsupported shape or backend on this device
                item[name] = {"error": str(failure)[:300]}
        record["results"].append(item)
        del q, k, v
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["results"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
