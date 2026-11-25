#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Real lm_head weights: native FP16 binding, logits and graph/fallback checks.

The CUDA primitive has its own FP64 tests. This compares its framework path
with PyTorch FP16 linear on seeded synthetic hidden states, not model captures.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def explained_half_tie(native_logits: list[float], fp64_logits: list[float]) -> bool:
    """Both competing channels must have the independently correct FP16 RNE value."""
    if (len(native_logits) != 2 or len(fp64_logits) != 2
            or not all(math.isfinite(x) for x in native_logits + fp64_logits)
            or native_logits[0] != native_logits[1]):
        return False
    try:
        rounded = [struct.unpack("e", struct.pack("e", x))[0] for x in fp64_logits]
    except OverflowError:
        return False
    return native_logits == rounded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    from safetensors import safe_open
    from eclair_vllm import native
    from eclair_vllm.fp16_sm70 import NativeLmHeadMethod

    torch.manual_seed(42)
    record: dict = {"schema": "eclair.fp16-lm-head-checks/1", "passed": False,
                    "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "input": "128 seeded synthetic hidden-state rows, not model captures",
                    "reference": "PyTorch FP16 linear; primitive FP64 tests are separate",
                    "top1_gate": "agreement or a native FP16 tie verified by FP64 dots and independent RNE",
                    "results": [], "top1_disagreements": []}
    try:
        library = native.current_library()
        if library is None or not library.bf16_linear_ready:
            raise RuntimeError("native BF16/FP16 linear image is unavailable")
        record["library_sha256"] = hashlib.sha256(native.library_path().read_bytes()).hexdigest()
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(args.checkpoint / index["lm_head.weight"], framework="pt") as source:
            weight = source.get_tensor("lm_head.weight").half().cuda()
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
        method = NativeLmHeadMethod()
        method.process_weights_after_loading(layer)
        if method._library is None:
            raise RuntimeError("lm_head selected the fallback")
        agreed, rows = 0, 0
        for _ in range(32):
            x = torch.randn(4, weight.shape[1], device="cuda", dtype=torch.float16)
            expected = torch.nn.functional.linear(x, weight)
            actual = method.apply(layer, x)
            rel = float(torch.linalg.vector_norm(actual.float() - expected.float())
                        / torch.linalg.vector_norm(expected.float()))
            same = int((actual.argmax(-1) == expected.argmax(-1)).sum())
            for row in (actual.argmax(-1) != expected.argmax(-1)).nonzero().flatten().tolist():
                left, right = int(expected[row].argmax()), int(actual[row].argmax())
                indices = [left, right]
                exact = x[row].double() @ weight[indices].double().T
                native_logits = actual[row, indices].float().tolist()
                fp64_logits = exact.tolist()
                record["top1_disagreements"].append({
                    "row": rows + row, "channels_baseline_native": indices,
                    "baseline_logits": expected[row, indices].float().tolist(),
                    "native_logits": native_logits, "fp64_logits": fp64_logits,
                    "rounding_explained": explained_half_tie(native_logits, fp64_logits)})
            agreed += same
            rows += 4
            record["results"].append({"rel_fro": rel, "top1_agreed": same,
                                       "finite": bool(torch.isfinite(actual).all())})
        record.update(top1_agreed=agreed, rows=rows)
        # Both an oversized batch and a bias must retain upstream behavior.
        for m, with_bias in ((16, False), (3, True)):
            x = torch.randn(m, weight.shape[1], device="cuda", dtype=torch.float16)
            bias = torch.zeros(weight.shape[0], device="cuda", dtype=torch.float16) if with_bias else None
            actual = method.apply(layer, x, bias)
            expected = torch.nn.functional.linear(x, weight, bias)
            if not torch.equal(actual, expected):
                raise RuntimeError("FP16 fallback differs from PyTorch")
        x = torch.randn(3, weight.shape[1], device="cuda", dtype=torch.float16)
        method.apply(layer, x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = method.apply(layer, x)
        for change in (0.0, 0.125):
            x.add_(change)
            graph.replay()
            expected = torch.nn.functional.linear(x, weight)
            if not torch.allclose(actual.float(), expected.float(), atol=0.01, rtol=0.005):
                raise RuntimeError("FP16 CUDA graph replay differs")
        bad = native._Bf16LinearArgs()
        bad.abi = 99
        if library.eclair_bf16_linear(ctypes.byref(bad), None) != 1:
            raise RuntimeError("C ABI accepted a wrong version")
        record.update(graph_replay=True, fallback=True, wrong_abi_rejected=True)
        record["top1_unexplained"] = sum(not d["rounding_explained"]
                                          for d in record["top1_disagreements"])
        record["passed"] = (record["top1_unexplained"] == 0 and all(
            r["finite"] and r["rel_fro"] < 0.005 for r in record["results"]))
    except Exception as error:  # preserve setup, binding and numerical failures
        record["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("passed", "error", "top1_agreed", "rows")}))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
