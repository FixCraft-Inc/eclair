#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""V100 check of the plugin's Gated DeltaNet prefill against vLLM's own FLA path.

Both run on the same inputs as vLLM's Qwen3.5 prefill produces them: L2-normalized
binary16 q and k, binary16 v, FP32 log-gates and FP32 beta, binary16 recurrent
states for continuing sequences and zeros for new ones, and packed sequences
with cu_seqlens. Outputs and final states are compared, and each path is timed
with CUDA events after a warmup. Shapes match Qwen3.8 27B: 48 value heads, 16
key heads, key and value size 128. Run inside the vLLM venv on an idle V100.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

HEADS, KEY_HEADS, SIZE = 48, 16, 128
TOLERANCE = 2e-2  # relative to the largest reference magnitude


def make_inputs(torch, lengths: list[int], generator):
    tokens = sum(lengths)
    normal = lambda *shape: torch.randn(*shape, device="cuda", generator=generator)
    q = torch.nn.functional.normalize(normal(1, tokens, KEY_HEADS, SIZE), dim=-1).half()
    k = torch.nn.functional.normalize(normal(1, tokens, KEY_HEADS, SIZE), dim=-1).half()
    v = normal(1, tokens, HEADS, SIZE).half()
    g = -(0.01 + 0.29 * torch.rand(1, tokens, HEADS, device="cuda", generator=generator))
    beta = torch.sigmoid(normal(1, tokens, HEADS))
    initial = (0.3 * normal(len(lengths), HEADS, SIZE, SIZE)).half()
    initial[::2] = 0  # Every other sequence starts fresh, as new requests do.
    cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device="cuda",
                              dtype=torch.int32)
    return q, k, v, g, beta, initial, cu_seqlens


def relative(torch, actual, expected) -> float:
    scale = max(1.0, float(expected.float().abs().max()))
    return float((actual.float() - expected.float()).abs().max()) / scale


def timed(torch, call) -> float:
    call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-native-output", action="store_true",
                        help="fail unless the native output symbol and image are available")
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.index import (prepare_chunk_indices,
                                                                   prepare_chunk_offsets)
    from eclair_vllm import gdn_sm70, native

    record: dict = {"schema": "eclair.vllm-gdn-checks/1", "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0), "tolerance_relative": TOLERANCE,
                    "native_library": native.current_library() is not None, "results": []}
    try:
        library = native.current_library()
        record["native_output"] = bool(library and library.gated_delta_output_ready)
        if args.require_native_output and not record["native_output"]:
            raise RuntimeError("native Gated DeltaNet output is required for this check")
        with set_current_vllm_config(VllmConfig()):
            generator = torch.Generator(device="cuda").manual_seed(5)
            for lengths in ([3072], [100, 2000, 64, 7, 901], [30720]):
                q, k, v, g, beta, initial, cu = make_inputs(torch, lengths, generator)
                indices = prepare_chunk_indices(cu, 64)
                offsets = prepare_chunk_offsets(cu, 64)
                common = dict(initial_state=initial, output_final_state=True, cu_seqlens=cu,
                              chunk_indices=indices, chunk_offsets=offsets,
                              use_qk_l2norm_in_kernel=False)
                upstream = lambda: chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta,
                                                          **common)
                eclair = lambda: gdn_sm70.chunk_gated_delta_rule(library, q, k, v, g, beta,
                                                                 **common)
                def state_only():
                    ready = library.gated_delta_output_ready
                    library.gated_delta_output_ready = False
                    try:
                        return eclair()
                    finally:
                        library.gated_delta_output_ready = ready
                expected_o, expected_state = upstream()
                previous_o, _ = state_only()
                actual_o, actual_state = eclair()
                result = {"lengths": lengths,
                          "output_relative": relative(torch, actual_o, expected_o),
                          "output_relative_vs_state_only": relative(torch, actual_o, previous_o),
                          "state_relative": relative(torch, actual_state, expected_state),
                          "finite": bool(torch.isfinite(actual_o).all()
                                         and torch.isfinite(actual_state).all()),
                          "upstream_seconds": timed(torch, upstream),
                          "state_only_seconds": timed(torch, state_only),
                          "eclair_seconds": timed(torch, eclair)}
                result["speedup"] = result["upstream_seconds"] / result["eclair_seconds"]
                result["passed"] = (result["finite"] and result["output_relative"] <= TOLERANCE
                                    and result["output_relative_vs_state_only"] <= TOLERANCE
                                    and result["state_relative"] <= TOLERANCE)
                record["results"].append(result)
                torch.cuda.empty_cache()
        record["passed"] = record["native_library"] and all(r["passed"] for r in record["results"])
    except Exception as error:  # record setup failures verbatim
        record["passed"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=1)[:4000])
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
