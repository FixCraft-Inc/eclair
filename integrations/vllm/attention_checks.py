#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""V100 check of the Eclair sm_70 prefill and decode paths against vLLM's own Triton attention.

Both implementations read one int8 per-token-head paged cache that vLLM's own
update kernel wrote from the same FP16 keys and values. Each output is compared
with an FP32 attention over the dequantized cache contents, which both paths
are meant to compute, and with the other path. Latencies use CUDA events with a
warmup. Shapes match Qwen3.8 27B: 24 query heads, 4 KV heads, head size 256.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

HEADS, KV_HEADS, DIM = 24, 4, 256
TOLERANCE = 2e-2  # maximum absolute error for outputs of magnitude about 1


def fp32_reference(torch, q, k, v, context: int, rows: int, head: int = 0):
    """FP32 attention of the first `rows` queries of one head over the dequantized K/V it uses."""
    kv = head // (HEADS // KV_HEADS)
    scores = (q[:rows, head].float() @ k[:, kv].float().T) / math.sqrt(DIM)
    positions = torch.arange(k.shape[0], device=q.device)
    allowed = positions[None, :] <= (context + torch.arange(rows, device=q.device))[:, None]
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v[:, kv].float()


def metadata_for(torch, metadata_type, context: int, rows: int, block_table, slots, segments: int):
    """vLLM's builder settings for one sequence: up to 128 // KV heads single-token sequences
    use its segmented 3-D kernel with `segments` softmax segments in FP32 scratch buffers."""
    threshold = 128 // KV_HEADS
    return metadata_type(
        num_actual_tokens=rows, max_query_len=rows,
        query_start_loc=torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        max_seq_len=context + rows,
        seq_lens=torch.tensor([context + rows], device="cuda", dtype=torch.int32),
        block_table=block_table, slot_mapping=slots[context:], seq_threshold_3D=threshold,
        num_par_softmax_segments=segments,
        softmax_segm_output=torch.empty(threshold, HEADS, segments, DIM, device="cuda"),
        softmax_segm_max=torch.empty(threshold, HEADS, segments, device="cuda"),
        softmax_segm_expsum=torch.empty(threshold, HEADS, segments, device="cuda"),
        causal=True)


def timed(torch, call) -> float:
    """Seconds of one call after one warmup, by CUDA events."""
    call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def run_case(torch, modules: dict, context: int, rows: int, generator) -> dict:
    triton_impl, eclair_impl, metadata_type, dequantize = (
        modules["triton"], modules["eclair"], modules["metadata"], modules["dequantize"])
    total = context + rows
    block = modules["block"]
    blocks = (total + block - 1) // block
    kv_cache = torch.zeros(blocks, KV_HEADS, block, 2 * (DIM + 4), dtype=torch.int8, device="cuda")
    key = torch.randn(total, KV_HEADS, DIM, device="cuda", dtype=torch.float16, generator=generator)
    value = torch.randn(total, KV_HEADS, DIM, device="cuda", dtype=torch.float16, generator=generator)
    query = torch.randn(rows, HEADS, DIM, device="cuda", dtype=torch.float16, generator=generator)
    block_table = torch.arange(blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
    slots = torch.arange(total, device="cuda", dtype=torch.int64)
    for impl in (triton_impl, eclair_impl):
        impl._k_scale_cache = impl._v_scale_cache = None
    # vLLM's view creation resets every scale to 1. Its update call creates its views
    # before writing. Eclair's views write nothing and are created here.
    eclair_impl._ensure_scale_caches(kv_cache)
    triton_impl.do_kv_cache_update(None, key, value, kv_cache, slots)
    # Prefill chunks run with the shared workspace an engine worker allocates.
    eclair_impl._workspace = tuple(torch.empty(max(modules["window"], rows), KV_HEADS, DIM,
                                               dtype=torch.float16, device="cuda") for _ in range(2))
    outputs, seconds = {}, {}
    variants = [("triton", triton_impl, None), ("eclair", eclair_impl, None)]
    library = modules["native"]
    # Chunks of 16 or more rows run the native prefill, shorter ones the native decode.
    if library is not None and (rows >= 16 or modules["native_module"].decode_supported(
            library, HEADS, KV_HEADS, DIM, rows)):
        variants.append(("eclair_native", eclair_impl, library))
    for name, impl, native in variants:
        eclair_impl._native = eclair_impl._native_decode = native
        metadata = metadata_for(torch, metadata_type, context, rows, block_table, slots, 16)
        # vLLM's attention layer hands implementations a [tokens, heads, head_size] buffer.
        out = torch.empty(rows, HEADS, DIM, device="cuda", dtype=torch.float16)
        seconds[name] = timed(torch, lambda: impl.forward(None, query, key[context:], value[context:],
                                                         kv_cache, metadata, out))
        outputs[name] = out
    key_cache, value_cache = eclair_impl._pth_key_value_caches(kv_cache)
    k = dequantize(key_cache, eclair_impl._k_scale_cache, block_table[0], 0, total, DIM)
    v = dequantize(value_cache, eclair_impl._v_scale_cache, block_table[0], 0, total, DIM)
    probe_rows = min(rows, 64)
    expected = fp32_reference(torch, query, k, v, context, probe_rows)
    result = {"context": context, "rows": rows, "seconds": seconds,
              "speedup": seconds["triton"] / min(s for n, s in seconds.items() if n != "triton"),
              "tflops": {n: 4.0 * DIM * HEADS * rows * (context + (rows + 1) / 2) / s / 1e12
                         for n, s in seconds.items()}}
    for name, out in outputs.items():
        result[f"{name}_max_abs_vs_fp32"] = (out[:probe_rows, 0].float() - expected).abs().max().item()
        result[f"{name}_finite"] = bool(torch.isfinite(out).all())
        result[f"{name}_max_abs_vs_triton"] = (outputs["triton"].float() - out.float()).abs().max().item()
        if rows < 16:
            # Decode and verification rows are cheap to check on every head and KV head.
            result[f"{name}_all_heads_max_abs_vs_fp32"] = max(
                (out[:, h].float() - fp32_reference(torch, query, k, v, context, rows, h)).abs().max().item()
                for h in range(HEADS))
    result["passed"] = all(result[f"{name}_finite"] and result[f"{name}_max_abs_vs_fp32"] <= TOLERANCE
                           and result[f"{name}_max_abs_vs_triton"] <= TOLERANCE
                           and result.get(f"{name}_all_heads_max_abs_vs_fp32", 0.0) <= TOLERANCE
                           for name in outputs)
    return result


def scale_view_checks(torch, modules: dict) -> dict:
    """Eclair's scale views address vLLM's scale bytes, and creating them writes nothing."""
    triton_impl, eclair_impl = modules["triton"], modules["eclair"]
    sentinel = torch.full((8, KV_HEADS, modules["block"], 2 * (DIM + 4)), 0x5A, dtype=torch.int8,
                          device="cuda")
    for impl in (triton_impl, eclair_impl):
        impl._k_scale_cache = impl._v_scale_cache = None
    eclair_impl._ensure_scale_caches(sentinel)
    untouched = bool((sentinel == 0x5A).all())
    triton_impl._ensure_scale_caches(sentinel)
    upstream_writes = not bool((sentinel == 0x5A).all())
    same = all(a.data_ptr() == b.data_ptr() and a.stride() == b.stride() and a.shape == b.shape
               and bool((a == b).all())
               for a, b in ((eclair_impl._k_scale_cache, triton_impl._k_scale_cache),
                            (eclair_impl._v_scale_cache, triton_impl._v_scale_cache)))
    return {"eclair_creation_writes_nothing": untouched, "upstream_creation_writes": upstream_writes,
            "views_match_upstream": same, "passed": untouched and upstream_writes and same}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="context:rows, for example 32768:8192")
    parser.add_argument("--block", type=int, default=16,
                        help="tokens per cache block; the 27B hybrid engine uses 1552")
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl, TritonAttentionMetadata
    from eclair_vllm import native
    from eclair_vllm.attention_sm70 import WINDOW, EclairSm70AttentionImpl, dequantize

    cases = [tuple(int(x) for x in c.split(":")) for c in (args.case or [
        "0:64", "1000:37", "0:2048", "8192:8192", "65536:8192", "196608:8192"])]
    record: dict = {"schema": "eclair.vllm-attention-checks/1", "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0), "tolerance_max_abs": TOLERANCE,
                    "kv_cache": f"int8_per_token_head, block {args.block}", "results": []}
    try:
        with set_current_vllm_config(VllmConfig()):
            settings = dict(num_heads=HEADS, head_size=DIM, scale=1.0 / math.sqrt(DIM),
                            num_kv_heads=KV_HEADS, alibi_slopes=None, sliding_window=None,
                            kv_cache_dtype="int8_per_token_head")
            modules = {"triton": TritonAttentionImpl(**settings),
                       "eclair": EclairSm70AttentionImpl(**settings),
                       "metadata": TritonAttentionMetadata, "dequantize": dequantize,
                       "block": args.block, "window": WINDOW, "native": native.current_library(),
                       "native_module": native}
            record["native_library"] = modules["native"] is not None
            record["scale_views"] = scale_view_checks(torch, modules)
            generator = torch.Generator(device="cuda").manual_seed(11)
            for context, rows in cases:
                record["results"].append(run_case(torch, modules, context, rows, generator))
                torch.cuda.empty_cache()
        record["passed"] = (record["scale_views"]["passed"]
                            and all(r["passed"] for r in record["results"]))
    except Exception as error:  # record setup failures verbatim
        record["passed"] = False
        record["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=1)[:6000])
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
