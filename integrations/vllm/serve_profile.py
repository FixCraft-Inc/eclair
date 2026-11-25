#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Write local vLLM 0.32.0 sm_70 arguments and environment settings to JSON.

The profile binds to loopback, disables usage statistics, and uses local
model files. The script itself starts no server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Vision: up to four images per request, no video. Images are resized to at most
# 1638400 pixels (1600 merged 32x32-pixel tokens), which bounds encoder memory.
IMAGES_PER_REQUEST = 4
IMAGE_PIXELS = {"shortest_edge": 65536, "longest_edge": 1638400}
# FlashInfer has no sm_70 kernels and is not installed, so its sampler is switched off.
ENVIRONMENT = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
# What vLLM forces when a request's `thinking_token_budget` runs out: Qwen's published
# early-stopping sentence, then the qwen3 parser's end marker, so the answer starts
# from a closed thought instead of a cut-off one.
THINKING_BUDGET_END = ("\n\nConsidering the limited time by the user, I have to give the "
                       "solution based on the thinking directly now.\n</think>")


def vllm_argv(venv: Path, model: Path, *, port: int = 18080, context: int = 262144,
              memory_fraction: float = 0.94, mtp: int = 0, graphs: str = "none",
              quantized: bool = True, batched_tokens: int = 8192,
              attention: str = "CUSTOM", state_checkpoints: str = "dense",
              vision: bool = False, budget_phrase: bool = False,
              kv_cache_dtype: str = "int8_per_token_head", prefix_match_unit: int | None = None,
              shared_prefix_checkpoint: bool = False) -> list[str]:
    if not 2048 <= context <= 262144 or not 0.5 <= memory_fraction <= 0.97:
        raise ValueError("context or memory fraction outside the supported profile")
    if graphs not in ("none", "decode"):
        raise ValueError("graphs must be none (eager) or decode (FULL_DECODE_ONLY, no torch.compile)")
    if attention not in ("CUSTOM", "TRITON_ATTN"):
        raise ValueError("attention must be CUSTOM (Eclair prefill) or TRITON_ATTN")
    if state_checkpoints not in ("dense", "boundaries"):
        raise ValueError("state checkpoints must be dense or boundaries")
    if kv_cache_dtype not in ("int8_per_token_head", "auto", "float16"):
        raise ValueError("KV cache dtype must be int8_per_token_head, auto or float16")
    if prefix_match_unit is not None and (type(prefix_match_unit) is not int
            or not 1 <= prefix_match_unit <= 4096):
        raise ValueError("prefix matching unit must be an integer in 1..4096")
    if shared_prefix_checkpoint and (prefix_match_unit is None or not mtp):
        raise ValueError("shared prefix checkpoints require finer matching and MTP")
    if not 0 <= mtp <= 4 or not 512 <= batched_tokens <= 32768 or not 1024 <= port <= 65535:
        raise ValueError("MTP depth, batch tokens or port outside the supported profile")
    executable = venv / "bin" / "vllm"
    if not executable.is_file() or not (model / "config.json").is_file():
        raise ValueError("vLLM executable or model directory is missing")
    argv = [str(executable), "serve", str(model), "--served-model-name", "eclair-qwen",
            "--host", "127.0.0.1", "--port", str(port), "--dtype", "float16",
            "--kv-cache-dtype", kv_cache_dtype, "--attention-backend", attention,
            "--max-model-len", str(context), "--max-num-seqs", "1",
            "--max-num-batched-tokens", str(batched_tokens),
            "--gpu-memory-utilization", str(memory_fraction), "--enable-prefix-caching",
            "--generation-config", "vllm",
            "--reasoning-parser", "qwen3", "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_xml", "--enable-prompt-tokens-details",
            "--disable-uvicorn-access-log"]
    if vision:
        # vLLM's FlashAttention has no sm_70 build, so the encoder uses PyTorch SDPA.
        argv += ["--limit-mm-per-prompt", json.dumps({"image": IMAGES_PER_REQUEST, "video": 0}),
                 "--mm-processor-kwargs", json.dumps({"size": IMAGE_PIXELS}),
                 "--mm-encoder-attn-backend", "TORCH_SDPA"]
    else:
        argv.append("--language-model-only")
    if quantized:
        argv += ["--quantization", "eclair-ct-sm70"]
    if state_checkpoints == "dense":
        # A prefix hit in this hybrid model needs the Gated DeltaNet state at the hit
        # position. vLLM 0.32 keeps it only at each prompt's end unless retention is
        # dense, and a compaction request branches earlier. Retained states are freed
        # cache blocks, evicted oldest first, so they do not reserve capacity.
        argv += ["--prefix-cache-retention-interval", "None"]
    if graphs == "none":
        argv.append("--enforce-eager")
    else:
        # Decode steps replay CUDA graphs. Prefill, including the Eclair tensor-core path
        # for chunks of 16+ query tokens, stays eager and torch.compile stays off.
        argv += ["-cc.mode=none", "-cc.cudagraph_mode=FULL_DECODE_ONLY"]
    if mtp:
        argv += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": mtp})]
    if budget_phrase:
        argv += ["--reasoning-config", json.dumps({"reasoning_end_str": THINKING_BUDGET_END})]
    if prefix_match_unit is not None:
        argv += ["--prefix-match-unit", str(prefix_match_unit)]
    if shared_prefix_checkpoint:
        argv.append("--enable-mamba-shared-prefix-checkpoint")
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--memory-fraction", type=float, default=0.94)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--graphs", choices=("none", "decode"), default="none",
                        help="none is fully eager; decode captures decode steps only")
    parser.add_argument("--unquantized", action="store_true")
    parser.add_argument("--batched-tokens", type=int, default=8192)
    parser.add_argument("--attention", choices=("CUSTOM", "TRITON_ATTN"), default="CUSTOM",
                        help="CUSTOM is the Eclair tensor-core prefill backend")
    parser.add_argument("--kv-cache-dtype", choices=("int8_per_token_head", "auto", "float16"),
                        default="int8_per_token_head",
                        help="float16 forces FP16 (auto may inherit checkpoint FP8); reduce context for A/B")
    parser.add_argument("--vision", action="store_true", help="load the vision encoder for images")
    parser.add_argument("--state-checkpoints", choices=("dense", "boundaries"), default="dense",
                        help="recurrent-state prefix checkpoints at every chunk or only prompt ends")
    parser.add_argument("--budget-phrase", action="store_true",
                        help="end an exhausted thinking_token_budget with Qwen's transition sentence")
    parser.add_argument("--prefix-match-unit", type=int, help="opt-in finer hybrid prefix matching")
    parser.add_argument("--shared-prefix-checkpoint", action="store_true",
                        help="opt-in Mamba checkpoint at a shared MTP prefix")
    parser.add_argument("--cache-control", action="store_true",
                        help="opt in to the separately installed authenticated cache-control plugin")
    args = parser.parse_args()
    try:
        argv = vllm_argv(args.venv, args.model, context=args.context,
                         memory_fraction=args.memory_fraction, mtp=args.mtp,
                         graphs=args.graphs, quantized=not args.unquantized,
                         batched_tokens=args.batched_tokens, attention=args.attention,
                         state_checkpoints=args.state_checkpoints, vision=args.vision,
                         budget_phrase=args.budget_phrase, kv_cache_dtype=args.kv_cache_dtype,
                         prefix_match_unit=args.prefix_match_unit,
                         shared_prefix_checkpoint=args.shared_prefix_checkpoint)
        environment = dict(ENVIRONMENT)
        if args.cache_control:
            environment["VLLM_PLUGINS"] = "eclair_ct_sm70,eclair_cache_control"
        with args.output.open("x") as output:
            json.dump({"argv": argv, "env": environment}, output, indent=2)
            output.write("\n")
    except (OSError, ValueError) as error:
        print(f"profile failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
