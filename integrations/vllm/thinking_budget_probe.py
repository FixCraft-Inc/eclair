#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""CPU-only check of the legacy vLLM 0.32 thinking-budget tracker.

Uses synthetic token IDs and an isolated budget tracker, never a model or server.
The bare-end control changes only that synthetic tracker's configuration. Neither
mode changes a serving profile, installed module, client setting or GPU state.
This is not Model Runner V2's separate GPU tracker, used by adopted Eclair.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def probe(end_mode: str) -> dict:
    """Exercise the installed legacy tracker with and without a transition phrase."""
    # Imports stay here so --help does not load torch or vLLM.
    import torch
    from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

    installed_version = version("vllm")
    if installed_version.split("+", 1)[0] != "0.32.0":
        raise ValueError("this probe requires vLLM 0.32.0; recheck its private tracker API")
    if end_mode not in ("phrase", "bare"):
        raise ValueError("end mode must be phrase or bare")
    if torch.cuda.is_initialized():
        raise ValueError("the probe requires an uninitialized CUDA runtime")

    # Natural start/end are distinct single tokens; forcing is a longer sequence.
    # Only CPU tensors are used. The budget matches production's 8192 setting.
    start, natural, forced = [1], [2], [3, 4, 2]
    end = forced if end_mode == "phrase" else natural
    budget = 8192
    rows = []
    for spec_tokens in (0, 2):
        for case in ("before_budget", "natural_end", "closed_prompt", "budget_exhausted"):
            config = SimpleNamespace(reasoning_start_token_ids=start,
                                     reasoning_end_token_ids=end,
                                     natural_reasoning_end_token_ids=natural)
            holder = ThinkingBudgetStateHolder(config, 1, spec_tokens,
                                              torch.device("cpu"), False)
            prompt = start + [0] + (natural if case == "closed_prompt" else [])
            holder._state = {0: holder._init_state_entry(prompt, budget)}
            output = [0, 0] + (natural if case == "natural_end" else [])
            if case in ("budget_exhausted", "closed_prompt"):
                output += [0] * (budget + 4)
            holder.update_state([output], [[]])
            logits = torch.zeros((1, 16), device="cpu")
            holder.apply_to_logits(logits, spec_tokens > 0, [[]])
            changed = torch.nonzero(logits[0], as_tuple=False).flatten().tolist()
            state = holder._state[0]
            initial = {"in_think": state["in_think"], "in_end": state["in_end"],
                       "changed_logit_ids": changed}
            expected = [end[0]] if case == "budget_exhausted" else []
            passed = changed == expected
            if case in ("natural_end", "closed_prompt"):
                passed = passed and not state["in_think"] and not state["in_end"]
            if case == "natural_end":
                # Answer text must never restart forcing after a natural close.
                output += [0] * (budget + 4)
                holder.update_state([output], [[]])
                logits.zero_()
                holder.apply_to_logits(logits, spec_tokens > 0, [[]])
                after_answer = torch.nonzero(logits[0], as_tuple=False).flatten().tolist()
                passed = passed and not after_answer
            else:
                after_answer = None
            rows.append({"speculative_capacity": spec_tokens, "case": case,
                         "initial": initial, "expected_changed_logit_ids": expected,
                         "changed_ids_after_answer": after_answer, "passed": passed})
    initialized = torch.cuda.is_initialized()
    return {"vllm_version": installed_version, "torch_version": torch.__version__,
            "tracker": "legacy v1.sample.thinking_budget_state.ThinkingBudgetStateHolder",
            "adopted_model_runner": "V2; not exercised by this CPU probe",
            "device": "cpu", "synthetic_token_ids": True, "budget": budget,
            "speculative_note": "empty draft lists; no GPU rejection-sampling claim",
            "end_mode": end_mode, "cuda_initialized": initialized, "cases": rows,
            "passed": all(row["passed"] for row in rows) and not initialized}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-mode", choices=("phrase", "bare"), default="phrase")
    parser.add_argument("--output", type=Path, help="new private JSON file; never overwritten")
    args = parser.parse_args()
    try:
        result = probe(args.end_mode)
        encoded = json.dumps(result, indent=2) + "\n"
        if args.output is not None:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600)
            with os.fdopen(fd, "w") as output:
                output.write(encoded)
        print(encoded, end="")
        return 0 if result["passed"] else 1
    except (ImportError, OSError, ValueError) as error:
        print(f"thinking budget probe: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
