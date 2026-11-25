# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Experimental greedy MTP vocabulary prefix, registered only on explicit opt-in.

The proposer treats greedy drafts as point masses. Standard rejection sampling
accepts a proposal with its target probability and draws rejected tokens from
the target distribution excluding that proposal. Changing the proposal leaves
that distribution intact. Probabilistic drafts need full proposal probabilities
and are refused here. Target logits and the shared head object are unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm import ModelRegistry
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

from .fp16_sm70 import NativeLmHeadMethod
from .wna16_sm70 import require_sm70


def vocabulary_prefix(value: str) -> int:
    """A bounded, tile-aligned prefix of the pinned model's token IDs."""
    if not value.isascii() or not value.isdecimal():
        raise ValueError("ECLAIR_MTP_DRAFT_VOCAB must be an integer")
    count = int(value)
    if not 8192 <= count <= 131072 or count % 256:
        raise ValueError("ECLAIR_MTP_DRAFT_VOCAB must be 8192..131072 in multiples of 256")
    return count


def validate_config(config: Any) -> None:
    spec = config.speculative_config
    if (spec is None or spec.method != "mtp" or spec.draft_sample_method != "greedy"
            or spec.rejection_sample_method != "standard"):
        raise ValueError("draft vocabulary prefix requires greedy MTP and standard rejection")
    parallel = config.parallel_config
    text = config.model_config.hf_text_config
    if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or config.lora_config is not None or text.hidden_size != 5120
            or text.vocab_size != 248320 or text.tie_word_embeddings):
        raise ValueError("draft vocabulary prefix requires the pinned single-GPU Qwen head")


class PrefixQwenMTP(Qwen3_5MTP):
    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        require_sm70()
        validate_config(vllm_config)
        self._draft_vocabulary = vocabulary_prefix(os.environ["ECLAIR_MTP_DRAFT_VOCAB"])
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_logits(self, hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> torch.Tensor | None:
        method = self.lm_head.quant_method
        if isinstance(method, NativeLmHeadMethod):
            logits = method.apply_vocabulary_range(
                self.lm_head, hidden_states, 0, self._draft_vocabulary)
            # All added/control tokens are >=248044 in the pinned tokenizer.
            # Include its last 512 rows so drafts can propose EOS, thinking
            # delimiters and tool markers as well as ordinary prefix tokens.
            tail_start = self.config.vocab_size - 512
            tail = method.apply_vocabulary_range(
                self.lm_head, hidden_states, tail_start, self.config.vocab_size)
            if logits is not None and tail is not None:
                # Full global vocabulary shape keeps the model interface and token
                # IDs intact. This allocates logits only, with no extra weight copy.
                padded = torch.nn.functional.pad(
                    logits, (0, self.config.vocab_size - self._draft_vocabulary),
                    value=float("-inf"))
                padded[:, tail_start:].copy_(tail)
                return padded
        return super().compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states).argmax(dim=-1)


def register() -> None:
    vocabulary_prefix(os.environ["ECLAIR_MTP_DRAFT_VOCAB"])
    ModelRegistry.register_model("Qwen3_5MTP", "eclair_vllm.mtp_sm70:PrefixQwenMTP")
