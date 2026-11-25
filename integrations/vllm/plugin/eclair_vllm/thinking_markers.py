# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Opt-in V2 logits processor that discourages hesitation markers while thinking is open.

Inside an open `<think>` span it subtracts a fixed penalty from the logits of a small set of
single-token hesitation words ("Wait", "Hmm", "Alternatively", ...). Answers, tool calls and
requests whose thinking is closed are unchanged. The processor imports nothing else from
Eclair, so an experiment may load this file on its own by fully qualified class name:

    --logits-processors eclair_vllm.thinking_markers:ThinkingMarkerPenalty

The penalty is `vllm_xargs={"eclair_marker_penalty": x}` per request, or the engine-wide
default `ECLAIR_MARKER_PENALTY` (unset or 0 leaves every request unchanged). It runs before
the thinking budget, so a forced budget ending still wins.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsContext,
        LogitsProcRequestState,
    )

ARGUMENT = "eclair_marker_penalty"
ENVIRONMENT = "ECLAIR_MARKER_PENALTY"
MAX_PENALTY = 10.0
# Hesitation and backtracking words only. Content words that coding reasoning needs
# ("or", "error", "instead", "could") are deliberately absent.
WORDS = ("wait", "hmm", "alternatively", "but", "actually", "perhaps", "maybe")
THINK_START, THINK_END = "<think>", "</think>"
COLD_SCAN_BLOCK = 1024


def penalty_value(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= MAX_PENALTY:
        raise ValueError(f"{ARGUMENT} must be a number between 0 and {MAX_PENALTY}")
    return float(value)


def default_penalty() -> float:
    text = os.environ.get(ENVIRONMENT, "").strip()
    try:
        return penalty_value(float(text)) if text else 0.0
    except ValueError:
        raise ValueError(f"{ENVIRONMENT} must be a number between 0 and {MAX_PENALTY}") from None


def marker_token_ids(encode, token_id) -> tuple[list[int], int, int]:
    """Single-token spellings of WORDS (bare, capitalized, with a leading space) and the
    thinking start and end IDs, from a tokenizer's `encode(text) -> ids` and `token_id(text)`."""
    ids = set()
    for word in WORDS:
        for spelling in (word, word.capitalize(), " " + word, " " + word.capitalize()):
            encoded = encode(spelling)
            if len(encoded) == 1:
                ids.add(encoded[0])
    start, end = token_id(THINK_START), token_id(THINK_END)
    if not ids or start is None or end is None or {start, end} & ids:
        raise ValueError("tokenizer lacks the thinking markers or single-token hesitation words")
    return sorted(ids), start, end


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def scan_committed(slots_ptr, active_ptr, all_ids_ptr, all_ids_stride, total_len_ptr,
                       last_start_ptr, last_end_ptr, scan_pos_ptr, START: tl.constexpr,
                       END: tl.constexpr, BLOCK: tl.constexpr):
        # One program per request: fold newly committed tokens into the last start/end positions.
        slot = tl.load(slots_ptr + tl.program_id(0))
        if tl.load(active_ptr + slot) == 0:
            return
        total = tl.load(total_len_ptr + slot)
        position = tl.load(scan_pos_ptr + slot)
        last_start = tl.load(last_start_ptr + slot)
        last_end = tl.load(last_end_ptr + slot)
        if position > total:
            position = 0
            last_start = -1
            last_end = -1
        row = all_ids_ptr + slot.to(tl.int64) * all_ids_stride
        for low in tl.range(position, total, BLOCK):
            offsets = low + tl.arange(0, BLOCK)
            ids = tl.load(row + offsets, mask=offsets < total, other=-1)
            last_start = tl.maximum(last_start, tl.max(tl.where(ids == START, offsets, -1), axis=0))
            last_end = tl.maximum(last_end, tl.max(tl.where(ids == END, offsets, -1), axis=0))
        tl.store(last_start_ptr + slot, last_start)
        tl.store(last_end_ptr + slot, last_end)
        tl.store(scan_pos_ptr + slot, total)

    @triton.jit
    def penalize(logits_ptr, logits_stride, rows_to_slots_ptr, local_pos_ptr, input_ids_ptr,
                 active_ptr, penalty_ptr, last_start_ptr, last_end_ptr, markers_ptr,
                 NUM_MARKERS: tl.constexpr, MARKER_BLOCK: tl.constexpr, START: tl.constexpr,
                 END: tl.constexpr):
        # One program per logits row. Draft tokens before this row (input positions 1..local)
        # can open or close thinking after the committed tokens.
        row = tl.program_id(0).to(tl.int64)
        slot = tl.load(rows_to_slots_ptr + row)
        if tl.load(active_ptr + slot) == 0:
            return
        local = tl.load(local_pos_ptr + row)
        first = row - local
        state = tl.where(tl.load(last_start_ptr + slot) > tl.load(last_end_ptr + slot), 1, 0)
        for draft in tl.range(1, local + 1):
            token = tl.load(input_ids_ptr + first + draft)
            state = tl.where(token == START, 1, tl.where(token == END, 0, state))
        if state == 0:
            return
        penalty = tl.load(penalty_ptr + slot)
        offsets = tl.arange(0, MARKER_BLOCK)
        markers = tl.load(markers_ptr + offsets, mask=offsets < NUM_MARKERS, other=0)
        pointers = logits_ptr + row * logits_stride + markers
        values = tl.load(pointers, mask=offsets < NUM_MARKERS, other=0.0)
        tl.store(pointers, values - penalty, mask=offsets < NUM_MARKERS)

    return scan_committed, penalize


class ThinkingMarkerPenalty(LogitsProcessor):
    """Subtract a per-request penalty from hesitation-marker logits while thinking is open."""

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState) -> None:
        from tokenizers import Tokenizer

        path = os.path.join(vllm_config.model_config.tokenizer, "tokenizer.json")
        tokenizer = Tokenizer.from_file(path)
        self.configure(req_states, *marker_token_ids(
            lambda text: tokenizer.encode(text, add_special_tokens=False).ids, tokenizer.token_to_id))

    def configure(self, req_states: LogitsProcRequestState, markers: list[int], start: int,
                  end: int) -> None:
        import numpy as np
        import torch

        self.markers, self.start, self.end = markers, start, end
        self.default = default_penalty()
        self.req_states = req_states
        device, slots = req_states.device, req_states.max_num_reqs
        self.marker_ids = torch.tensor(self.markers, dtype=torch.int32, device=device)
        self.penalty = np.zeros(slots, dtype=np.float32)
        self.active = np.zeros(slots, dtype=np.int32)
        self.penalty_gpu = torch.zeros(slots, dtype=torch.float32, device=device)
        self.active_gpu = torch.zeros(slots, dtype=torch.int32, device=device)
        self.last_start = torch.full((slots,), -1, dtype=torch.int32, device=device)
        self.last_end = torch.full((slots,), -1, dtype=torch.int32, device=device)
        self.scan_pos = torch.zeros(slots, dtype=torch.int32, device=device)
        self.reset: list[int] = []
        self.dirty = False
        self.scan_committed, self.penalize = _kernels()

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        if ARGUMENT in extra:
            penalty_value(extra[ARGUMENT])

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        extra = sampling_params.extra_args or {}
        penalty = penalty_value(extra[ARGUMENT]) if ARGUMENT in extra else self.default
        active = int(penalty > 0)
        if self.penalty[req_idx] != penalty or self.active[req_idx] != active:
            self.penalty[req_idx], self.active[req_idx] = penalty, active
            self.dirty = True
        if active:
            self.reset.append(req_idx)
        return bool(active)

    def apply_staged_writes(self) -> None:
        import torch

        if self.dirty:
            self.penalty_gpu.copy_(torch.from_numpy(self.penalty), non_blocking=False)
            self.active_gpu.copy_(torch.from_numpy(self.active), non_blocking=False)
            self.dirty = False
        if self.reset:
            index = torch.tensor(self.reset, dtype=torch.int64, device=self.scan_pos.device)
            self.last_start.index_fill_(0, index, -1)
            self.last_end.index_fill_(0, index, -1)
            self.scan_pos.index_fill_(0, index, 0)
            self.reset.clear()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self.active[ctx.idx_mapping_np].any():
            return logits
        all_ids = self.req_states.all_token_ids.gpu
        self.scan_committed[(ctx.idx_mapping.shape[0],)](
            ctx.idx_mapping, self.active_gpu, all_ids, all_ids.stride(0),
            self.req_states.total_len.gpu, self.last_start, self.last_end, self.scan_pos,
            START=self.start, END=self.end, BLOCK=COLD_SCAN_BLOCK)
        self.penalize[(logits.shape[0],)](
            logits, logits.stride(0), ctx.expanded_idx_mapping, ctx.expanded_local_pos,
            ctx.input_ids, self.active_gpu, self.penalty_gpu, self.last_start, self.last_end,
            self.marker_ids, NUM_MARKERS=len(self.markers),
            MARKER_BLOCK=max(16, 1 << (len(self.markers) - 1).bit_length()),
            START=self.start, END=self.end)
        return logits
