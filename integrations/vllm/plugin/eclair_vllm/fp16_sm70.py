# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Opt-in skinny FP16 lm_head through the registered quantization config seam."""

from __future__ import annotations

import os

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod,
)

from . import native


class NativeLmHeadMethod(UnquantizedEmbeddingMethod):
    def __init__(self) -> None:
        super().__init__()
        self._library = None

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        self._library = None
        weight = layer.weight
        if weight.is_cuda and weight.dtype == torch.float16 and weight.is_contiguous():
            library = native.current_library()
            if library is not None and library.bf16_linear_ready:
                self._library = library
                # Owned with the method, immutable and ready before graph capture.
                self._activation_shifts = torch.zeros(4, dtype=torch.int32, device=weight.device)
                self._weight_shifts = torch.zeros(weight.shape[0], dtype=torch.int32,
                                                   device=weight.device)

    def apply_vocabulary_range(self, layer, x: torch.Tensor, begin: int, end: int) -> torch.Tensor | None:
        """Draft-only projection of contiguous rows; never change the shared head."""
        if (self._library is None or not x.is_cuda or x.dtype != torch.float16
                or x.dim() != 2 or not 1 <= x.shape[0] <= 4 or not x.is_contiguous()
                or x.device != layer.weight.device or not 0 <= begin < end <= layer.weight.shape[0]):
            return None
        return native.fp16_linear(self._library, x, layer.weight[begin:end],
                                   self._activation_shifts, self._weight_shifts)

    def apply(self, layer, x, bias=None):
        if (self._library is not None and bias is None and x.is_cuda and x.dtype == torch.float16
                and x.dim() == 2 and 1 <= x.shape[0] <= 4 and x.is_contiguous()
                and x.device == layer.weight.device):
            return native.fp16_linear(self._library, x, layer.weight,
                                       self._activation_shifts, self._weight_shifts)
        return super().apply(layer, x, bias)


def select_lm_head(layer, method):
    if (os.environ.get("ECLAIR_NATIVE_FP16_LINEAR") == "1" and method is None
            and isinstance(layer, ParallelLMHead)):
        return NativeLmHeadMethod()
    return method
