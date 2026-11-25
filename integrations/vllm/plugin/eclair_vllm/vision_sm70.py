# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Vision-encoder rotary embedding for vLLM builds without its FlashAttention extension.

vLLM's `ApplyRotaryEmb.forward_cuda` imports `vllm.vllm_flash_attn`, which raises
when neither FlashAttention extension was built, as on sm_70. Qwen2.5-VL-style
encoders, including Qwen3.5's, create the op with `enforce_enable=True`, so
configuration cannot select another path. This registers vLLM's own
out-of-tree replacement whose CUDA path is vLLM's PyTorch implementation.
"""

from __future__ import annotations

import importlib

from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


class EclairApplyRotaryEmb(ApplyRotaryEmb):
    """vLLM's `forward_native` rotary computation on CUDA."""

    def forward_cuda(self, x, cos, sin):
        return self.forward_native(x, cos, sin)


def flash_attention_importable() -> bool:
    """False exactly when `ApplyRotaryEmb.forward_cuda` would fail on its import."""
    try:
        importlib.import_module("vllm.vllm_flash_attn")
    except ImportError:
        return False
    return True


def register() -> None:
    if not flash_attention_importable() and ApplyRotaryEmb.__name__ not in op_registry_oot:
        ApplyRotaryEmb.register_oot(EclairApplyRotaryEmb)
