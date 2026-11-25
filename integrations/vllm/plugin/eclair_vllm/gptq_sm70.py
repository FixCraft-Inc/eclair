# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Explicit GPTQ experiment using vLLM's existing sm_70 linear kernels.

AutoGPTQLinearMethod in vLLM 0.32 checks Marlin eligibility before its kernel
chooser can select Exllama. This adapter admits only symmetric, sequential
4-bit group-128 dense layers and checks the selected kernel after creation.
Loading, packing and arithmetic stay in vLLM. No upstream object is patched.
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig, AutoGPTQLinearMethod
from vllm.model_executor.layers.quantization.utils.gptq_utils import get_linear_quant_method

from .wna16_sm70 import SM70_KERNELS, require_sm70

METHOD = "eclair-gptq-sm70"


class Sm70GPTQLinear(AutoGPTQLinearMethod):
    """Upstream weight creation and execution with admission for the tested format."""

    def __init__(self, config: AutoGPTQConfig) -> None:
        require_sm70(METHOD)
        if (config.weight_bits != 4 or not config.is_sym or config.group_size != 128
                or config.desc_act):
            raise ValueError(f"{METHOD} requires symmetric int4, group 128 and no act-order")
        # The upstream constructor initializes these descriptors, then checks
        # Marlin specifically. Here the actual kernel chooser owns admission.
        LinearMethodBase.__init__(self)
        self.quant_config = config
        self.quant_type = config.quant_type
        self.input_dtype = None

    def create_weights(self, layer, *args: Any, **kwargs: Any) -> None:
        super().create_weights(layer, *args, **kwargs)
        name = type(self.kernel).__name__
        if name not in SM70_KERNELS:
            raise RuntimeError(f"{METHOD}: selected {name}, which is not an sm_70 W4A16 kernel")


class EclairGPTQSm70Config(AutoGPTQConfig):
    @classmethod
    def get_name(cls) -> str:
        return METHOD

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg: dict[str, Any], user_quant: str | None,
                                     hf_config: Any = None) -> str | None:
        if user_quant != METHOD or hf_quant_cfg.get("quant_method") != "gptq":
            return None
        require_sm70(METHOD)
        return METHOD

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, RoutedExperts):
            raise ValueError(f"{METHOD} does not implement MoE")
        # Retain upstream handling of fused names, unquantized layers and
        # dynamic exclusions (including the checkpoint's BF16 MTP block).
        return get_linear_quant_method(self, layer, prefix, Sm70GPTQLinear)


def register() -> None:
    register_quantization_config(METHOD)(EclairGPTQSm70Config)
