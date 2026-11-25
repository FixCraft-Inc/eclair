# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""compressed-tensors W4A16 on sm_70 through kernels that vLLM itself provides for it.

vLLM 0.32.0 gates its whole W4A16 scheme at compute capability 7.5, although
its kernel chooser also offers ExllamaLinearKernel (minimum 6.0) and
TritonW4A16LinearKernel (no minimum). This method admits the scheme on sm_70
only when the chooser selects one of those two for the layer, and fails the
layer otherwise. Every other scheme keeps vLLM's own capability gate. The
numerics are vLLM's unless an opt-in Eclair path serves the call:
ECLAIR_NATIVE_W4A16=1 for captured M=2..4 decode and
ECLAIR_NATIVE_W4A16_PREFILL=1 for eager calls within each shape's measured
row range. Both cover only measured layouts and shapes and keep Exllama for
everything else.
Eclair's V100 checks compare them with an independent dequantization of the
checkpoint.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from vllm.platforms import current_platform

from . import METHOD, native

SM70_KERNELS = frozenset({"ExllamaLinearKernel", "TritonW4A16LinearKernel"})
# Split counts for captured decode at M=2, M=3 and M=4.
NATIVE_SPLITS = {(5120, 6144): (8, 8, 8), (5120, 17408): (8, 16, 8),
                 (34816, 5120): (4, 4, 4), (16384, 5120): (4, 4, 4),
                 (14336, 5120): (8, 4, 4)}
# Measured row ranges for the eager GEMM path, keyed by (N, K).
# Outside these ranges the plugin uses Exllama.
PREFILL_ROWS = {(5120, 6144): (32, 384), (5120, 17408): (32, 64),
                (34816, 5120): (32, 1024), (16384, 5120): (32, 1536),
                (14336, 5120): (32, 1536)}


def captured_decode() -> bool:
    """Only full uniform decode graphs; eager calls and prefill retain Exllama."""
    if not is_forward_context_available():
        return False
    context = get_forward_context()
    batch = context.batch_descriptor
    # V2 captures its full graph inside mode NONE with no BatchDescriptor;
    # the outer manager owns the graph. Preparation admitted only the
    # FULL_DECODE_ONLY profile, so this cannot capture a prefill graph.
    return (context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and (batch is None or batch.uniform)) or (
                batch is None and torch.cuda.is_current_stream_capturing())


def require_sm70(method: str = METHOD) -> None:
    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() != 70:
        raise ValueError(f"{method} is only for compute capability 7.0 devices")


class Sm70WNA16(CompressedTensorsWNA16):
    """The upstream scheme, admitted on sm_70 when an sm_70-capable kernel serves the layer."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(self, layer, *args: Any, **kwargs: Any) -> None:
        super().create_weights(layer, *args, **kwargs)
        name = type(self.kernel).__name__
        if name not in SM70_KERNELS:
            raise RuntimeError(f"{METHOD}: layer {self.layer_name} selected {name}, "
                               "which is not an sm_70 W4A16 kernel")

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        self._native_library = None
        self._prefill_gemm = None
        decode = os.environ.get("ECLAIR_NATIVE_W4A16") == "1"
        prefill = os.environ.get("ECLAIR_NATIVE_W4A16_PREFILL") == "1"
        if (not (decode or prefill)
                or type(self.kernel).__name__ != "ExllamaLinearKernel"
                or self.num_bits != 4 or self.group_size != 128 or not self.symmetric):
            return
        weights, scales = layer.weight_packed, layer.weight_scale
        n, k = layer.output_size_per_partition, layer.input_size_per_partition
        if ((n, k) not in NATIVE_SPLITS or weights.shape != (k // 8, n)
                or scales.shape != (k // 128, n) or weights.dtype != torch.int32
                or scales.dtype != torch.float16 or not weights.is_cuda
                or weights.device != scales.device
                or not weights.is_contiguous() or not scales.is_contiguous()):
            return
        library = native.current_library()
        if library is None:
            return
        self._native_shape = n, k
        if (decode and library.w4a16_linear_ready
                and get_current_vllm_config().compilation_config.cudagraph_mode
                == CUDAGraphMode.FULL_DECODE_ONLY):
            self._native_library = library
        if prefill and (n, k) in PREFILL_ROWS and library.w4a16_gemm_ready:
            context = native.w4a16_gemm_context(library)
            if context is not None:
                self._prefill_gemm = library, context, PREFILL_ROWS[(n, k)]

    def apply_weights(self, layer, x, bias):
        if (self._native_library is not None and bias is None and x.dim() == 2
                and 2 <= x.shape[0] <= 4 and x.is_cuda and x.dtype == torch.float16
                and x.device == layer.weight_packed.device and x.is_contiguous()
                and x.shape[1] == self._native_shape[1] and captured_decode()):
            splits = NATIVE_SPLITS[self._native_shape][x.shape[0] - 2]
            return native.w4a16_linear(self._native_library, x, layer.weight_packed,
                                       layer.weight_scale, splits)
        # Prefill runs eager. A capture keeps Exllama rather than recording
        # per-call temporaries.
        if (self._prefill_gemm is not None and bias is None and x.dim() == 2
                and self._prefill_gemm[2][0] <= x.shape[0] <= self._prefill_gemm[2][1]
                and x.is_cuda and x.dtype == torch.float16
                and x.device == layer.weight_packed.device and x.is_contiguous()
                and x.shape[1] == self._native_shape[1]
                and not torch.cuda.is_current_stream_capturing()):
            library, context, _ = self._prefill_gemm
            return native.w4a16_gemm(library, context, x, layer.weight_packed,
                                     layer.weight_scale)
        return super().apply_weights(layer, x, bias)


class EclairCompressedTensorsSm70Config(CompressedTensorsConfig):
    def get_quant_method(self, layer, prefix: str):
        from .fp16_sm70 import select_lm_head

        return select_lm_head(layer, super().get_quant_method(layer, prefix))

    def get_name(self) -> str:
        return METHOD

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg: dict[str, Any], user_quant: str | None,
                                     hf_config: Any = None) -> str | None:
        # Only an explicit request claims a checkpoint. vLLM then rejects a non-matching format.
        if user_quant != METHOD or hf_quant_cfg.get("quant_method") != "compressed-tensors":
            return None
        require_sm70()
        return METHOD

    def _get_scheme_from_parts(self, *args: Any, **kwargs: Any):
        scheme = super()._get_scheme_from_parts(*args, **kwargs)
        if type(scheme) is not CompressedTensorsWNA16:
            return scheme
        return Sm70WNA16(strategy=scheme.strategy, num_bits=scheme.num_bits,
                         group_size=None if scheme.group_size == -1 else scheme.group_size,
                         symmetric=scheme.symmetric, layer_name=scheme.layer_name)


def register() -> None:
    register_quantization_config(METHOD)(EclairCompressedTensorsSm70Config)
