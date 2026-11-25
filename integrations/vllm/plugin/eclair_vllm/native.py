# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""ctypes bindings of Eclair's native kernels (libeclair_kernels) for the sm_70 plugin.

The library implements eclair/ops/paged_attention.hpp (paged int8 prefill and
batched decode attention, reading vLLM's int8 per-token-head cache directly) and
eclair/ops/gated_delta*.hpp (Gated DeltaNet chunk state and output passes) with Volta
tensor cores, and eclair/ops/w4a16_linear.hpp (skinny decode rows natively, and
large-M calls as an exact reconstruction followed by an FP32-accumulating cuBLAS GEMM). It is found next to this module, where install_plugin.py places
it, or at the absolute path in ECLAIR_KERNELS_LIBRARY. ECLAIR_NATIVE_KERNELS=0
disables it, which selects the plugin's and vLLM's own paths for comparison.
It links the CUDA runtime dynamically and shares the copy PyTorch loaded.
Calls enqueue on the current PyTorch stream and synchronize nothing. Every
call checks its tensors' layouts and readable extents. A missing library or an
unsupported layout is reported, never guessed around.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

PREFILL_ABI = 1
DECODE_ABI = 1
GATED_DELTA_ABI = 1
# Context splits per sequence and KV head in the native decode, measured on the V100.
DECODE_SPLITS = 64
LIBRARY_NAME = "libeclair_kernels.so"
MIN_ROWS = 16
CHUNK = 64

# Device index -> the loaded library, or None when it is absent, disabled or
# cannot run there. One load per worker process.
_LIBRARIES: dict[int, ctypes.CDLL | None] = {}
# Device index -> the large-M W4A16 cuBLAS context. Created once and kept
# for the worker's lifetime: destroying it at interpreter exit could run
# after CUDA teardown.
_GEMMS: dict[int, ctypes.c_void_p] = {}
# cuBLAS split-K scratch per large-M call, from PyTorch's allocator.
GEMM_WORKSPACE_BYTES = 32 * 1024 * 1024


class _Args(ctypes.Structure):
    """EclairPagedPrefillArgs of include/eclair/c/paged_attention.h, ABI 1."""

    _fields_ = [("abi", ctypes.c_uint32),
                ("query", ctypes.c_void_p), ("query_elements", ctypes.c_uint64),
                ("query_row_stride", ctypes.c_uint64),
                ("output", ctypes.c_void_p), ("output_elements", ctypes.c_uint64),
                ("output_row_stride", ctypes.c_uint64),
                ("keys", ctypes.c_void_p), ("values", ctypes.c_void_p),
                ("cache_bytes", ctypes.c_uint64),
                ("key_scales", ctypes.c_void_p), ("value_scales", ctypes.c_void_p),
                ("scale_elements", ctypes.c_uint64),
                ("block_table", ctypes.c_void_p), ("block_table_entries", ctypes.c_uint64),
                ("blocks", ctypes.c_uint64), ("block_size", ctypes.c_uint64),
                ("kv_heads", ctypes.c_uint64), ("head_size", ctypes.c_uint64),
                ("block_stride", ctypes.c_uint64), ("slot_stride", ctypes.c_uint64),
                ("head_stride", ctypes.c_uint64),
                ("scale_block_stride", ctypes.c_uint64), ("scale_slot_stride", ctypes.c_uint64),
                ("scale_head_stride", ctypes.c_uint64),
                ("heads", ctypes.c_uint64), ("rows", ctypes.c_uint64),
                ("context", ctypes.c_uint64), ("scale", ctypes.c_float)]


class _DecodeArgs(ctypes.Structure):
    """EclairPagedDecodeArgs of include/eclair/c/paged_attention.h, ABI 1."""

    _fields_ = [("abi", ctypes.c_uint32),
                ("query", ctypes.c_void_p), ("query_elements", ctypes.c_uint64),
                ("query_row_stride", ctypes.c_uint64),
                ("output", ctypes.c_void_p), ("output_elements", ctypes.c_uint64),
                ("output_row_stride", ctypes.c_uint64),
                ("keys", ctypes.c_void_p), ("values", ctypes.c_void_p),
                ("cache_bytes", ctypes.c_uint64),
                ("key_scales", ctypes.c_void_p), ("value_scales", ctypes.c_void_p),
                ("scale_elements", ctypes.c_uint64),
                ("block_tables", ctypes.c_void_p), ("block_table_entries", ctypes.c_uint64),
                ("block_table_stride", ctypes.c_uint64),
                ("max_blocks_per_sequence", ctypes.c_uint64),
                ("query_starts", ctypes.c_void_p), ("seq_lens", ctypes.c_void_p),
                ("workspace", ctypes.c_void_p), ("workspace_elements", ctypes.c_uint64),
                ("blocks", ctypes.c_uint64), ("block_size", ctypes.c_uint64),
                ("kv_heads", ctypes.c_uint64), ("head_size", ctypes.c_uint64),
                ("block_stride", ctypes.c_uint64), ("slot_stride", ctypes.c_uint64),
                ("head_stride", ctypes.c_uint64),
                ("scale_block_stride", ctypes.c_uint64), ("scale_slot_stride", ctypes.c_uint64),
                ("scale_head_stride", ctypes.c_uint64),
                ("heads", ctypes.c_uint64), ("sequences", ctypes.c_uint64),
                ("rows", ctypes.c_uint64), ("max_rows_per_sequence", ctypes.c_uint64),
                ("splits", ctypes.c_uint64), ("scale", ctypes.c_float)]


class _GatedDeltaArgs(ctypes.Structure):
    """EclairGatedDeltaStatesArgs of include/eclair/c/gated_delta.h, ABI 1."""

    _fields_ = [("abi", ctypes.c_uint32), ("initial_format", ctypes.c_uint32),
                ("keys", ctypes.c_void_p), ("keys_elements", ctypes.c_uint64),
                ("w", ctypes.c_void_p), ("w_elements", ctypes.c_uint64),
                ("u", ctypes.c_void_p), ("u_elements", ctypes.c_uint64),
                ("gates", ctypes.c_void_p), ("gates_elements", ctypes.c_uint64),
                ("initial", ctypes.c_void_p), ("initial_elements", ctypes.c_uint64),
                ("sequence_starts", ctypes.c_void_p), ("chunk_starts", ctypes.c_void_p),
                ("states", ctypes.c_void_p), ("states_elements", ctypes.c_uint64),
                ("new_values", ctypes.c_void_p), ("new_values_elements", ctypes.c_uint64),
                ("final_states", ctypes.c_void_p), ("final_states_elements", ctypes.c_uint64),
                ("tokens", ctypes.c_uint64), ("sequences", ctypes.c_uint64),
                ("chunks", ctypes.c_uint64), ("heads", ctypes.c_uint64),
                ("key_heads", ctypes.c_uint64), ("key_size", ctypes.c_uint64),
                ("value_size", ctypes.c_uint64)]


class _GatedDeltaOutputArgs(ctypes.Structure):
    """EclairGatedDeltaOutputArgs of include/eclair/c/gated_delta_output.h, ABI 1."""

    _fields_ = [("abi", ctypes.c_uint32),
                ("query", ctypes.c_void_p), ("query_elements", ctypes.c_uint64),
                ("keys", ctypes.c_void_p), ("keys_elements", ctypes.c_uint64),
                ("values", ctypes.c_void_p), ("values_elements", ctypes.c_uint64),
                ("states", ctypes.c_void_p), ("states_elements", ctypes.c_uint64),
                ("gates", ctypes.c_void_p), ("gates_elements", ctypes.c_uint64),
                ("sequence_starts", ctypes.c_void_p),
                ("sequence_starts_elements", ctypes.c_uint64),
                ("chunk_indices", ctypes.c_void_p), ("chunk_indices_elements", ctypes.c_uint64),
                ("output", ctypes.c_void_p), ("output_elements", ctypes.c_uint64),
                ("tokens", ctypes.c_uint64), ("sequences", ctypes.c_uint64),
                ("chunks", ctypes.c_uint64), ("heads", ctypes.c_uint64),
                ("key_heads", ctypes.c_uint64), ("key_size", ctypes.c_uint64),
                ("value_size", ctypes.c_uint64), ("scale", ctypes.c_float)]


class NativeKernelError(RuntimeError):
    """The library refused a call it should have accepted, or failed to enqueue it."""


class _Bf16LinearArgs(ctypes.Structure):
    """eclair/c/bf16_linear.h, ABI 1."""

    _fields_ = [("abi", ctypes.c_uint32),
                ("activations", ctypes.c_void_p), ("activation_elements", ctypes.c_uint64),
                ("activation_shifts", ctypes.c_void_p), ("activation_shift_elements", ctypes.c_uint64),
                ("weights", ctypes.c_void_p), ("weight_elements", ctypes.c_uint64),
                ("weight_shifts", ctypes.c_void_p), ("weight_shift_elements", ctypes.c_uint64),
                ("bias", ctypes.c_void_p), ("bias_elements", ctypes.c_uint64),
                ("output", ctypes.c_void_p), ("output_elements", ctypes.c_uint64),
                ("m", ctypes.c_uint64), ("n", ctypes.c_uint64), ("k", ctypes.c_uint64)]


class _W4A16GemmArgs(ctypes.Structure):
    """eclair/c/w4a16_gemm.h, ABI 1; the workspace is in bytes."""

    _fields_ = [("abi", ctypes.c_uint32),
                ("activations", ctypes.c_void_p), ("activation_elements", ctypes.c_uint64),
                ("weights", ctypes.c_void_p), ("weight_elements", ctypes.c_uint64),
                ("scales", ctypes.c_void_p), ("scale_elements", ctypes.c_uint64),
                ("dequantized", ctypes.c_void_p), ("dequantized_elements", ctypes.c_uint64),
                ("output", ctypes.c_void_p), ("output_elements", ctypes.c_uint64),
                ("workspace", ctypes.c_void_p), ("workspace_bytes", ctypes.c_uint64),
                ("m", ctypes.c_uint64), ("n", ctypes.c_uint64), ("k", ctypes.c_uint64)]


class _W4A16LinearArgs(ctypes.Structure):
    """eclair/c/w4a16_linear.h, ABI 1; scratch is private to each call."""

    _fields_ = [("abi", ctypes.c_uint32),
                ("activations", ctypes.c_void_p), ("activation_elements", ctypes.c_uint64),
                ("weights", ctypes.c_void_p), ("weight_elements", ctypes.c_uint64),
                ("scales", ctypes.c_void_p), ("scale_elements", ctypes.c_uint64),
                ("output", ctypes.c_void_p), ("output_elements", ctypes.c_uint64),
                ("workspace", ctypes.c_void_p), ("workspace_elements", ctypes.c_uint64),
                ("m", ctypes.c_uint64), ("n", ctypes.c_uint64), ("k", ctypes.c_uint64),
                ("splits", ctypes.c_uint64)]


def library_path() -> Path:
    override = os.environ.get("ECLAIR_KERNELS_LIBRARY")
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError("ECLAIR_KERNELS_LIBRARY must be an absolute path")
        return path
    return Path(__file__).with_name(LIBRARY_NAME)


def load(device_index: int) -> ctypes.CDLL | None:
    """The library when it exists, is enabled and holds kernel images this device can load."""
    path = library_path()
    if os.environ.get("ECLAIR_NATIVE_KERNELS") == "0" or not path.is_file():
        return None
    library = ctypes.CDLL(str(path))
    library.eclair_paged_prefill_int8.argtypes = [ctypes.POINTER(_Args), ctypes.c_void_p]
    library.eclair_gated_delta_states.argtypes = [ctypes.POINTER(_GatedDeltaArgs),
                                                  ctypes.c_void_p]
    library.eclair_paged_prefill_heads_supported.argtypes = [ctypes.c_uint64] * 3
    for name in ("eclair_paged_prefill_int8", "eclair_gated_delta_states",
                 "eclair_paged_prefill_heads_supported", "eclair_paged_prefill_image",
                 "eclair_gated_delta_image"):
        getattr(library, name).restype = ctypes.c_int
    for name in ("eclair_paged_prefill_image", "eclair_gated_delta_image"):
        getattr(library, name).argtypes = [ctypes.c_int]
    images = (library.eclair_paged_prefill_image(device_index),
              library.eclair_gated_delta_image(device_index))
    if images != (0, 0):
        return None
    # A library built before the decode kernels lacks its symbols and still serves prefill.
    library.decode_ready = False
    library.gated_delta_output_ready = False
    library.bf16_linear_ready = False
    library.w4a16_linear_ready = False
    library.w4a16_gemm_ready = False
    if (os.environ.get("ECLAIR_NATIVE_W4A16_PREFILL") == "1"
            and all(hasattr(library, name) for name in (
                "eclair_w4a16_gemm_create", "eclair_w4a16_gemm", "eclair_w4a16_gemm_image"))):
        library.eclair_w4a16_gemm_create.argtypes = [ctypes.c_int,
                                                     ctypes.POINTER(ctypes.c_void_p)]
        library.eclair_w4a16_gemm_create.restype = ctypes.c_int
        library.eclair_w4a16_gemm.argtypes = [ctypes.c_void_p, ctypes.POINTER(_W4A16GemmArgs),
                                              ctypes.c_void_p]
        library.eclair_w4a16_gemm.restype = ctypes.c_int
        library.eclair_w4a16_gemm_image.argtypes = [ctypes.c_int]
        library.eclair_w4a16_gemm_image.restype = ctypes.c_int
        library.w4a16_gemm_ready = library.eclair_w4a16_gemm_image(device_index) == 0
    if (os.environ.get("ECLAIR_NATIVE_W4A16") == "1"
            and hasattr(library, "eclair_w4a16_linear")
            and hasattr(library, "eclair_w4a16_linear_image")):
        library.eclair_w4a16_linear.argtypes = [ctypes.POINTER(_W4A16LinearArgs), ctypes.c_void_p]
        library.eclair_w4a16_linear.restype = ctypes.c_int
        library.eclair_w4a16_linear_image.argtypes = [ctypes.c_int]
        library.eclair_w4a16_linear_image.restype = ctypes.c_int
        library.w4a16_linear_ready = library.eclair_w4a16_linear_image(device_index) == 0
    if hasattr(library, "eclair_bf16_linear"):
        library.eclair_bf16_linear.argtypes = [ctypes.POINTER(_Bf16LinearArgs), ctypes.c_void_p]
        library.eclair_bf16_linear.restype = ctypes.c_int
        library.eclair_bf16_linear_image.argtypes = [ctypes.c_int]
        library.eclair_bf16_linear_image.restype = ctypes.c_int
        library.bf16_linear_ready = library.eclair_bf16_linear_image(device_index) == 0
    if hasattr(library, "eclair_gated_delta_output"):
        library.eclair_gated_delta_output.argtypes = [ctypes.POINTER(_GatedDeltaOutputArgs),
                                                      ctypes.c_void_p]
        library.eclair_gated_delta_output.restype = ctypes.c_int
        library.eclair_gated_delta_output_image.argtypes = [ctypes.c_int]
        library.eclair_gated_delta_output_image.restype = ctypes.c_int
        # Numerical tests pass, but the model's thinking-quality gate has not.
        # Keep the established FLA output path until that separate gate passes.
        library.gated_delta_output_ready = (
            os.environ.get("ECLAIR_NATIVE_GDN_OUTPUT") == "1"
            and library.eclair_gated_delta_output_image(device_index) == 0)
    if hasattr(library, "eclair_paged_decode_int8"):
        library.eclair_paged_decode_int8.argtypes = [ctypes.POINTER(_DecodeArgs), ctypes.c_void_p]
        library.eclair_paged_decode_int8.restype = ctypes.c_int
        library.eclair_paged_decode_image.argtypes = [ctypes.c_int]
        library.eclair_paged_decode_image.restype = ctypes.c_int
        library.eclair_paged_decode_heads_supported.argtypes = [ctypes.c_uint64] * 4
        library.eclair_paged_decode_heads_supported.restype = ctypes.c_int
        library.eclair_paged_decode_workspace_elements.argtypes = [ctypes.c_uint64] * 4
        library.eclair_paged_decode_workspace_elements.restype = ctypes.c_uint64
        library.decode_ready = library.eclair_paged_decode_image(device_index) == 0
    return library


def current_library() -> ctypes.CDLL | None:
    """The current CUDA device's library, loaded once per worker process."""
    device = torch.cuda.current_device()
    if device not in _LIBRARIES:
        _LIBRARIES[device] = load(device)
    return _LIBRARIES[device]


def heads_supported(library: ctypes.CDLL, heads: int, kv_heads: int, head_size: int) -> bool:
    """The library's own rule for which head configurations its kernel implements."""
    return library.eclair_paged_prefill_heads_supported(heads, kv_heads, head_size) == 1


def decode_supported(library: ctypes.CDLL, heads: int, kv_heads: int, head_size: int,
                     max_rows: int) -> bool:
    """Whether the native decode implements chunks of up to `max_rows` query rows here."""
    return (library.decode_ready
            and library.eclair_paged_decode_heads_supported(heads, kv_heads, head_size,
                                                            max_rows) == 1)


def readable(tensor: torch.Tensor) -> int:
    """Elements of `tensor`'s dtype from its first element to the end of its storage."""
    return tensor.untyped_storage().nbytes() // tensor.element_size() - tensor.storage_offset()


def fp16_linear(library: ctypes.CDLL, x: torch.Tensor, weights: torch.Tensor,
                 activation_shifts: torch.Tensor, weight_shifts: torch.Tensor) -> torch.Tensor:
    """FP16 linear through the zero-shift BF16 contract, followed by FP16 RNE."""
    if (not library.bf16_linear_ready or x.dim() != 2 or weights.dim() != 2
            or not 1 <= x.shape[0] <= 4 or x.shape[1] != weights.shape[1]
            or x.dtype != torch.float16 or weights.dtype != x.dtype or not x.is_cuda
            or any(t.device != x.device or not t.is_contiguous()
                   for t in (x, weights, activation_shifts, weight_shifts))
            or activation_shifts.dtype != torch.int32 or weight_shifts.dtype != torch.int32
            or activation_shifts.numel() < x.shape[0] or weight_shifts.numel() < weights.shape[0]):
        raise ValueError("unsupported native FP16 linear layout")
    # The caller supplies immutable zero shifts prepared before graph capture.
    m, k = x.shape
    n = weights.shape[0]
    sums = torch.empty((m, n), dtype=torch.float32, device=x.device)
    args = _Bf16LinearArgs(1, x.data_ptr(), readable(x), activation_shifts.data_ptr(),
                           readable(activation_shifts), weights.data_ptr(), readable(weights),
                           weight_shifts.data_ptr(), readable(weight_shifts), None, 0,
                           sums.data_ptr(), readable(sums), m, n, k)
    status = library.eclair_bf16_linear(
        ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status:
        raise NativeKernelError(f"native FP16 linear enqueue failed: CUDA status {status}")
    return sums.to(torch.float16)


def w4a16_linear(library: ctypes.CDLL, x: torch.Tensor, weights: torch.Tensor,
                  scales: torch.Tensor, splits: int) -> torch.Tensor:
    """Measured skinny Exllama packing, separate FP32 split scratch and FP16 output."""
    if (not library.w4a16_linear_ready or x.dim() != 2 or weights.dim() != 2
            or scales.dim() != 2 or not 2 <= x.shape[0] <= 4 or not x.is_cuda
            or x.dtype != torch.float16 or weights.dtype != torch.int32
            or scales.dtype != torch.float16 or x.shape[1] % 128
            or weights.shape != (x.shape[1] // 8, scales.shape[1])
            or scales.shape[0] != x.shape[1] // 128 or splits not in (4, 8, 16, 32)
            or any(t.device != x.device or not t.is_contiguous() for t in (x, weights, scales))):
        raise ValueError("unsupported native W4A16 layout")
    m, k = x.shape
    n = weights.shape[1]
    output = torch.empty((m, n), dtype=torch.float16, device=x.device)
    scratch = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    args = _W4A16LinearArgs(1, x.data_ptr(), readable(x), weights.data_ptr(), readable(weights),
                           scales.data_ptr(), readable(scales), output.data_ptr(), readable(output),
                           scratch.data_ptr(), readable(scratch), m, n, k, splits)
    status = library.eclair_w4a16_linear(
        ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status:
        raise NativeKernelError(f"native W4A16 enqueue failed: CUDA status {status}")
    return output


def w4a16_gemm_context(library: ctypes.CDLL) -> ctypes.c_void_p | None:
    """The current device's large-M W4A16 context, or None when it cannot be created."""
    device = torch.cuda.current_device()
    if device not in _GEMMS:
        context = ctypes.c_void_p()
        if library.eclair_w4a16_gemm_create(device, ctypes.byref(context)) != 0:
            return None
        _GEMMS[device] = context
    return _GEMMS[device]


def w4a16_gemm(library: ctypes.CDLL, context: ctypes.c_void_p, x: torch.Tensor,
               weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Large-M W4A16 on the measured Exllama packing: Eclair writes the binary16
    weights into a per-call [K,N] temporary, as the installed path does, then
    cuBLAS multiplies them with FP32 accumulation into a binary16 output."""
    if (not library.w4a16_gemm_ready or not context or x.dim() != 2 or weights.dim() != 2
            or scales.dim() != 2 or not x.is_cuda or x.dtype != torch.float16
            or weights.dtype != torch.int32 or scales.dtype != torch.float16
            or x.shape[1] % 128 or weights.shape != (x.shape[1] // 8, scales.shape[1])
            or scales.shape[0] != x.shape[1] // 128
            or any(t.device != x.device or not t.is_contiguous() for t in (x, weights, scales))):
        raise ValueError("unsupported native W4A16 GEMM layout")
    m, k = x.shape
    n = weights.shape[1]
    dequantized = torch.empty((k, n), dtype=torch.float16, device=x.device)
    output = torch.empty((m, n), dtype=torch.float16, device=x.device)
    workspace = torch.empty(GEMM_WORKSPACE_BYTES, dtype=torch.uint8, device=x.device)
    args = _W4A16GemmArgs(1, x.data_ptr(), readable(x), weights.data_ptr(), readable(weights),
                          scales.data_ptr(), readable(scales), dequantized.data_ptr(),
                          readable(dequantized), output.data_ptr(), readable(output),
                          workspace.data_ptr(), workspace.numel(), m, n, k)
    status = library.eclair_w4a16_gemm(
        context, ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status:
        raise NativeKernelError(f"native W4A16 GEMM enqueue failed: CUDA status {status}")
    return output


def prefill(library: ctypes.CDLL, query: torch.Tensor, key_cache: torch.Tensor,
            value_cache: torch.Tensor, key_scales: torch.Tensor, value_scales: torch.Tensor,
            blocks: torch.Tensor, context: int, scale: float, output: torch.Tensor) -> None:
    """Writes the [rows, heads, head_size] attention of `query` into `output`.

    `key_cache` and `value_cache` are vLLM's int8 [num_blocks, block_size, kv_heads,
    padded] views, `key_scales` and `value_scales` FP32 [num_blocks, block_size,
    kv_heads] views, and `blocks` the sequence's int32 block ids. The first
    head_size bytes of a cell are its values.
    """
    rows, heads, head_size = query.shape
    if (query.dtype != torch.float16 or output.dtype != torch.float16
            or output.shape != query.shape or key_cache.dtype != torch.int8
            or key_cache.stride() != value_cache.stride()
            or key_scales.stride() != value_scales.stride() or blocks.dtype != torch.int32
            or blocks.dim() != 1 or not blocks.is_contiguous()
            or query.stride()[1:] != (head_size, 1) or output.stride()[1:] != (head_size, 1)):
        raise NativeKernelError("tensor layout outside the native prefill contract")
    num_blocks, block_size, kv_heads, _ = key_cache.shape
    args = _Args(
        abi=PREFILL_ABI, query=query.data_ptr(), query_elements=readable(query),
        query_row_stride=query.stride(0), output=output.data_ptr(),
        output_elements=readable(output), output_row_stride=output.stride(0),
        keys=key_cache.data_ptr(), values=value_cache.data_ptr(),
        cache_bytes=min(readable(key_cache), readable(value_cache)),
        key_scales=key_scales.data_ptr(), value_scales=value_scales.data_ptr(),
        scale_elements=min(readable(key_scales), readable(value_scales)),
        block_table=blocks.data_ptr(), block_table_entries=blocks.numel(), blocks=num_blocks,
        block_size=block_size, kv_heads=kv_heads, head_size=head_size,
        block_stride=key_cache.stride(0), slot_stride=key_cache.stride(1),
        head_stride=key_cache.stride(2), scale_block_stride=key_scales.stride(0),
        scale_slot_stride=key_scales.stride(1), scale_head_stride=key_scales.stride(2),
        heads=heads, rows=rows, context=context, scale=scale)
    status = library.eclair_paged_prefill_int8(ctypes.byref(args),
                                               ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status != 0:
        raise NativeKernelError(f"native paged prefill returned CUDA error {status}")


def decode(library: ctypes.CDLL, query: torch.Tensor, key_cache: torch.Tensor,
           value_cache: torch.Tensor, key_scales: torch.Tensor, value_scales: torch.Tensor,
           block_table: torch.Tensor, query_start: torch.Tensor, seq_lens: torch.Tensor,
           max_rows: int, scale: float, output: torch.Tensor) -> None:
    """Writes the attention of every sequence's query rows into `output`.

    `query` and `output` are [rows, heads, head_size]; sequence n owns rows
    query_start[n] .. query_start[n + 1] - 1, at most `max_rows` of them, and has
    seq_lens[n] cached tokens with block ids in row n of `block_table`. Rows no
    sequence owns are not written. The caches are the views `prefill` takes. The
    metadata stays on the device, so a CUDA graph capture records the call. The
    FP32 split workspace comes from PyTorch's allocator on the current stream.
    """
    rows, heads, head_size = query.shape
    sequences = seq_lens.shape[0]
    if (query.dtype != torch.float16 or output.dtype != torch.float16
            or output.shape != query.shape or key_cache.dtype != torch.int8
            or key_cache.stride() != value_cache.stride()
            or key_scales.stride() != value_scales.stride()
            or query.stride()[1:] != (head_size, 1) or output.stride()[1:] != (head_size, 1)
            or block_table.dtype != torch.int32 or block_table.dim() != 2
            or block_table.stride(1) != 1 or block_table.shape[0] < sequences
            or query_start.dtype != torch.int32 or not query_start.is_contiguous()
            or query_start.numel() < sequences + 1 or seq_lens.dtype != torch.int32
            or not seq_lens.is_contiguous()):
        raise NativeKernelError("tensor layout outside the native decode contract")
    num_blocks, block_size, kv_heads, _ = key_cache.shape
    elements = library.eclair_paged_decode_workspace_elements(rows, heads, head_size,
                                                              DECODE_SPLITS)
    workspace = torch.empty(elements, dtype=torch.float32, device=query.device)
    args = _DecodeArgs(
        abi=DECODE_ABI, query=query.data_ptr(), query_elements=readable(query),
        query_row_stride=query.stride(0), output=output.data_ptr(),
        output_elements=readable(output), output_row_stride=output.stride(0),
        keys=key_cache.data_ptr(), values=value_cache.data_ptr(),
        cache_bytes=min(readable(key_cache), readable(value_cache)),
        key_scales=key_scales.data_ptr(), value_scales=value_scales.data_ptr(),
        scale_elements=min(readable(key_scales), readable(value_scales)),
        block_tables=block_table.data_ptr(), block_table_entries=readable(block_table),
        block_table_stride=block_table.stride(0), max_blocks_per_sequence=block_table.shape[1],
        query_starts=query_start.data_ptr(), seq_lens=seq_lens.data_ptr(),
        workspace=workspace.data_ptr(), workspace_elements=elements, blocks=num_blocks,
        block_size=block_size, kv_heads=kv_heads, head_size=head_size,
        block_stride=key_cache.stride(0), slot_stride=key_cache.stride(1),
        head_stride=key_cache.stride(2), scale_block_stride=key_scales.stride(0),
        scale_slot_stride=key_scales.stride(1), scale_head_stride=key_scales.stride(2),
        heads=heads, sequences=sequences, rows=rows, max_rows_per_sequence=max_rows,
        splits=DECODE_SPLITS, scale=scale)
    status = library.eclair_paged_decode_int8(
        ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status != 0:
        raise NativeKernelError(f"native paged decode returned CUDA error {status}")


def gated_delta_supported(k: torch.Tensor, v: torch.Tensor,
                          initial_state: torch.Tensor | None) -> bool:
    """Whether the native state pass implements a layer with these keys and values.

    FLA's w and u have the shapes and dtypes of k and v with the value heads.
    """
    return (k.dim() == 4 and v.dim() == 4 and k.shape[0] == 1 and k.dtype == torch.float16
            and v.dtype == torch.float16 and k.shape[-1] in (64, 128) and v.shape[-1] % 64 == 0
            and v.shape[-2] % k.shape[-2] == 0
            and (initial_state is None or initial_state.dtype in (torch.float16, torch.float32)))


def gated_delta_states(library: ctypes.CDLL, k: torch.Tensor, w: torch.Tensor,
                       u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor | None,
                       cu_seqlens: torch.Tensor, chunk_offsets: torch.Tensor,
                       chunks: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FLA's chunk_gated_delta_rule_fwd_h with a scalar gate, as (h, v_new, final_state).

    k is [1, tokens, key_heads, K], w [1, tokens, heads, K], u [1, tokens, heads, V],
    g FP32 [1, tokens, heads] cumulative within chunks, cu_seqlens int32 [N + 1] and
    chunk_offsets int32 with each sequence's first chunk. Inputs are made contiguous.
    """
    k, w, u, g = (t.contiguous() for t in (k, w, u, g))
    cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
    chunk_offsets = chunk_offsets.to(torch.int32).contiguous()
    _, tokens, key_heads, key_size = k.shape
    heads, value_size = u.shape[-2], u.shape[-1]
    sequences = cu_seqlens.numel() - 1
    if g.dtype != torch.float32 or chunk_offsets.numel() < sequences:
        raise NativeKernelError("gates must be FP32 and chunk offsets cover every sequence")
    h = k.new_empty(1, chunks, heads, value_size, key_size)
    v_new = torch.empty_like(u)
    final_state = k.new_empty(sequences, heads, value_size, key_size, dtype=torch.float32)
    initial_format, initial, initial_elements = 0, None, 0
    if initial_state is not None:
        initial_state = initial_state.contiguous()
        initial_format = 1 if initial_state.dtype == torch.float16 else 2
        initial, initial_elements = initial_state.data_ptr(), initial_state.numel()
    args = _GatedDeltaArgs(
        abi=GATED_DELTA_ABI, initial_format=initial_format,
        keys=k.data_ptr(), keys_elements=k.numel(), w=w.data_ptr(), w_elements=w.numel(),
        u=u.data_ptr(), u_elements=u.numel(), gates=g.data_ptr(), gates_elements=g.numel(),
        initial=initial, initial_elements=initial_elements,
        sequence_starts=cu_seqlens.data_ptr(), chunk_starts=chunk_offsets.data_ptr(),
        states=h.data_ptr(), states_elements=h.numel(), new_values=v_new.data_ptr(),
        new_values_elements=v_new.numel(), final_states=final_state.data_ptr(),
        final_states_elements=final_state.numel(), tokens=tokens, sequences=sequences,
        chunks=chunks, heads=heads, key_heads=key_heads, key_size=key_size,
        value_size=value_size)
    status = library.eclair_gated_delta_states(
        ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status != 0:
        raise NativeKernelError(f"native gated delta state pass returned CUDA error {status}")
    return h, v_new, final_state


def gated_delta_output(library: ctypes.CDLL, q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor, h: torch.Tensor, g: torch.Tensor,
                       cu_seqlens: torch.Tensor, chunk_indices: torch.Tensor,
                       scale: float, core_attn_out: torch.Tensor | None = None) -> torch.Tensor:
    """Scalar-gated 64-token output step, with binary16 correlations and states."""
    q, k, v, h, g = (t.contiguous() for t in (q, k, v, h, g))
    starts = cu_seqlens.to(torch.int32).contiguous()
    indices = chunk_indices.to(torch.int32).contiguous()
    if (q.shape != k.shape or q.dim() != 4 or q.shape[0] != 1
            or q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype
            or h.dtype != q.dtype or g.dtype != torch.float32
            or v.dim() != 4 or v.shape[:2] != q.shape[:2]
            or g.shape != v.shape[:3] or indices.dim() != 2 or indices.shape[1] != 2
            or starts.dim() != 1 or starts.numel() < 2):
        raise NativeKernelError("tensor layout outside the gated delta output contract")
    _, tokens, key_heads, key_size = q.shape
    heads, value_size = v.shape[-2:]
    chunks = indices.shape[0]
    if h.shape != (1, chunks, heads, value_size, key_size):
        raise NativeKernelError("state snapshots do not cover the output chunks")
    tensors = (q, k, v, h, g, starts, indices)
    if not all(t.is_cuda and t.device == q.device for t in tensors):
        raise NativeKernelError("output tensors must share one CUDA device")
    if core_attn_out is None:
        out = torch.empty_like(v)
    else:
        if (core_attn_out.dtype != v.dtype or core_attn_out.device != v.device
                or not core_attn_out.is_contiguous() or core_attn_out.numel() < v.numel()):
            raise NativeKernelError("output workspace must be contiguous binary16 with enough values")
        out = core_attn_out.view(-1)[:v.numel()].view(v.shape)
    args = _GatedDeltaOutputArgs(
        abi=1, query=q.data_ptr(), query_elements=q.numel(),
        keys=k.data_ptr(), keys_elements=k.numel(), values=v.data_ptr(), values_elements=v.numel(),
        states=h.data_ptr(), states_elements=h.numel(), gates=g.data_ptr(), gates_elements=g.numel(),
        sequence_starts=starts.data_ptr(), sequence_starts_elements=starts.numel(),
        chunk_indices=indices.data_ptr(), chunk_indices_elements=indices.numel(),
        output=out.data_ptr(), output_elements=out.numel(), tokens=tokens,
        sequences=starts.numel()-1, chunks=chunks, heads=heads, key_heads=key_heads,
        key_size=key_size, value_size=value_size, scale=scale)
    status = library.eclair_gated_delta_output(
        ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    if status != 0:
        raise NativeKernelError(f"native gated delta output returned CUDA error {status}")
    return out
