# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""sm_70 attention over vLLM's int8 per-token-head KV cache.

Triton 3.7.1 lowers `tl.dot` to FP32 FMA on sm_70, so vLLM's unified Triton
attention runs without tensor cores. This backend keeps vLLM's Triton backend
for metadata and cache writes, and as the fallback. For a chunk of at least
PREFILL_MIN_QUERY query tokens it reads the sequence's keys and values from the
same paged cache, dequantizes them exactly as the Triton kernel does (int8
times the FP32 scale in FP32, then rounded once to FP16), and runs PyTorch's
memory-efficient SDPA kernel, which uses Volta tensor cores, with the
lower-right causal mask of a chunk at the end of its sequence. Anything the
fast path does not cover goes to the unchanged Triton implementation.

Prefill chunks run Eclair's native Volta flash-attention kernel
(libeclair_kernels, see native.py), which reads the int8 cache
directly. Without that library, or for shapes it does not implement, the
fallback below runs instead.

A chunk whose whole context exceeds WINDOW tokens attends to its cached prefix
in WINDOW-token pieces with PyTorch's efficient-attention operator, which also
returns each row's log-sum-exp, and merges the pieces exactly. The chunk's own
keys form a square causal piece. The dequantized keys and values therefore fit
one FP16 workspace of WINDOW tokens at any context length. Layers run one after
another on the worker's stream, so they share it. It is allocated while the
model is built, so vLLM's memory profile counts it before sizing the KV cache.

Chunks shorter than PREFILL_MIN_QUERY, which are decode steps and speculative
verification rows, go to Eclair's native batched decode kernel (native.py),
which reads every K and V tile once for all of a sequence's query rows and
heads and computes on tensor cores. Without the library, or for more rows than
it holds, the split-context Triton kernel in `decode_sm70` runs instead.
"""

from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

from eclair_vllm import decode_sm70, native

PREFILL_MIN_QUERY = 16
WINDOW = 32768
# Keys and values are dequantized in slices of this many tokens to bound FP32 temporaries.
DEQUANT_SLICE = 8192


# (device index, tokens, heads, head size) -> FP16 K and V buffers, owned by the worker process.
_WORKSPACES: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def workspace(tokens: int, heads: int, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The shared FP16 [tokens, heads, head_size] K and V buffers of the current CUDA device."""
    key = (torch.cuda.current_device(), tokens, heads, head_size)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = tuple(torch.empty(tokens, heads, head_size, dtype=torch.float16,
                                             device=f"cuda:{key[0]}") for _ in range(2))
    return _WORKSPACES[key]


def dequantize(cache: torch.Tensor, scales: torch.Tensor, blocks: torch.Tensor, start: int,
               stop: int, head_size: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP16 [stop - start, heads, head_size] for tokens [start, stop) of a paged sequence.

    `cache` holds int8 values as [num_blocks, block_size, heads, padded] and `scales`
    the FP32 per-token-head scales as [num_blocks, block_size, heads]. `blocks` lists
    the sequence's block ids in order. The result is the leading rows of `out` when
    it is given.
    """
    block_size, heads = cache.shape[1], cache.shape[2]
    length = stop - start
    if out is None:
        out = torch.empty(length, heads, head_size, dtype=torch.float16, device=cache.device)
    elif out.dtype != torch.float16 or out.shape[0] < length or tuple(out.shape[1:]) != (heads, head_size):
        raise ValueError("dequantization workspace is too small or has the wrong layout")
    out = out[:length]
    per_slice = max(1, DEQUANT_SLICE // block_size)
    end_block = -(-stop // block_size)
    for first in range(start // block_size, end_block, per_slice):
        ids = blocks[first:min(end_block, first + per_slice)]
        offset = first * block_size
        low, high = max(start, offset) - offset, min(stop, offset + ids.numel() * block_size) - offset
        values = cache.index_select(0, ids).reshape(-1, heads, cache.shape[3])[low:high, :, :head_size]
        scale = scales.index_select(0, ids).reshape(-1, heads)[low:high]
        out[offset + low - start:offset + high - start] = (values.float()
                                                           * scale.unsqueeze(-1)).to(torch.float16)
    return out


def scale_views(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 [num_blocks, block_size, heads] views of the per-token-head K and V scales.

    `kv_cache` is the logical [num_blocks, heads, block_size, 2 * padded] int8 page
    array. Each (block, head, slot) holds [K | K scale | V | V scale], where each
    scale is one FP32 value in the last 4 bytes of its half. Physical strides are
    taken from the tensor, so HND and NHD layouts both work.
    """
    if kv_cache.element_size() != 1:
        raise ValueError("per-token-head scale views need a one-byte cache element")
    blocks, heads, block_size, content = kv_cache.shape
    padded = content // 2
    stride_block, stride_head, stride_slot, stride_content = kv_cache.stride()
    if stride_content != 1 or padded <= 4 or any(
            value % 4 for value in (stride_block, stride_head, stride_slot, kv_cache.storage_offset(),
                                    padded)):
        raise ValueError("unexpected per-token-head cache layout")
    words = torch.empty(0, dtype=torch.float32, device=kv_cache.device).set_(kv_cache.untyped_storage())
    base = kv_cache.storage_offset() // 4
    shape, strides = (blocks, block_size, heads), (stride_block // 4, stride_slot // 4, stride_head // 4)
    k_scale = torch.as_strided(words, shape, strides, base + (padded - 4) // 4)
    v_scale = torch.as_strided(words, shape, strides, base + (2 * padded - 4) // 4)
    return k_scale, v_scale


class EclairSm70AttentionImpl(TritonAttentionImpl):
    """vLLM's Triton attention with Eclair prefill and decode paths for the int8 cache."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._workspace = None
        self._native = None
        self._native_decode = None
        config = get_current_vllm_config()
        if (self._is_per_token_head_quant and config.model_config is not None
                and torch.cuda.is_available()):
            library = native.current_library()
            if library is not None and native.decode_supported(
                    library, self.num_heads, self.num_kv_heads, self.head_size, 1):
                self._native_decode = library
            if library is not None and native.heads_supported(
                    library, self.num_heads, self.num_kv_heads, self.head_size):
                self._native = library
            else:
                # Built during model construction, so the memory profile includes it. It holds
                # one prefix window or one scheduled chunk, and whole contexts up to WINDOW.
                tokens = max(WINDOW, config.scheduler_config.max_num_batched_tokens)
                self._workspace = workspace(min(tokens, config.model_config.max_model_len),
                                            self.num_kv_heads, self.head_size)

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        # vLLM 0.32 also fills every scale slot with 1.0 when it first creates these
        # views. In a hybrid model the same pages hold other layers' recurrent state,
        # so that fill corrupts the first request. Reads only cover written slots, so
        # the views are created without writing.
        if self._k_scale_cache is None:
            self._k_scale_cache, self._v_scale_cache = scale_views(kv_cache)

    def _fast_path(self, query: torch.Tensor, attn_metadata, output_scale, output_block_scale) -> bool:
        return (attn_metadata is not None and output_scale is None and output_block_scale is None
                and self._is_per_token_head_quant and query.dtype == torch.float16
                and attn_metadata.causal
                and (attn_metadata.max_query_len >= PREFILL_MIN_QUERY
                     or decode_sm70.supported(self.num_heads, self.num_kv_heads, self.head_size))
                and self.alibi_slopes is None and self.sinks is None
                and self.sliding_window in (None, (-1, -1)) and not self.logits_soft_cap
                and getattr(attn_metadata, "mm_prefix_range_tensor", None) is None
                and getattr(attn_metadata, "rswa_window", None) is None)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if not self._fast_path(query, attn_metadata, output_scale, output_block_scale):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                                   output_scale, output_block_scale)
        self._ensure_scale_caches(kv_cache)
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        if attn_metadata.max_query_len < PREFILL_MIN_QUERY:
            if self._native_decode is not None and native.decode_supported(
                    self._native_decode, self.num_heads, self.num_kv_heads, self.head_size,
                    attn_metadata.max_query_len):
                # All query rows, so every query_start_loc value, padded or not, is in range.
                rows = query.shape[0]
                native.decode(self._native_decode, query, key_cache, value_cache,
                              self._k_scale_cache, self._v_scale_cache, attn_metadata.block_table,
                              attn_metadata.query_start_loc, attn_metadata.seq_lens,
                              attn_metadata.max_query_len, self.scale,
                              output[:rows].view(rows, self.num_heads, self.head_size))
                return output
            tokens = attn_metadata.num_actual_tokens
            decode_sm70.decode(query[:tokens], key_cache, value_cache, self._k_scale_cache,
                               self._v_scale_cache, attn_metadata.block_table,
                               attn_metadata.query_start_loc, attn_metadata.seq_lens, self.scale,
                               output[:tokens].view(tokens, self.num_heads, self.head_size))
            return output
        starts = attn_metadata.query_start_loc.tolist()
        lengths = attn_metadata.seq_lens.tolist()
        block_size = key_cache.shape[1]
        for index, total in enumerate(lengths):
            first, last = starts[index], starts[index + 1]
            rows = last - first
            if rows == 0:
                continue
            blocks = attn_metadata.block_table[index, :(total + block_size - 1) // block_size]
            target = output[first:last].view(rows, self.num_heads, self.head_size)
            if self._native is not None and rows >= native.MIN_ROWS:
                chunk = query[first:last]
                # The kernel's tensor-core loads need 32-byte aligned query rows.
                if (chunk.stride()[1:] != (self.head_size, 1) or chunk.stride(0) % 16
                        or chunk.data_ptr() % 32):
                    chunk = chunk.contiguous()
                native.prefill(self._native, chunk, key_cache, value_cache,
                                       self._k_scale_cache, self._v_scale_cache, blocks,
                                       total - rows, self.scale, target)
            else:
                target.copy_(self._prefill(query[first:last], key_cache, value_cache, blocks,
                                           total - rows, rows))
        return output

    def _prefill(self, query, key_cache, value_cache, blocks, context: int, rows: int):
        """[rows, heads, dim] attention of a chunk that follows `context` cached tokens."""
        group = self.num_heads // self.num_kv_heads
        q = query.transpose(0, 1).reshape(self.num_kv_heads, group, rows, self.head_size)
        k_out, v_out = self._workspace or (None, None)

        def keys_values(start: int, stop: int):
            k = dequantize(key_cache, self._k_scale_cache, blocks, start, stop, self.head_size, k_out)
            v = dequantize(value_cache, self._v_scale_cache, blocks, start, stop, self.head_size, v_out)
            # The fused kernel needs equal head counts. Each KV head becomes a batch entry
            # broadcast across its query-head group with a zero stride, so nothing is copied.
            return tuple(t.transpose(0, 1).unsqueeze(1).expand(-1, group, -1, -1) for t in (k, v))

        total = context + rows
        if total <= WINDOW:
            k, v = keys_values(0, total)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attended = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_lower_right(rows, total), scale=self.scale)
        else:
            acc = running = None
            for start in (*range(0, context, WINDOW), context):
                own = start == context
                k, v = keys_values(start, total if own else min(context, start + WINDOW))
                part, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
                    q, k, v, None, True, 0.0, own, scale=self.scale)
                lse = lse[..., :rows]
                if acc is None:
                    acc, running = part.float(), lse
                    continue
                merged = torch.logaddexp(running, lse)
                acc = (acc * torch.exp(running - merged).unsqueeze(-1)
                       + part.float() * torch.exp(lse - merged).unsqueeze(-1))
                running = merged
            attended = acc.to(torch.float16)
        return attended.reshape(self.num_heads, rows, self.head_size).transpose(0, 1)


class EclairSm70AttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[EclairSm70AttentionImpl]:
        return EclairSm70AttentionImpl


def register() -> None:
    register_backend(AttentionBackendEnum.CUSTOM, f"{__name__}.EclairSm70AttentionBackend")
