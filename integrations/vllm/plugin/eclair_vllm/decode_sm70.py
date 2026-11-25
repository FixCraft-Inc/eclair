# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Split-context decode attention over vLLM's int8 per-token-head paged cache on sm_70.

On sm_70, Triton lowers `tl.dot` to FP32 FMA and pads the query rows of one KV
head to 16, so a 6-head group does 16/6 of the needed work and long-context
decode is compute-bound. Here each program owns one query token, one KV head and
one contiguous slice of that token's context. It loads every key and value tile
once and does exactly one FMA per query head and element: scores are the int8
key times the query in FP32 times the key's scale, and values are int8 times
their scale in FP32, as in vLLM's kernel. Each program keeps a running maximum,
sum and weighted value sum. A second kernel merges the slices.

Query token t of sequence b attends to the first seq_len[b] - (q_end[b] - 1 - t)
tokens, which is causal for one decode token and for speculative verification
rows alike. Sequence lookup happens in the kernel, so no host synchronization
or per-step tensors are needed and CUDA graph capture is unaffected.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Tuned on the V100 at 2K, 33K and 211K tokens: one warp per program, tiles of four tokens
# and 128 slices per token and KV head.
SPLITS = 128
TILE = 4
MAX_GROUP = 8


@triton.jit
def _head_step(k, v, k_scale, valid, q_h, m_h, l_h, acc_h):
    """One query head's online-softmax update with one tile of FP32 keys and scaled values."""
    score = tl.where(valid, tl.sum(k * q_h[None, :], axis=1) * k_scale, float("-inf"))
    new_max = tl.maximum(m_h, tl.max(score, axis=0))
    p = tl.exp(score - new_max)
    alpha = tl.exp(m_h - new_max)
    return new_max, l_h * alpha + tl.sum(p, axis=0), acc_h * alpha + tl.sum(p[:, None] * v, axis=0)


@triton.jit
def _store_head(part_max, part_sum, part_acc, index, dims, m_h, l_h, acc_h, HEAD: tl.constexpr):
    tl.store(part_max + index, m_h)
    tl.store(part_sum + index, l_h)
    tl.store(part_acc + index * HEAD + dims, acc_h)


@triton.jit
def _partials(query, keys, values, key_scales, value_scales, block_table, query_start, seq_lens,
              part_max, part_sum, part_acc, num_seqs, scale, block_size,
              q_token, q_head, c_block, c_slot, c_head, s_block, s_slot, s_head, t_row,
              GROUP: tl.constexpr, HEAD: tl.constexpr, TILE: tl.constexpr, SPLITS: tl.constexpr,
              NUM_HEADS: tl.constexpr):
    # Each of up to MAX_GROUP query heads keeps its own registers. Selecting rows of one
    # [group, head] tensor with masks every tile costs more than the attention arithmetic.
    token = tl.program_id(0)
    kv = tl.program_id(1)
    split = tl.program_id(2)
    seq = tl.full([], 0, tl.int32)
    for b in range(1, num_seqs + 1):
        seq += (tl.load(query_start + b) <= token).to(tl.int32)
    length = tl.load(seq_lens + seq) - (tl.load(query_start + seq + 1) - 1 - token)
    per_split = tl.cdiv(tl.cdiv(length, SPLITS), TILE) * TILE
    start = split * per_split
    stop = tl.minimum(length, start + per_split)

    dims = tl.arange(0, HEAD)
    q_base = query + token * q_token + kv * GROUP * q_head + dims
    q0 = tl.load(q_base).to(tl.float32) * scale
    q1 = tl.zeros([HEAD], tl.float32)
    if GROUP > 1:
        q1 = tl.load(q_base + 1 * q_head).to(tl.float32) * scale
    q2 = tl.zeros([HEAD], tl.float32)
    if GROUP > 2:
        q2 = tl.load(q_base + 2 * q_head).to(tl.float32) * scale
    q3 = tl.zeros([HEAD], tl.float32)
    if GROUP > 3:
        q3 = tl.load(q_base + 3 * q_head).to(tl.float32) * scale
    q4 = tl.zeros([HEAD], tl.float32)
    if GROUP > 4:
        q4 = tl.load(q_base + 4 * q_head).to(tl.float32) * scale
    q5 = tl.zeros([HEAD], tl.float32)
    if GROUP > 5:
        q5 = tl.load(q_base + 5 * q_head).to(tl.float32) * scale
    q6 = tl.zeros([HEAD], tl.float32)
    if GROUP > 6:
        q6 = tl.load(q_base + 6 * q_head).to(tl.float32) * scale
    q7 = tl.zeros([HEAD], tl.float32)
    if GROUP > 7:
        q7 = tl.load(q_base + 7 * q_head).to(tl.float32) * scale
    empty = tl.full([], float("-inf"), tl.float32)
    zero = tl.zeros([], tl.float32)
    acc = tl.zeros([HEAD], tl.float32)
    m0, l0, a0 = empty, zero, acc
    m1, l1, a1 = empty, zero, acc
    m2, l2, a2 = empty, zero, acc
    m3, l3, a3 = empty, zero, acc
    m4, l4, a4 = empty, zero, acc
    m5, l5, a5 = empty, zero, acc
    m6, l6, a6 = empty, zero, acc
    m7, l7, a7 = empty, zero, acc
    for first in range(start, stop, TILE):
        positions = first + tl.arange(0, TILE)
        valid = positions < stop
        block = tl.load(block_table + seq * t_row + positions // block_size, mask=valid, other=0)
        slot = positions % block_size
        cell = block.to(tl.int64) * c_block + slot * c_slot + kv * c_head
        k = tl.load(keys + cell[:, None] + dims[None, :], mask=valid[:, None], other=0).to(tl.float32)
        v = tl.load(values + cell[:, None] + dims[None, :], mask=valid[:, None], other=0).to(tl.float32)
        scale_cell = block.to(tl.int64) * s_block + slot * s_slot + kv * s_head
        k_scale = tl.load(key_scales + scale_cell, mask=valid, other=0.0)
        v = v * tl.load(value_scales + scale_cell, mask=valid, other=0.0)[:, None]
        m0, l0, a0 = _head_step(k, v, k_scale, valid, q0, m0, l0, a0)
        if GROUP > 1:
            m1, l1, a1 = _head_step(k, v, k_scale, valid, q1, m1, l1, a1)
        if GROUP > 2:
            m2, l2, a2 = _head_step(k, v, k_scale, valid, q2, m2, l2, a2)
        if GROUP > 3:
            m3, l3, a3 = _head_step(k, v, k_scale, valid, q3, m3, l3, a3)
        if GROUP > 4:
            m4, l4, a4 = _head_step(k, v, k_scale, valid, q4, m4, l4, a4)
        if GROUP > 5:
            m5, l5, a5 = _head_step(k, v, k_scale, valid, q5, m5, l5, a5)
        if GROUP > 6:
            m6, l6, a6 = _head_step(k, v, k_scale, valid, q6, m6, l6, a6)
        if GROUP > 7:
            m7, l7, a7 = _head_step(k, v, k_scale, valid, q7, m7, l7, a7)

    base = (token * NUM_HEADS + kv * GROUP) * SPLITS + split
    _store_head(part_max, part_sum, part_acc, base, dims, m0, l0, a0, HEAD)
    if GROUP > 1:
        _store_head(part_max, part_sum, part_acc, base + 1 * SPLITS, dims, m1, l1, a1, HEAD)
    if GROUP > 2:
        _store_head(part_max, part_sum, part_acc, base + 2 * SPLITS, dims, m2, l2, a2, HEAD)
    if GROUP > 3:
        _store_head(part_max, part_sum, part_acc, base + 3 * SPLITS, dims, m3, l3, a3, HEAD)
    if GROUP > 4:
        _store_head(part_max, part_sum, part_acc, base + 4 * SPLITS, dims, m4, l4, a4, HEAD)
    if GROUP > 5:
        _store_head(part_max, part_sum, part_acc, base + 5 * SPLITS, dims, m5, l5, a5, HEAD)
    if GROUP > 6:
        _store_head(part_max, part_sum, part_acc, base + 6 * SPLITS, dims, m6, l6, a6, HEAD)
    if GROUP > 7:
        _store_head(part_max, part_sum, part_acc, base + 7 * SPLITS, dims, m7, l7, a7, HEAD)


@triton.jit
def _merge(part_max, part_sum, part_acc, out, o_token, o_head,
           HEAD: tl.constexpr, SPLITS: tl.constexpr, NUM_HEADS: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, HEAD)
    base = (token * NUM_HEADS + head) * SPLITS
    maxima = tl.load(part_max + base + splits)
    # Empty slices carry -inf. A token with an empty context, which only occurs in
    # CUDA graph capture with placeholder lengths, yields zeros instead of NaN.
    weights = tl.where(maxima == float("-inf"), 0.0, tl.exp(maxima - tl.max(maxima, axis=0)))
    total = tl.sum(tl.load(part_sum + base + splits) * weights, axis=0)
    acc = tl.load(part_acc + (base + splits)[:, None] * HEAD + dims[None, :])
    result = tl.sum(acc * weights[:, None], axis=0) / tl.where(total > 0, total, 1.0)
    tl.store(out + token * o_token + head * o_head + dims, result.to(out.dtype.element_ty))


def supported(num_heads: int, num_kv_heads: int, head_size: int) -> bool:
    """Shapes the kernels are written for: power-of-two head size and whole GQA groups."""
    return (num_kv_heads > 0 and num_heads % num_kv_heads == 0
            and num_heads // num_kv_heads <= MAX_GROUP and head_size in (64, 128, 256))


def decode(query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
           key_scales: torch.Tensor, value_scales: torch.Tensor, block_table: torch.Tensor,
           query_start: torch.Tensor, seq_lens: torch.Tensor, scale: float,
           out: torch.Tensor) -> None:
    """Write [tokens, heads, head_size] attention of `query` over the cached context into `out`.

    `key_cache` and `value_cache` are int8 [num_blocks, block_size, kv_heads, padded]
    views whose first head_size elements per cell are the values. `key_scales` and
    `value_scales` are FP32 [num_blocks, block_size, kv_heads]. `query_start` holds
    num_seqs + 1 token offsets and `seq_lens` the context lengths including the queries.
    """
    tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[2]
    if (not supported(num_heads, num_kv_heads, head_size) or key_cache.dtype != torch.int8
            or key_cache.stride() != value_cache.stride() or key_cache.stride(3) != 1
            or query.stride(2) != 1 or out.shape != query.shape or out.stride(2) != 1
            or key_scales.stride() != value_scales.stride()):
        raise ValueError("unsupported decode attention layout")
    group = num_heads // num_kv_heads
    part_max = torch.empty(tokens, num_heads, SPLITS, dtype=torch.float32, device=query.device)
    part_sum = torch.empty_like(part_max)
    part_acc = torch.empty(tokens, num_heads, SPLITS, head_size, dtype=torch.float32,
                           device=query.device)
    _partials[(tokens, num_kv_heads, SPLITS)](
        query, key_cache, value_cache, key_scales, value_scales, block_table, query_start, seq_lens,
        part_max, part_sum, part_acc, seq_lens.shape[0], scale, key_cache.shape[1],
        query.stride(0), query.stride(1), key_cache.stride(0), key_cache.stride(1),
        key_cache.stride(2), key_scales.stride(0), key_scales.stride(1), key_scales.stride(2),
        block_table.stride(0),
        GROUP=group, HEAD=head_size, TILE=TILE, SPLITS=SPLITS, NUM_HEADS=num_heads, num_warps=1)
    _merge[(tokens, num_heads)](part_max, part_sum, part_acc, out, out.stride(0), out.stride(1),
                                HEAD=head_size, SPLITS=SPLITS, NUM_HEADS=num_heads, num_warps=4)
