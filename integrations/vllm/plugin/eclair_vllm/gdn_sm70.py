# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Gated DeltaNet prefill with Eclair's tensor-core state and output passes on sm_70.

vLLM 0.32 runs Qwen3.5-family Gated DeltaNet prefill through FLA's Triton
kernels. Their chunk-to-chunk state pass, chunk_gated_delta_rule_fwd_h, walks
each sequence one 64-token chunk at a time with small `tl.dot` products, which
Triton lowers to FP32 FMA on sm_70. It took 30% of a 30K-token prefill on the
V100. This registers vLLM's own out-of-tree replacement for the
`chunk_gated_delta_rule` custom op. Its forward runs FLA's other steps unchanged
and the state/output passes in libeclair_kernels (eclair/ops/gated_delta*.hpp), and
falls back to vLLM's path when the library is absent or the inputs are outside
what the native kernel implements. FLA's steps are imported by name from vLLM
0.32.0, the release the plugin is pinned to.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule
from vllm.model_executor.custom_op import op_registry_oot
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import (prepare_chunk_indices,
                                                               prepare_chunk_offsets)
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd

from eclair_vllm import native


def chunk_gated_delta_rule(library, q, k, v, g, beta, initial_state, output_final_state,
                           cu_seqlens, chunk_indices=None, chunk_offsets=None,
                           use_qk_l2norm_in_kernel=False, core_attn_out=None):
    """FLA's step order with Eclair's state pass and optional native output pass.

    Takes FLA's packed layouts (batch 1, cu_seqlens) and returns (o, final_state).
    Inputs are made contiguous, as FLA's input guard does.
    """
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)
    scale = k.shape[-1] ** -0.5
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, native.CHUNK)
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, native.CHUNK)
    g = chunk_local_cumsum(g, chunk_size=native.CHUNK, cu_seqlens=cu_seqlens,
                           chunk_indices=chunk_indices)
    a = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g, cu_seqlens=cu_seqlens,
                                 chunk_indices=chunk_indices, output_dtype=torch.float32)
    a = solve_tril(A=a, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a, g_cumsum=g, cu_seqlens=cu_seqlens,
                             chunk_indices=chunk_indices)
    h, v_new, final_state = native.gated_delta_states(
        library, k, w, u, g, initial_state, cu_seqlens, chunk_offsets,
        chunks=chunk_indices.shape[0])
    if library.gated_delta_output_ready and chunk_indices.shape[0] <= 65535:
        o = native.gated_delta_output(library, q, k, v_new, h, g, cu_seqlens,
                                      chunk_indices, scale, core_attn_out)
    else:
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens,
                        chunk_indices=chunk_indices, core_attn_out=core_attn_out)
    return o.to(q.dtype), (final_state if output_final_state else None)


class EclairChunkGatedDeltaRule(ChunkGatedDeltaRule):
    """vLLM's chunked gated delta rule op with Eclair's state pass on the Triton/FLA backend."""

    def forward_native(self, q, k, v, g, beta, initial_state, output_final_state,
                       cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                       use_qk_l2norm_in_kernel=True, core_attn_out=None,
                       aiter_prefill_metadata=None):
        library = native.current_library() if torch.cuda.is_available() else None
        if (library is None or cu_seqlens is None or q.shape[0] != 1
                or not native.gated_delta_supported(k, v, initial_state)):
            return super().forward_native(
                q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, chunk_indices,
                chunk_offsets, use_qk_l2norm_in_kernel, core_attn_out, aiter_prefill_metadata)
        return chunk_gated_delta_rule(library, q, k, v, g, beta, initial_state,
                                      output_final_state, cu_seqlens, chunk_indices,
                                      chunk_offsets, use_qk_l2norm_in_kernel, core_attn_out)


def register() -> None:
    if ChunkGatedDeltaRule.__name__ not in op_registry_oot:
        ChunkGatedDeltaRule.register_oot(EclairChunkGatedDeltaRule)
