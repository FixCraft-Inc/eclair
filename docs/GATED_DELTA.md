# Gated delta rule state and output passes

The chunk-to-chunk state pass of Gated DeltaNet prefill with one scalar gate
per head and token, as FLA's `chunk_gated_delta_rule_fwd_h` computes it for
Qwen3.5-family models. vLLM 0.32 runs it in Triton, which on `sm_70` lowers
`tl.dot` to FP32 FMA and walks each sequence one 64-token chunk at a time. It
took 30% of a 30K-token Qwen3.8 27B prefill on the V100.

| Layer | File | Role |
| --- | --- | --- |
| Contract | `include/eclair/ops/gated_delta.hpp`, `src/eclair/ops/gated_delta.cpp` | Layout, semantics, validation |
| Reference | `include/eclair/host/gated_delta.hpp`, `src/eclair/host/gated_delta.cpp` | FP64 sums with the contract's roundings, no allocation |
| Kernel | `include/eclair/cuda/gated_delta.hpp`, `src/eclair/cuda/gated_delta.cu` | Volta WMMA kernel and launcher |
| C interface | `include/eclair/c/gated_delta.h`, `src/eclair/c/gated_delta.cpp` | `libeclair_kernels.so`, for ctypes |
| Tests | `tests/gated_delta_tests.cpp`, `tests/cuda_gated_delta_tests.cu` | Host known answers, GPU equivalence and timing |
| Consumer | `integrations/vllm/plugin/eclair_vllm/gdn_sm70.py` and `native.py`, checked by `integrations/vllm/gdn_checks.py` | vLLM out-of-tree `chunk_gated_delta_rule` op |

## Semantics

Sequences are packed along the token axis (`sequence_starts`, `chunk_starts`).
Value head `h` reads key head `h / (heads / key_heads)`. For a chunk of tokens
`a .. b - 1` with FP32 state `S` (V x K, the initial state or zero):

```text
states[chunk, h] = binary16(S)
d_t              = u_t - w_t * binary16(S)^T          (FP32 sums)
new_values[t, h] = binary16(d_t)
e_t              = binary16(d_t * exp(g_{b-1} - g_t))
S                = S * exp(g_{b-1}) + sum_t e_t^T k_t   (FP32 sums)
```

`final_states[n, h]` is `S` after the sequence's last chunk. Gates are FP32 and
cumulative within each chunk. `k`, `w` and `u` are binary16. The initial state
may be binary16 (vLLM's default `mamba_ssm_cache_dtype`) or FP32. Every rounding
point matches FLA's kernel. Device backends skip a sequence whose bounds fall
outside the shape rather than read past it.

## Volta kernel

One thread block per sequence, head and 64 state rows, eight warps. The FP32
state stays in WMMA accumulators for the whole sequence: warp `w` holds 16
state rows and half of the key columns. Per chunk the block loads `w`, `k`,
`u` and the gates into shared memory with 16-byte accesses, stores the
binary16 state, computes `d = u - w S16^T` and the update `e^T k` on tensor
cores, and writes the new values with coalesced stores. Element positions of
accumulator fragments come from fragments loaded with row and column numbers.
Supported: key size 64 or 128 and value sizes in multiples of 64.

## Evidence, 2026-10-26

Tesla PG500-216, driver 580.196.07, CUDA 12.9.86, GCC 12.2, Release `sm_70`.
`ctest -R gated_delta` passes. Against the FP64 reference, states, new values
and final states agree within 6e-4 of the largest magnitude (tolerance 1e-2)
for one and four sequences, binary16, FP32 and zero initial states and key
sizes 64 and 128.

At Qwen3.8 27B's shapes (48 value heads, 16 key heads, size 128), the state
pass takes 1.19 ms for one 3072-token prefill chunk, against 30.5 ms for FLA's
Triton kernel in vLLM's profile, and 10.9 ms for a 30720-token sequence.

`integrations/vllm/gdn_checks.py` compares the whole prefill step (FLA's other
kernels plus the Eclair state pass) with vLLM's upstream path on the same
inputs. Output differences are at most 3e-5 and state differences at most
1.3e-4 of the largest magnitude, for 3072 tokens, five packed sequences and
30720 tokens. For a 3072-token chunk, the comparison took 34.8 ms with
upstream FLA and 7.0 ms with the Eclair state pass.

## Output pass

The independent output primitive is in `ops/gated_delta_output.hpp`,
`host/gated_delta_output.cpp` (FP64 oracle), `cuda/gated_delta_output.cu`
and `c/gated_delta_output.h` (ABI 1). It consumes the state snapshot and new
values from the state pass, binary16 queries/keys and cumulative FP32 gates.
Packed chunk indices name `(sequence, local chunk)` in sequence order. For
token i within a 64-token chunk:

```text
a_ij = binary16(sum_d q_i[d] * k_j[d] * exp(g_i - g_j)), j <= i
o_i  = binary16(scale * (exp(g_i) * q_i * state^T + sum_j a_ij * new_values_j))
```

This follows the public chunked algorithm and FLA's rounding boundaries;
no FLA implementation source is copied. One Volta block per chunk, value
head and 64 value columns uses four warps, each owning 16 token rows. WMMA
computes both state projections and correlations in FP32; gated causal
correlations round to binary16 before multiplying the new values. Partial
chunks load no token past their sequence's end. Invalid device index pairs
are skipped; metadata uniqueness/order is a caller precondition. The host
reference validates complete ordered coverage before writing.

The plugin detects the optional native symbol/image and requires
`ECLAIR_NATIVE_GDN_OUTPUT=1` before worker startup; FLA's output remains the
default pending the coding-quality gate. Device support is explicit: key size 64/128, values in
multiples of 64, aligned contiguous binary16 buffers, at most 65535 chunks.
The capability catalog reports `gated_delta.output` from its own image probe
and host known answer.

### Output evidence, 2026-10-27

Tesla PG500-216, driver 580.196.07, CUDA 12.9.86, GCC 12.2, Release `sm_70`.
The host and CUDA output tests pass, including packed lengths 1/63/0/65/130,
key sizes 64/128, grouped heads, partial chunks, nonfinite values, signed
zero, subnormals, rejected arguments and CUDA graph replay with changing
sequence bounds. Maximum FP64 reference error is 0.000244 (tolerance 0.004).
The output kernel takes 1.100 ms for 3072 tokens, 48 value heads, 16 key
heads and K/V 128, using CUDA events with end-event synchronization.

Same-process whole-operation comparison on identical seeded inputs, including
FLA's unchanged preparation steps (`gdn_checks.py --require-native-output --output gdn-results.json` in the
[pinned vLLM runtime](../integrations/vllm/README.md)):

| Tokens / packing | Upstream FLA | Native state + FLA output | Native state + output |
| --- | ---: | ---: | ---: |
| 3072 | 37.60 ms | 7.65 ms | 4.22 ms |
| 100/2000/64/7/901 | 39.81 ms | 7.16 ms | 3.78 ms |
| 30720 | 362.99 ms | 68.65 ms | 37.15 ms |

Maximum relative output error was 6.11e-5 against either baseline. The final
state error was 1.33e-4, within the 0.02 tolerance. The comparison used one
synchronized CUDA-event sample per path after warmup.

The corrected kernel applies scaling after the sum to preserve cancellation
when separately scaled terms would overflow. Host and V100 regression tests
cover that case and the intermediate binary16 correlation rounding. The
corrected kernel took 37.17 ms at 30720 tokens, with a 68.64 ms comparison
for native state plus FLA output. Native output is opt-in.
