# Paged int8 attention

Causal attention over a paged int8 KV cache with one FP32 scale per token and
KV head, the layout of vLLM's `int8_per_token_head` cache, in two operations:
prefill of long chunks and batched decode of short ones (decode steps and
speculative verification rows). Eclair implements both for Volta tensor cores
because no FlashAttention release builds for `sm_70`, and vLLM's Triton
attention runs there without tensor cores.

| Layer | File | Role |
| --- | --- | --- |
| Contract | `include/eclair/ops/paged_attention.hpp`, `src/eclair/ops/paged_attention.cpp` | Layout, semantics, validation |
| Reference | `include/eclair/host/paged_attention.hpp`, `src/eclair/host/paged_attention.cpp` | FP64 oracle, no allocation |
| Kernel | `include/eclair/cuda/paged_attention.hpp`, `src/eclair/cuda/paged_attention.cu` (prefill), `src/eclair/cuda/paged_decode.cu` (decode) | Volta tensor-core kernels and launchers |
| C interface | `include/eclair/c/paged_attention.h`, `src/eclair/c/paged_attention.cpp` | `libeclair_kernels.so`, for ctypes |
| Tests | `tests/paged_attention_tests.cpp`, `tests/cuda_paged_attention_tests.cu`, `tests/cuda_paged_decode_tests.cu` | Host known answers, GPU equivalence, graph replay and timing |
| Consumer | `integrations/vllm/plugin/eclair_vllm/native.py`, `attention_sm70.py` | vLLM sm_70 attention backend |

## Semantics

A chunk of `rows` query tokens sits at positions `context .. context + rows - 1`
of a sequence whose first `context + rows` tokens are in the cache. Query head
`h` reads KV head `h / (heads / kv_heads)`. For row `i` and key `j <= context + i`:

```text
k_j   = binary16(fp32(int8 key * key scale))
v_j   = binary16(fp32(int8 value * value scale))
s_ij  = scale * sum_d q_i[d] * k_j[d]
out_i = binary16(sum_j softmax_j(s_ij) * v_j)
```

Keys and values have separate base pointers with shared byte strides per
block, slot and KV head. Scales are FP32 with shared element strides. vLLM's
cell is `[K(256) | K scale | V(256) | V scale]` (520 bytes), so the value base
is only 4-byte aligned. An FP16 cell of the same size needs 1,024 bytes
without scales. The int8 cell saves about 49% of the KV payload. Block
padding and kernel scratch can affect total allocator usage. Slots at or past `context + rows` are never read. A
backend may multiply a masked key's value by a zero weight, so a NaN value of a
key inside the chunk can also reach earlier rows of that chunk. vLLM's own
Triton kernel has the same property.

## Volta kernel

- One thread block per KV head and 16 query rows. Two warps per query head of
  the group, each owning the FP32 output of half the head size, so head size
  256 needs 64 accumulator registers and twelve warps fit an SM.
- Each 32-key tile is dequantized once into shared binary16 rows for the whole
  group. Warps own whole tokens and copy rows with 128-byte coalesced loads.
  Packed words are fetched one tile ahead and table/scale metadata two tiles
  ahead. The loader is specialized for each supported head group, so only the
  words that warp needs occupy registers. Words stay packed until the shared
  store, avoiding a dependency wait at byte extraction.
- Signed bytes become exact FP32 values by constructing `8388608 + (byte xor
  128)` and subtracting 8388736, avoiding Volta's quarter-rate conversion pipe.
  FP32 multiplication and final binary16 rounding remain unchanged. Prefill
  does **not** round the scale to binary16: tiny scales can still produce
  representable products, and a large finite scale times zero must stay zero.
- Q K^T is split over the head size between the warp pair and the halves are
  added through shared memory. The online softmax runs in FP32, eight rows per
  warp. Probabilities are rounded to binary16 for P V.
- Head size 256/group six uses direct `mma.m8n8k4` QK/PV fragments, 32 packed
  query words and 64 FP32 output accumulators per lane, and 61440 shared bytes.
  Its alternating sixteen-dimension QK partials and reciprocal multiplication
  match the qualified WMMA order in the tested fixtures. Other supported
  shapes retain WMMA, whose unspecified element-to-row mapping is read once
  from a fragment loaded with row numbers to apply per-row rescaling.
- The last row tile is shifted back to end at the chunk's last row, so every
  tile reads 16 real rows. Chunks shorter than 16 rows belong to the decode
  kernel (`integrations/vllm/plugin/eclair_vllm/decode_sm70.py`).
- Supported: head size 128 or 256, at most 8 query heads per KV head and no
  more than fit Volta's 96 KiB of shared memory (6 at 256). Anything else is
  refused with `cudaErrorInvalidValue`, and the vLLM backend falls back.

## Evidence, 2026-10-26

Tesla PG500-216 (V100 PCIe 32 GiB), driver 580.196.07, CUDA 12.9.86, GCC 12.2,
Release `sm_70`. `ctest -R paged_attention` passes: every case matches the
FP64 reference within 2e-3 (tolerance 8e-3) for head sizes 128 and 256, groups
of 1, 4, 6 and 8, blocks of 16 and 48, shifted tiles and padded query rows.
Unwritten slots holding NaN scales never reach the output, and rejected
configurations leave the output untouched.

Qwen3.8 27B shapes (24 query heads, 4 KV heads, head size 256, 1552-token
blocks), median of nine CUDA-event samples:

| Context | Rows | Time | TFLOP/s |
| ---: | ---: | ---: | ---: |
| 0 | 2048 | 3.0 ms | 17.3 |
| 8192 | 4096 | 49.9 ms | 20.7 |
| 32768 | 4096 | 173 ms | 20.3 |
| 196608 | 8192 | 1.95 s | 20.7 |

That is 1.25-1.4 times PyTorch's memory-efficient SDPA kernel on dequantized
copies at 8192-row chunks, and equal at the engine's 4096-row chunks, where the
plugin's windowed SDPA fallback also reaches about 20 TFLOP/s. The native path
needs no dequantization workspace.

Nsight Compute showed the remaining limits: shared-memory fragment traffic
(each K and V fragment is read once per query head), 2-way bank conflicts on
WMMA loads, and 12 warps per SM. Wider padding removes the conflicts but does
not fit six heads of size 256 in 96 KiB. A larger step needs register-resident
Q fragments, which head size 256 does not leave room for, or Volta's
`mma.sync.m8n8k4` with documented fragment layouts.

## Batched decode

Decode steps and speculative verification have a few query rows per sequence
and a long cached context, so the work is reading the cache, not arithmetic.
`launch_paged_decode` takes a batch of such chunks with the metadata an engine
keeps on the device: `query_starts` (rows of each sequence), `seq_lens`
(cached tokens including the chunk) and one block-table row per sequence. Each
chunk follows the prefill semantics with `context = seq_lens[n] - rows`. A
sequence whose metadata breaks the contract's rules (rows past the batch or
above `max_rows_per_sequence`, fewer tokens than rows, tokens past its table
row) is skipped on the device and refused by the host reference, which runs
the prefill reference per sequence. Rows no sequence owns are not written.

The contract lets a backend keep keys exact and scale the FP32 score instead,
and round the value scale to binary16 before the product. The Volta kernel
does both, because converting int8 through FP32 ran on Volta's quarter-rate
conversion pipe and was a third of its stall samples.

### Volta kernel

- Grid: context split, KV head, sequence. Each block holds all of a
  sequence's query rows times the KV head's query heads as rows of the
  tensor-core tiles: 8, 16 or 32 rows (one, two or three to five query rows
  at six heads per KV head), so every K and V tile is read once for all of
  them. 64 splits per context, chosen from a sweep of 64 and 128.
- Tiles of 32 keys. A tile's packed int8 words are loaded into registers one
  tile ahead, and the block-table and scale lookups two tiles ahead, so no warp
  waits for global memory before computing. Words stay packed until
  conversion: unpacking `char4` bytes at the load made every warp wait there.
- int8 becomes binary16 exactly with a byte permute and one half subtraction
  (`0x6400 | (b ^ 0x80)` is `1152 + b`). Values are then multiplied by the
  binary16 value scale. Key scales multiply the scores in the softmax.
- Q K^T and P V on WMMA, query fragments held in registers for the whole
  split, FP32 online softmax with binary16 probabilities, element-to-row
  positions read once from a fragment loaded with row numbers, as in prefill.
  The 16- and 32-row kernels run 16 warps per block, the 8-row kernel 8 warps
  with two blocks per SM.
- Each split writes its maximum, sum and unnormalized FP32 output per row and
  head to a caller-owned workspace. A merge kernel (one block per query head,
  row and sequence) combines the splits. A split whose maximum is -inf saw no
  key of that row and is skipped. The maximum keeps NaN, so a split whose
  visible scores are all NaN still makes the row NaN.
- Supported: head size 128 or 256, rows times group heads at most 32, at most
  1024 splits. Anything else returns `cudaErrorInvalidValue` and the vLLM
  backend uses its Triton decode kernel.

### Evidence, 2026-10-27

Tesla PG500-216, driver 580.196.07, CUDA 12.9.86, GCC 12.2, Release `sm_70`.
`ctest -R paged` passes (host reference, prefill and decode). The decode test
compares every output with the FP64 reference within 2e-3 (tolerance 8e-3)
for each row capacity, head sizes 128 and 256, one to three sequences, splits
that are empty, single or short, blocks of 16, 48 and 1552 and an unowned row.
It also checks that unwritten slots, the spare block and table entries past a
sequence's length never reach the output, that a NaN reaches only the rows of
its KV head that may see it, that skipped metadata leaves its rows untouched,
and that a graph captured in global capture mode replays correctly as
`seq_lens` changes. Rejected arguments leave the output untouched.

Qwen3.8 27B shapes, one sequence, 1552-token blocks, 64 splits, median of
nine CUDA-event samples, against the plugin's Triton decode kernel measured in
`attention_checks.py`:

| Context | Rows | Triton decode | Native decode | K/V read rate |
| ---: | ---: | ---: | ---: | ---: |
| 2000 | 3 | 0.11 ms | 0.074 ms | 57 GB/s |
| 33000 | 3 | 0.81 ms | 0.31 ms | 220 GB/s |
| 100000 | 3 | 2.26 ms | 0.80 ms | 262 GB/s |
| 211000 | 1 | 2.02 ms | 0.86 ms | 512 GB/s |
| 211000 | 2 | 3.37 ms | 1.49 ms | 294 GB/s |
| 211000 | 3 | 4.70 ms | 1.61 ms | 272 GB/s |

To test the same cache layout and compare with Triton, run
`integrations/vllm/attention_checks.py` in the [pinned vLLM runtime](../integrations/vllm/README.md).
Pass `--block 1552 --case 211000:3` and an `--output` path for the comparison
at 211,000 context tokens and three query rows.

### Register-fragment candidate

For head size 256 and 17–24 query-head rows, the register-fragment
decode path uses 24-row/64-key tiles. Wide K loads require eight-byte
alignment. Four-byte K and other row capacities use WMMA. V keeps its
four-byte alignment contract.

### Ordered prefill

The head-size-256/group-six prefill specialization preserves WMMA's
alternating 16-dimension QK partials and reciprocal multiplication. In
six seeded HND fixtures on V100, all binary16 outputs matched the previous
path bit for bit. These fixtures exercise the validated shape and data
layout. Other supported head groups use WMMA.
