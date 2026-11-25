# Numerical formats

This page says what each Eclair conversion does with every kind of input. The
code lives in `include/eclair/formats/`. Each format has a portable reference
written from its definition, plus integer-only bit functions marked
`ECLAIR_HOST_DEVICE` that host code and CUDA kernels compile from one source.
Host tests compare the bit functions with the reference, and GPU tests compare
kernel output with the host reference bit for bit.

## FP8 (OCP E4M3FN and E5M2)

| | E4M3FN | E5M2 |
| --- | --- | --- |
| Exponent and fraction bits, bias | 4 and 3, bias 7 | 5 and 2, bias 15 |
| Largest finite value | 448 (`0x7E`) | 57344 (`0x7B`) |
| Smallest normal, subnormal | 2^-6, 2^-9 | 2^-14, 2^-16 |
| Infinity | none | `0x7C`, `0xFC` |
| NaN | `0x7F`, `0xFF` | `0x7D`-`0x7F`, `0xFD`-`0xFF` |

Decoding is exact, because every FP8 value is also a binary16 value.
`fp8_to_fp16_bits` widens to binary16 and `decode_fp8` to float.

Encoding from float or binary16 (`encode_fp8`, `fp32_bits_to_fp8`) rounds to
the nearest value with ties to the even code. Below half the smallest subnormal
the result is a signed zero. Finite magnitudes above the largest finite value
saturate to it in both formats. Infinity saturates to ±448 in E4M3FN and stays
infinite in E5M2. NaN becomes `S.1111111` with the input's sign (`0x7F` or
`0xFF`).

E5M2 saturation differs from PyTorch. PyTorch 2.14.0's CPU `.to(torch.float8_e5m2)`
rounds magnitudes from 61440 up to infinity, where Eclair returns 57344. The two
agree on E4M3FN for the values checked (57344, 61440, 10^6 and infinity all give
`0x7E`). Compare encoders before mixing bytes produced by different libraries.

`encode_fp8` searches all finite codes for the nearest one, which makes it easy
to check by hand. Above the largest finite value it saturates before searching.
Without that step, float distances to the top E5M2 candidates round to the same
number from about 2^40 up, and the tie rule returned 49152 instead of 57344.
`tests/fp8_tests.cpp` pins that case.

## IEEE binary16

`encode_fp16` rounds to nearest even with gradual underflow, and overflow
becomes infinity from 65520 up. NaN payloads are not kept. The linear epilogue
in [FP8_LINEAR.md](FP8_LINEAR.md) uses the same rounding for its FP16 output.

## bfloat16

bfloat16 is the upper half of a binary32 value: one sign, eight exponent and
seven fraction bits. It has binary32's range, subnormals included, but only 8
significant bits.

Widening to FP32 (`bf16_to_fp32_bits`, `decode_bf16`) is exact. Narrowing
(`fp32_bits_to_bf16`, `encode_bf16`) rounds to nearest even. Overflow becomes
infinity from `0x7F7F8000`, the midpoint between the largest value and 2^128.
FP32 subnormals round to BF16 subnormals. NaN keeps its sign and becomes the
quiet NaN `0x7FC0` or `0xFFC0`. A NaN whose payload lies only in the dropped
half stays NaN and never becomes infinity.

### Scaled onto binary16

`bf16_to_fp16_scaled_bits(bits, shift)` returns binary16 nearest-even of the
BF16 value times 2^shift, with gradual underflow and overflow to infinity.
Infinity stays infinite, zero keeps its sign and NaN becomes `S.11111.1000000000`.
Because BF16 has 8 significant bits, every scaled value in [2^-17, 2^16) is
exact. `bf16_fp16_shift(largest)` picks the shift that puts a row's largest
finite magnitude in [2^15, 2^16), which keeps exact every element of that row
whose exponent is at most 32 below the largest. Shifts are clamped to +-160,
past which every finite nonzero value is 0 or infinity anyway. The BF16
linear operation is built on this. See [BF16_LINEAR.md](BF16_LINEAR.md).

## Where each conversion runs

| Conversion | CPU reference | NVIDIA GPU |
| --- | --- | --- |
| FP8 to binary16 | `decode_fp8` | `launch_fp8_to_fp16` |
| binary16 or FP32 to FP8 | `encode_fp8` | `launch_fp16_to_fp8`, `launch_fp32_to_fp8` |
| BF16 to and from FP32 | `decode_bf16`, `encode_bf16` | `launch_bf16_to_fp32`, `launch_fp32_to_bf16` |
| BF16 rows scaled onto binary16 | `host::bf16_row_scaling_reference` | `launch_bf16_row_scaling` |

The GPU launchers are in `include/eclair/cuda/fp8_convert.hpp` and
`bf16_convert.hpp`. They take device buffers and a stream that the caller owns.
Before enqueueing anything they reject null, overlapping, wrapping or oversized
views and unknown formats. They allocate and synchronize nothing. When both
pointers are aligned, each thread moves four elements per access, and
otherwise one at a time. The kernels run on any GPU that can load their images
and use no FP8 or BF16 hardware instructions, so `eclair-smi` reports them as
software emulation.

## Evidence

Host tests cover all 256 FP8 patterns of both formats, every binary16 pattern,
every bfloat16 pattern, the rounding midpoints of every FP8 value, a spread of
binary32 patterns and the cases above. The scaled conversion is checked for
every BF16 pattern at every shift from -170 to 170 against exact double
arithmetic. `tests/bf16_tests.cpp` checks every
exponent and kept fraction against an exact-distance oracle that does not use
the carry trick. The GPU tests run the same inputs through the vector and
scalar paths, with tails and misaligned views, and check for stray stores.

On 2026-10-24 the GPU tests passed bit-exact on the V100 (Tesla PG500-216,
`sm_70`, driver 580.196.07, CUDA 12.9.86, GCC 12.2 in the Debian 12 build
root), in Debug and Release builds. An RTX 3050 Ti laptop GPU (`sm_86`,
CUDA 13.4) passed the same tests as a generic check.

## Conversion bandwidth on the V100

`eclair_cuda_fp8_tests --benchmark`, Release, same device and source:
2^25 elements, aligned buffers (vector path), one nonblocking stream, a 100 ms
warmup per path, then nine CUDA-event samples. Bandwidth counts bytes read plus
bytes written at the median. The baseline is `cudaMemcpyAsync` device to
device moving the same total of read plus written bytes. FP8 input covers every
code, binary16 input is a hashed spread over all patterns, and FP32 input is
one repeated value (0x3F3F3F3F), so the FP32 rows are a best case for branch
uniformity. The run was not repeated and GPU clocks were left alone.

| Path | E4M3FN median (max) | E5M2 median (max) | E4M3FN, E5M2 GB/s |
| --- | --- | --- | --- |
| FP8 to binary16 | 166.9 (170.0) us | 121.9 (122.9) us | 603, 826 |
| binary16 to FP8 | 207.9 (210.9) us | 205.8 (206.8) us | 484, 489 |
| FP32 to FP8 | 222.2 (227.3) us | 224.3 (228.4) us | 755, 748 |
| Copy baseline, 192 MiB moved | 200.7 (201.7) us | | 1003 |

The E5M2 decode is a shift and runs at 82% of copy bandwidth. E4M3FN decode
rebiases and normalizes subnormals, and both encodes run the integer rounding
in `fp32_bits_to_fp8`, so they reach 48-75%. The benchmark runs only after
the full correctness suite passes in the same process, on separate
deterministic vectors. Correctness ran on separate deterministic vectors from the timed FP8 inputs.
BF16 ran the correctness tests only.
