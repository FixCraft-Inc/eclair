# eclair-smi

`eclair-smi` shows your GPU and the operations Eclair can run now. Use
`nvidia-smi` for temperatures, clocks, power and running processes.

```sh
eclair-smi                 # GPU and usable operations
eclair-smi --pretty        # the same words, with color on terminals
eclair-smi --details       # how each operation runs, IDs, unavailable operations, CPU checks
eclair-smi -i 0            # select a GPU; --id host selects CPU checks
eclair-smi -L              # one line per device
eclair-smi --catalog       # what each operation does, with its feature key
eclair-smi --json          # complete machine-readable report
ECLAIR_DISABLE=fp8.* eclair-smi --disable bf16.compat
```

On the tested V100 the default report reads:

```text
Eclair 0.1.0

NVIDIA GPU 0  Tesla PG500-216
  Volta sm_70, 72 compute units, 31.7 GiB
  CUDA driver API 13.0, runtime 12.9, kernels for sm 70
  FP16 tensor cores present, FP8 tensor cores absent

  Number formats                    Status  Runs as          Used by
  FP8 E4M3FN decode                 Ready   emulated on GPU
  FP8 E5M2 decode                   Ready   emulated on GPU
  FP8 E4M3FN encode                 Ready   emulated on GPU
  FP8 E5M2 encode                   Ready   emulated on GPU
  FP8-weight matrix multiplication  Ready   emulated on GPU  PyTorch, ComfyUI
  BF16 conversion                   Ready   emulated on GPU
  BF16 matrix multiplication        Ready   emulated on GPU

  Model kernels                     Status  Runs as          Used by
  Paged int8 attention, prefill     Ready   emulated on GPU  eclair-vllm
  Paged int8 attention, decode      Ready   emulated on GPU  eclair-vllm
  Gated DeltaNet state pass         Ready   native on GPU    eclair-vllm
  Gated DeltaNet output pass        Ready   native on GPU    eclair-vllm

  Other                             Status  Runs as          Used by
  Ray-tracing emulation             Ready   emulated on GPU  eclair-rt

CPU reference: 12 known-answer checks pass (--details lists them).

Ready: built in, and the GPU kernel loads (or the CPU check gives the known
answer). Speed, accuracy and model support come from tests, not this report.
--details: how each one runs   --catalog: what each one does   --json: scripts
```

Memory is CUDA's usable capacity. The CUDA line names the driver API and
runtime versions and the architectures the kernels were compiled for.
The output pass appears in the new build; its vLLM consumer remains opt-in
pending coding qualification. The production plugin uses FLA output.

Operations are grouped for reading. **Number formats** are the FP8 and BF16
conversions and matrix multiplications Volta lacks in hardware. **Model
kernels** are the attention and Gated DeltaNet kernels the vLLM plugin calls
([PAGED_ATTENTION.md](PAGED_ATTENTION.md), [GATED_DELTA.md](GATED_DELTA.md)).
"Runs as" says how a Ready operation executes: emulated when the device lacks
units for its number format (FP8, BF16, or the int8 cache without int8 tensor
cores), native when the device's units do the math, or the CPU reference.
"Used by" names the Eclair integrations that call the GPU kernel: the PyTorch
adapter and ComfyUI extension for FP8-weight linear layers, `eclair-vllm` for
the model kernels. `eclair-rt` consumes the bounded analytic ray-tracing kernel, validated on
sm_70 only ([rendering contract](REALTIME_RENDERING.md)). RT cores remain
absent on V100; a missing image reports unavailable. The CPU ray known-answer
probe is an emissive red pixel (linear 1 maps to RGBA8 red 232 after the
ACES fit and sRGB encoding). Material behavior is separately covered by host
known-angle/energy tests and CPU/CUDA frame comparisons. The CPU reference rows name none, because no integration
calls them. The groups and "Used by" are presentation, kept in the CLI.

## CPU, GPU and CUDA

**NVIDIA GPU** is the accelerator. **CUDA** is NVIDIA's programming runtime
used to access it through the installed driver; `cuda:0` is a technical device
ID, with zero identifying the first GPU.

**CPU reference** is portable C++ used to check numerical results. Its machine
ID is `host`. It does not accelerate a GPU workload. It appears as one line
when a GPU is available, and in full with `--details`, `--id host`, or when no
GPU is available.

Hardware lines describe dedicated units on the device. V100 has FP16 tensor
cores and no FP8 tensor cores. Eclair's software FP8 support does not change
those facts. An architecture-dependent fact that cannot be established is
`unknown`.

## Readiness and details

The default report lists ready and explicitly disabled operations. It does not
advertise future implementations. `--details` exposes every catalog entry and
its reason, along with the runtime/driver versions and device IDs.

| Display | JSON state | Meaning |
| --- | --- | --- |
| Ready | `enabled` | Implementation exists and its device probe passed |
| Disabled | `disabled` | A usable operation was turned off by policy |
| Not implemented | `not-implemented` | No implementation for this backend; path is `none` |
| Unavailable | `unsupported` | Device or build cannot provide the operation |
| Not applicable | `n/a` | This operation is not applicable to this device |

Execution paths are `emulated` (software on the GPU), `host-reference` (CPU),
`hardware` (the device's own units, as for the Gated DeltaNet state pass on FP16
tensor cores), or `none`. A path describes an existing implementation, never a
proposed one.

The CPU backend runs known-answer checks, values the contracts fix: for the
paged attention references a two-token cache whose tied scores must average
the values 2 and 6 to 4, and for the Gated DeltaNet reference one token whose
state goes from 2 to 4. CUDA checks loadable kernel images
with `cudaFuncGetAttributes`, initializing the primary device context. This
establishes image availability, not completed execution, numerical correctness,
speed or model compatibility. Numerical tests supply that evidence separately.
A build containing only `sm_89` SASS reports unavailable kernels on V100. The
tensor-core kernels (FP8 and BF16 linear, paged attention, the Gated DeltaNet
state/output passes) are validated on `sm_70` only, so other GPUs report them as not
implemented, and GPUs without FP16 tensor cores as unavailable. See
[FP8_LINEAR.md](FP8_LINEAR.md) and [PYTORCH.md](PYTORCH.md) for tested contracts.

A failed backend probe remains visible in the default report with its diagnostic.
A CPU-only build explains that GPU support was not included. `Ready` does not
mean the framework adapter is installed in another process or that a model
loader has selected it.

## Policy and execution

`--disable` and `ECLAIR_DISABLE` accept comma-separated feature keys. A trailing
`.*` selects a group (`fp8.*`, `fp8.e4m3fn.*`). Unknown keys fail with exit 2,
without partially applying policy. Policy only turns an enabled operation off.

`FeaturePolicy` is shared by the CLI and generic runtime selection. The native
PyTorch adapter also reads `ECLAIR_DISABLE` before executing. CLI switches apply
to that CLI process; they do not reconfigure an already-running model server.
Raw CUDA launch functions are explicit low-level APIs: their callers enforce
policy and memory-domain selection before launching.

## Presentation

Plain is the default. Pretty uses the same words and ASCII punctuation with
color; neither inserts custom status glyphs or depends on locale. A color
change never carries meaning without a status word.

Style precedence is command line (`--plain`, `--pretty`, `--style`), then
`ECLAIR_SMI_STYLE=plain|pretty`, then the CMake default
`ECLAIR_SMI_DEFAULT_STYLE` (plain). Invalid styles are usage errors.

Automatic color requires pretty mode and a terminal, and honors nonempty
`NO_COLOR` and `TERM=dumb`. `--color=always` forces color even in a pipe or plain
mode; `--color=never` and `--no-color` remove it. JSON never contains styling.

## JSON and compatibility

`--json` always returns the full report, independently of `--details`.
The schema is `eclair-smi.report/2`; the feature catalog remains
`eclair-smi.catalog/1`. Version 2 replaces the old future-status value with
`not-implemented`, clears nonexistent execution paths, and adds backend `label`
and `role` (`reference` or `accelerator`). IDs (`host`, `cuda`, `cuda:0`) and feature keys remain stable.
Missing numbers are `null`. On 2026-10-27 the catalog gained
`attention.paged_int8.prefill`, `attention.paged_int8.decode` and
`gated_delta.states`, followed by `gated_delta.output`. New keys do not change
the schema's meaning, so it stays version 2. `bf16.compat` kept its key when its title narrowed
to "BF16 conversion". It covers conversion to and from FP32 only, and BF16
arithmetic will get its own key.

The public interfaces remain experimental. Scripts should validate `schema`
and select fields by key. They must not parse display text or infer model
compatibility from `fp8.gemm` or `bf16.gemm`. A schema meaning change requires another version.

Exit status is 0 for a written report, including a backend reported unavailable;
1 for an internal/output failure; 2 for a usage or policy error.

## Maintenance

`capability/` owns states, catalog and policy; backends own probes and factual
support decisions; the CLI owns labels, filtering and presentation. See
`CONTRIBUTING.md`. Add a feature with a stable key, evaluation in every backend,
state tests and an update here. Do not encode device or framework policy in the
renderer.
