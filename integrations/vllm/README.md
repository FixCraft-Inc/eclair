# vLLM on a V100

The experimental `eclair-vllm` plugin targets vLLM 0.32.0 built for
`sm_70`. It calls `libeclair_kernels.so` through Eclair's versioned C
interface. The framework owns the buffers and CUDA stream. The native
kernels include [paged int8 attention](../../docs/PAGED_ATTENTION.md),
[Gated DeltaNet](../../docs/GATED_DELTA.md) and
[int4-weight linear operations](../../docs/W4A16_LINEAR.md).

## Build and install

The build needs Linux, bubblewrap, a Debian 12 build root with GCC 12 and
development headers, CUDA 12.9, and CUTLASS 4.7.1 headers. The script
expects the build root at `/var/lib/eclair-build-envs/debian12` and CUDA at
`/usr/local/cuda-12.9`. Use a Python 3.11–3.14 installation that runs
inside that root and a build virtual environment with CMake 3.26.1 or newer,
Ninja and `torch==2.13.0+cu126`. Keep the interpreter, virtual environment,
CUTLASS checkout, source and build output under `/home`. Run the build as
an unprivileged user.

Prepare the official vLLM 0.32.0 source archive. The patcher checks its
SHA-256 and requires a new output directory. It carries excerpts from the
upstream CMake and Python files under the [Apache 2.0 license](LICENSE.vllm).

```sh
python3 integrations/vllm/prepare_source.py --sdist "$vllm_sdist" \
  --output "$prepared_dir"
source_dir="$prepared_dir/vllm-0.32.0"

sh integrations/vllm/build_sm70.sh configure \
  "$source_dir" "$build_venv" "$python_dir" "$cutlass_dir" "$output_dir" 8
sh integrations/vllm/build_sm70.sh build \
  "$source_dir" "$build_venv" "$python_dir" "$cutlass_dir" "$output_dir" 8
```

Use a separate runtime virtual environment with `torch==2.13.0+cu126`
already installed. That pin differs from the standalone PyTorch adapter.
The runtime installer fetches Python dependencies while installing the
locally built vLLM extensions. The plugin installer stages the native
library from the [V100 build](../../docs/DEBIAN_V100.md) and installs
without downloads.

```sh
python3 integrations/vllm/install_runtime.py --venv "$runtime_venv" \
  --source "$source_dir" --build "$output_dir/cmake" \
  --plugin integrations/vllm/plugin --record runtime-install
python3 integrations/vllm/install_plugin.py --venv "$runtime_venv" \
  --plugin integrations/vllm/plugin \
  --library build-v100/libeclair_kernels.so --record plugin-install.json
```

## Test and select features

The plugin uses vLLM's general-plugin entry point. Select
`--quantization eclair-ct-sm70` for supported compressed-tensors W4A16
layouts. Use `--attention-backend CUSTOM --kv-cache-dtype int8_per_token_head`
for Eclair paged attention. These paths remain tied to
the pinned vLLM release and require compatible model shapes and weights.

`ECLAIR_NATIVE_W4A16=1` enables the two-to-four-row captured-decode
path when its layout and device checks pass. The larger prefill path
has a separate opt-in, `ECLAIR_NATIVE_W4A16_PREFILL=1`. The native
Gated DeltaNet output pass is opt-in through `ECLAIR_NATIVE_GDN_OUTPUT=1`.
The default Gated DeltaNet output uses the framework implementation.

`serve_profile.py` writes local serving arguments and environment settings
to JSON. It binds vLLM to loopback and disables usage reporting and model
fetches. The generated profile expects a local model directory. The optional
control plugin requires an explicit plugin allowlist, loopback binding and
engine authentication.

Run the host-side installer and dispatch tests with:

```sh
python3 -m unittest discover -s integrations/vllm
```

Model and GPU probes require the pinned runtime and a compatible V100.
