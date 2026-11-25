#!/bin/sh
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.

# Build vLLM 0.32.0's CUDA extensions for sm_70 in the isolated Debian 12 root.
# The root, toolkit, interpreter, build venv and CUTLASS headers are read-only.
# Network and GPU devices are not exposed. Only the prepared source and a new
# output directory are writable.
set -eu

if [ "$#" -ne 7 ]; then
    echo 'Usage: build_sm70.sh configure|build SOURCE BUILD_VENV PYTHON_HOME CUTLASS OUTPUT JOBS' >&2
    exit 2
fi
mode=$1
shift
case "$mode" in
    configure|build) ;;
    *) echo 'Mode must be configure or build.' >&2; exit 2 ;;
esac
if [ "$(id -u)" -eq 0 ]; then
    echo 'Run this build as an unprivileged user.' >&2
    exit 2
fi
source_dir=$(realpath -- "$1")
venv=$(realpath -s -- "$2")
python_home=$(realpath -- "$3")
cutlass=$(realpath -- "$4")
jobs=$6
build_root=/var/lib/eclair-build-envs/debian12
cuda_root=/usr/local/cuda-12.9
case "$jobs" in
    ''|*[!0-9]*) echo 'JOBS must be a positive integer.' >&2; exit 2 ;;
esac
if [ "$jobs" -lt 1 ] || [ "$jobs" -gt 32 ]; then
    echo 'JOBS must be between 1 and 32.' >&2
    exit 2
fi
test -f "$source_dir/eclair-source.json" || test -f "$source_dir/../eclair-source.json"
grep -q 'VLLM_ECLAIR_SM70_ONLY' "$source_dir/CMakeLists.txt"
test -x "$venv/bin/python"
test -x "$venv/bin/cmake"
test -x "$venv/bin/ninja"
test -f "$cutlass/include/cutlass/cutlass.h"
test -x "$cuda_root/bin/nvcc"
test -d "$build_root/usr/include"
if [ "$mode" = configure ]; then
    mkdir -m 700 -- "$5"
else
    test -f "$5/cmake/build.ninja"
fi
output=$(realpath -- "$5")
for path in "$source_dir" "$venv" "$python_home" "$cutlass" "$output"; do
    case "$path" in
        /home/*) ;;
        *) echo "Build paths must be under /home: $path" >&2; exit 2 ;;
    esac
done

exec /usr/bin/bwrap \
    --unshare-user --unshare-pid --unshare-net --die-with-parent \
    --ro-bind "$build_root" / \
    --proc /proc --dev /dev \
    --tmpfs /tmp --tmpfs /home --tmpfs /usr/local \
    --ro-bind "$cuda_root" "$cuda_root" \
    --ro-bind "$python_home" "$python_home" \
    --ro-bind "$venv" "$venv" \
    --ro-bind "$cutlass" "$cutlass" \
    --bind "$source_dir" "$source_dir" \
    --bind "$output" "$output" \
    --chdir "$source_dir" \
    --clearenv \
    --setenv PATH "$venv/bin:$cuda_root/bin:/usr/bin:/bin" \
    --setenv LC_ALL C --setenv HOME /tmp \
    --setenv CUDA_HOME "$cuda_root" --setenv CUDACXX "$cuda_root/bin/nvcc" \
    --setenv TORCH_CUDA_ARCH_LIST 7.0 \
    --setenv VLLM_CUTLASS_SRC_DIR "$cutlass" \
    /bin/sh -ec '
        if [ "$5" = build ]; then
            exec cmake --build "$2/cmake" -j "$4" --target \
                _C_stable_libtorch _moe_C_stable_libtorch cumem_allocator spinloop fs_io_C
        fi
        python_path=$("$1/bin/python" -c "import sys; print(\":\".join(p for p in sys.path if p))")
        cmake -S . -B "$2/cmake" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DVLLM_TARGET_DEVICE=cuda \
            -DVLLM_ECLAIR_SM70_ONLY=ON \
            -DVLLM_PYTHON_EXECUTABLE="$1/bin/python" \
            -DVLLM_PYTHON_PATH="$python_path" \
            -DFETCHCONTENT_BASE_DIR="$2/deps" \
            -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
            -DCMAKE_CUDA_COMPILER="$3/bin/nvcc" \
            -DNVCC_THREADS=2 \
            -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
            -DCMAKE_JOB_POOLS:STRING=compile="$4"
    ' sh "$venv" "$output" "$cuda_root" "$jobs" "$mode"
