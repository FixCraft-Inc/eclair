#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Prepare the official vLLM 0.32.0 sdist for an sm_70-only CUDA 12.9 build.

The edits are build-configuration changes behind the new CMake option
VLLM_ECLAIR_SM70_ONLY. CUDA 12.9 still compiles sm_70, so the option adds 7.0
to the CUDA 12.8-12.9 architecture list and skips the external projects that
contain only sm_80+ kernels, together with the GCC 13 check that only those
projects need. Compiling for sm_70 says nothing about execution, which the
bring-up tests on the V100 check separately.

One Python edit lets vLLM import without the FlashAttention extensions, which
have no sm_70 kernels. vLLM already guards that import this way on ROCm. The
FlashAttention entry points then raise ImportError if called, so nothing can
silently use a kernel that is not there. Each edit must match exactly once in
its hash-pinned file, so a different vLLM release fails here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

SDIST_SHA256 = "d2c957d5eee3d0985012151315ddc62cbe0cd368fc96a5fc3510438d9dddf2ad"
CMAKELISTS_SHA256 = "22689f423fb8220c457ad579ee3806813c8950d39ac381487c1e41eb4bdb5937"
FA_UTILS = "vllm/v1/attention/backends/fa_utils.py"
FA_UTILS_SHA256 = "b18fc6eacf113c94adf76633fd17b6b7c745a9d66447ca5d3e39231b11cbb701"
TOP = "vllm-0.32.0"

# FA_EDITS and EDITS quote and modify excerpts from vLLM 0.32.0's
# vllm/v1/attention/backends/fa_utils.py and CMakeLists.txt, respectively.
# Copyright contributors to the vLLM project.
# The upstream excerpts are licensed under Apache-2.0 (see LICENSE.vllm).
FA_EDITS = (
    ('if current_platform.is_cuda():\n'
     '    from vllm._custom_ops import reshape_and_cache_flash\n'
     '    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]\n'
     '        compile_flash_attn_varlen_func_from_specs,\n'
     '        flash_attn_varlen_func,\n'
     '        get_scheduler_metadata,\n'
     '    )\n',
     'if current_platform.is_cuda():\n'
     '    from vllm._custom_ops import reshape_and_cache_flash\n'
     '    try:\n'
     '        from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]\n'
     '            compile_flash_attn_varlen_func_from_specs,\n'
     '            flash_attn_varlen_func,\n'
     '            get_scheduler_metadata,\n'
     '        )\n'
     '    except ImportError:\n'
     '        # Eclair sm_70 build: FlashAttention is not built. Calls fail loudly.\n'
     '        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]\n'
     '            raise ImportError("FlashAttention is not built for this sm_70 vLLM")\n'
     '\n'
     '        get_scheduler_metadata = flash_attn_varlen_func  # type: ignore[assignment]\n'
     '        compile_flash_attn_varlen_func_from_specs = None  # type: ignore[assignment]\n'),
)

EDITS = (
    ('  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")\n'
     'else()\n',
     '  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1")\n'
     '  if(VLLM_ECLAIR_SM70_ONLY)\n'
     '    # Eclair: CUDA 12.9 compiles sm_70. V100 tests, not this list, show it executes.\n'
     '    set(CUDA_SUPPORTED_ARCHS "7.0")\n'
     '  endif()\n'
     'else()\n'),
    ('if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")\n'
     '    include(cmake/external_projects/triton_kernels.cmake)\n',
     'if((VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP") AND NOT VLLM_ECLAIR_SM70_ONLY)\n'
     '    include(cmake/external_projects/triton_kernels.cmake)\n'),
    ('if (VLLM_GPU_LANG STREQUAL "CUDA"\n'
     '    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"\n',
     'if (VLLM_GPU_LANG STREQUAL "CUDA" AND NOT VLLM_ECLAIR_SM70_ONLY\n'
     '    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"\n'),
    ('if (VLLM_GPU_LANG STREQUAL "CUDA")\n'
     '    include(cmake/external_projects/deepgemm.cmake)\n',
     'if (VLLM_GPU_LANG STREQUAL "CUDA" AND NOT VLLM_ECLAIR_SM70_ONLY)\n'
     '    include(cmake/external_projects/deepgemm.cmake)\n'),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def apply_edits(text: str, edits: tuple = EDITS) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise ValueError(f"expected one match for an sm_70 build edit, found {count}")
        text = text.replace(old, new)
    return text


def prepare(sdist: Path, output: Path) -> dict:
    if sha256(sdist) != SDIST_SHA256:
        raise ValueError("sdist does not match the pinned vLLM 0.32.0 release")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tarfile.open(sdist, "r:gz") as archive:
        archive.extractall(output, filter="data")
    cmakelists = output / TOP / "CMakeLists.txt"
    if sha256(cmakelists) != CMAKELISTS_SHA256:
        raise ValueError("extracted CMakeLists.txt does not match the audited release")
    fa_utils = output / TOP / FA_UTILS
    if sha256(fa_utils) != FA_UTILS_SHA256:
        raise ValueError("extracted fa_utils.py does not match the audited release")
    cmakelists.write_text(apply_edits(cmakelists.read_text()))
    fa_utils.write_text(apply_edits(fa_utils.read_text(), FA_EDITS))
    record = {"schema": "eclair.vllm-source/2", "release": "v0.32.0",
              "commit": "5df63a7203cdb9328d824a27b682e77c7b9928be", "sdist_sha256": SDIST_SHA256,
              "cmakelists_original_sha256": CMAKELISTS_SHA256,
              "cmakelists_prepared_sha256": sha256(cmakelists),
              "fa_utils_original_sha256": FA_UTILS_SHA256,
              "fa_utils_prepared_sha256": sha256(fa_utils),
              "edits_sha256": hashlib.sha256(json.dumps([EDITS, FA_EDITS]).encode()).hexdigest(),
              "option": "VLLM_ECLAIR_SM70_ONLY"}
    (output / "eclair-source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.sdist, args.output), indent=2))
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"prepare failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
