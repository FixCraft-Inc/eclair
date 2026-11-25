#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Build the opt-in native adapter using installed tools only; no downloads."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eclair-build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operators-only", action="store_true",
                        help="register only eclair operators, without overriding aten")
    args = parser.parse_args()
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    if torch.__version__ != "2.7.1+cu126":
        parser.error("this adapter requires torch 2.7.1+cu126")
    root = Path(__file__).resolve().parents[2]
    build = args.eclair_build.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("output already exists; select a new path")
    if not output.parent.is_dir():
        parser.error("output parent directory must exist")
    archives = [build / f"lib{name}.a" for name in (
        "eclair_cuda", "eclair_cuda_architecture", "eclair_ops", "eclair_core", "eclair_capability"
    )]
    for archive in archives:
        if not archive.is_file():
            parser.error(f"missing archive: {archive}")
    compiler = os.environ.get("CXX", "/usr/bin/g++")
    command = [compiler, "-std=c++20", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-I" + str(root / "include")]
    if args.operators_only:
        command.append("-DECLAIR_TORCH_ATEN_OVERRIDE=0")
    for path in include_paths(device_type="cuda"):
        command.extend(["-isystem", path])
    command.append(str(root / "integrations/pytorch/adapter.cpp"))
    command.extend(str(archive) for archive in archives)
    for path in library_paths(device_type="cuda"):
        command.extend(["-L" + path, "-Wl,-rpath," + path])
    command.extend(["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart",
                    "-lcublas"])
    with tempfile.TemporaryDirectory(prefix=".eclair-adapter-", dir=output.parent) as temporary:
        artifact = Path(temporary) / "adapter.so"
        subprocess.run(command + ["-o", str(artifact)], check=True, timeout=300)
        # Link rather than replace: never overwrite an existing output.
        os.link(artifact, output)
    print(f"Built {output} for torch {torch.__version__}; no packages downloaded")


if __name__ == "__main__":
    main()
