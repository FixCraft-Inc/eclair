#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Install vLLM 0.32.0 with locally built sm_70 extensions into an isolated venv.

The owner-run installer packs the five libraries from build_sm70.sh into a
local archive, then lets vLLM's own setup.py install its official Python
sources in precompiled mode from that archive. Package downloads come from
PyPI and the PyTorch cu126 index. Every already-installed torch, Triton and
NVIDIA wheel is pinned as a constraint, so no dependency can replace the
sm_70-capable torch build. The venv's final package list is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

LIBRARIES = ("_C_stable_libtorch", "_moe_C_stable_libtorch", "cumem_allocator", "spinloop", "fs_io_C")
TORCH = "2.13.0+cu126"
BUILD_HELPERS = ("setuptools>=78.1.1,<81", "setuptools-scm>=8", "setuptools-rust>=1.9.0", "wheel",
                 "packaging>=24.2", "jinja2>=3.1.6", "regex")
EXTRA = ("torchvision==0.28.0+cu126",)
PYPI = "https://pypi.org/simple"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], *, env: dict[str, str] | None = None, timeout: int = 3600) -> str:
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-4000:]
        raise RuntimeError(f"command failed ({result.returncode}): {argv[:6]}\n{tail}")
    return result.stdout


def pack(build: Path, archive: Path) -> dict[str, str]:
    found = {}
    for name in LIBRARIES:
        matches = sorted(build.glob(f"{name}.*.so")) + sorted(build.glob(f"{name}.so"))
        if len(matches) != 1:
            raise ValueError(f"expected one built {name} library, found {len(matches)}")
        found[name] = matches[0]
    with zipfile.ZipFile(archive, "x") as output:
        for path in found.values():
            output.write(path, f"vllm/{path.name}")
    return {path.name: sha256(path) for path in found.values()}


def constraints(python: str, path: Path) -> list[str]:
    installed = json.loads(run([python, "-m", "pip", "list", "--format", "json"]))
    pinned = [f"{p['name']}=={p['version']}" for p in installed
              if p["name"] in ("torch", "triton") or p["name"].startswith(("nvidia-", "cuda-"))]
    if f"torch=={TORCH}" not in pinned:
        raise ValueError(f"the runtime venv must already contain torch {TORCH}")
    path.write_text("\n".join(pinned) + "\n")
    return pinned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="prepared vllm-0.32.0 directory")
    parser.add_argument("--build", type=Path, required=True, help="build_sm70.sh cmake directory")
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True, help="new directory for the record")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        python = str(args.venv / "bin" / "python")
        if not (args.source / "CMakeLists.txt").read_text().count("VLLM_ECLAIR_SM70_ONLY"):
            raise ValueError("source is not the prepared sm_70 tree")
        args.record.mkdir(mode=0o700)
        record: dict = {"schema": "eclair.vllm-install/1", "torch": TORCH}
        archive = args.record / "eclair-sm70-extensions.zip"
        record["libraries_sha256"] = pack(args.build, archive)
        record["archive_sha256"] = sha256(archive)
        pins = args.record / "constraints.txt"
        record["constraints"] = constraints(python, pins)
        pip = [python, "-m", "pip", "--isolated", "install", "--no-input",
               "--disable-pip-version-check", "--constraint", str(pins)]
        run(pip + ["--index-url", PYPI, *BUILD_HELPERS,
                   "-r", str(args.source / "requirements" / "common.txt")])
        run(pip + ["--index-url", TORCH_INDEX, "--extra-index-url", PYPI, *EXTRA])
        environment = {**os.environ, "VLLM_TARGET_DEVICE": "cuda", "VLLM_USE_PRECOMPILED": "1",
                       "VLLM_PRECOMPILED_WHEEL_LOCATION": str(archive),
                       "VLLM_VERSION_OVERRIDE": "0.32.0+eclair.sm70"}
        # Rebuilt Eclair artifacts keep their version, so replace them explicitly.
        run(pip + ["--no-build-isolation", "--no-deps", "--force-reinstall", "--index-url", PYPI,
                   str(args.source)], env=environment)
        run(pip + ["--no-build-isolation", "--no-deps", "--force-reinstall", "--index-url", PYPI,
                   str(args.plugin)])
        # vLLM's metadata still names CUDA 13 and sm_80+ packages that are deliberately absent.
        check = subprocess.run([python, "-m", "pip", "check"], capture_output=True, text=True,
                               timeout=300)
        record["pip_check"] = (check.stdout + check.stderr).splitlines()
        record["freeze"] = run([python, "-m", "pip", "freeze", "--all"]).splitlines()
        (args.record / "install.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "freeze"}, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"install failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
