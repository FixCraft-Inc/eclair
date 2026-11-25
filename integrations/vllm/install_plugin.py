#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Install the eclair-vllm plugin, with Eclair's native kernel library, into a vLLM venv.

The plugin source is copied to a fresh staging directory, the built
libeclair_kernels.so is added next to its modules, and the staging copy
is pip-installed with --no-deps --no-build-isolation --force-reinstall, so the
source tree never holds a binary. Nothing is downloaded: the plugin has no
dependencies and its build backend comes from the venv. The record names both
SHA-256 values. Without --library the plugin installs without the native
kernel and uses its PyTorch fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

LIBRARY = "libeclair_kernels.so"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(plugin: Path, library: Path | None, staging: Path) -> dict:
    """Copies the plugin without caches or stale binaries and adds the library."""
    target = staging / "plugin"
    shutil.copytree(plugin, target, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.so", "build", "*.egg-info"))
    record = {"plugin_files": sorted(str(p.relative_to(target)) for p in target.rglob("*")
                                     if p.is_file())}
    if library is not None:
        if not library.is_file():
            raise ValueError("native library is missing")
        shutil.copyfile(library, target / "eclair_vllm" / LIBRARY)
        record["library_sha256"] = digest(library)
    return record


def install(venv: Path, plugin: Path, library: Path | None, record_path: Path) -> dict:
    python = venv / "bin" / "python"
    if not python.is_file() or not (plugin / "pyproject.toml").is_file():
        raise ValueError("venv python or plugin pyproject.toml is missing")
    with tempfile.TemporaryDirectory(prefix="eclair-plugin-") as temporary:
        record = stage(plugin, library, Path(temporary))
        argv = [str(python), "-m", "pip", "--isolated", "install", "--no-input",
                "--disable-pip-version-check", "--no-deps", "--no-build-isolation",
                "--force-reinstall", str(Path(temporary) / "plugin")]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    record.update(schema="eclair.vllm-plugin-install/1", pip_exit=result.returncode,
                  pip_tail=result.stdout[-2000:] + result.stderr[-2000:])
    with record_path.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    if result.returncode != 0:
        raise RuntimeError("pip install of the plugin failed; see the record")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--library", type=Path, help="built libeclair_kernels.so")
    parser.add_argument("--record", type=Path, required=True, help="new JSON record file")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        record = install(args.venv, args.plugin, args.library, args.record)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"plugin install failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps({k: record[k] for k in ("pip_exit", "library_sha256") if k in record}))


if __name__ == "__main__":
    main()
