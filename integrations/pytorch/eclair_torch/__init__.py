# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Explicit loader for the versioned Eclair native CUDA operator adapter."""

from pathlib import Path


def load(library: str | Path) -> None:
    """Register the native CUDA implementation before the application's calls.

    Registration lasts for this process. Restart without loading to restore
    PyTorch's native implementation. This does not bypass model-loader checks.
    """
    import torch

    if torch.__version__ != "2.7.1+cu126":
        raise RuntimeError("Eclair adapter requires torch 2.7.1+cu126")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Eclair adapter requires an available sm_70 GPU")
    artifact = Path(library).resolve(strict=True)
    if not artifact.is_file():
        raise ValueError("Eclair adapter library must be a regular file")
    torch.ops.load_library(str(artifact))
