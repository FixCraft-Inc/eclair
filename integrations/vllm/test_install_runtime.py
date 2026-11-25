# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks: the precompiled archive holds exactly the built libraries, and torch stays pinned."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import install_runtime


class InstallRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.build = self.root / "cmake"
        self.build.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_archive_contains_each_library_under_vllm(self) -> None:
        for name in install_runtime.LIBRARIES:
            (self.build / f"{name}.abi3.so").write_bytes(name.encode())
        hashes = install_runtime.pack(self.build, self.root / "libs.zip")
        with zipfile.ZipFile(self.root / "libs.zip") as archive:
            self.assertEqual(sorted(archive.namelist()),
                             sorted(f"vllm/{n}.abi3.so" for n in install_runtime.LIBRARIES))
        self.assertEqual(len(hashes), len(install_runtime.LIBRARIES))

    def test_missing_or_ambiguous_library_is_refused(self) -> None:
        for name in install_runtime.LIBRARIES[1:]:
            (self.build / f"{name}.abi3.so").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "_C_stable_libtorch library, found 0"):
            install_runtime.pack(self.build, self.root / "a.zip")
        (self.build / "_C_stable_libtorch.abi3.so").write_bytes(b"x")
        (self.build / "_C_stable_libtorch.cpython-311.so").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "found 2"):
            install_runtime.pack(self.build, self.root / "b.zip")

    def test_constraints_pin_torch_triton_and_nvidia_wheels(self) -> None:
        listing = [{"name": "torch", "version": "2.13.0+cu126"}, {"name": "triton", "version": "3.7.1"},
                   {"name": "nvidia-cublas-cu12", "version": "12.6.4.1"}, {"name": "numpy", "version": "2"}]
        with patch.object(install_runtime, "run", return_value=json.dumps(listing)):
            pins = install_runtime.constraints("python", self.root / "c.txt")
        self.assertEqual(pins, ["torch==2.13.0+cu126", "triton==3.7.1", "nvidia-cublas-cu12==12.6.4.1"])
        listing[0]["version"] = "2.13.0"
        with patch.object(install_runtime, "run", return_value=json.dumps(listing)):
            with self.assertRaisesRegex(ValueError, "must already contain torch"):
                install_runtime.constraints("python", self.root / "d.txt")


if __name__ == "__main__":
    unittest.main()
