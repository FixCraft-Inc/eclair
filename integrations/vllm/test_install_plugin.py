# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks: the plugin installs from a staging copy that carries exactly the named library."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import install_plugin


class InstallPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.plugin = root / "plugin"
        (self.plugin / "eclair_vllm" / "__pycache__").mkdir(parents=True)
        (self.plugin / "pyproject.toml").write_text("[project]\n")
        (self.plugin / "eclair_vllm" / "__init__.py").write_text("")
        (self.plugin / "eclair_vllm" / "__pycache__" / "x.pyc").write_bytes(b"x")
        (self.plugin / "eclair_vllm" / "stale.so").write_bytes(b"old")
        self.venv = root / "venv"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "python").write_text("")
        self.library = root / install_plugin.LIBRARY
        self.library.write_bytes(b"native")
        self.record = root / "record.json"
        self.staged: list[str] = []

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fake_pip(self, argv, **kwargs):
        staged = Path(argv[-1])
        self.staged = sorted(str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file())
        self.assertIn("--no-deps", argv)
        self.assertIn("--no-build-isolation", argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    def test_staging_holds_sources_and_the_named_library_only(self) -> None:
        with patch("install_plugin.subprocess.run", side_effect=self.fake_pip):
            record = install_plugin.install(self.venv, self.plugin, self.library, self.record)
        self.assertEqual(self.staged, ["eclair_vllm/__init__.py",
                                       f"eclair_vllm/{install_plugin.LIBRARY}", "pyproject.toml"])
        self.assertEqual(record["library_sha256"], install_plugin.digest(self.library))
        self.assertEqual(json.loads(self.record.read_text())["pip_exit"], 0)
        self.assertFalse((self.plugin / "eclair_vllm" / install_plugin.LIBRARY).exists())

    def test_without_library_and_with_failed_pip(self) -> None:
        failed = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", "boom")
        with patch("install_plugin.subprocess.run", side_effect=failed), \
                self.assertRaisesRegex(RuntimeError, "pip install"):
            install_plugin.install(self.venv, self.plugin, None, self.record)
        self.assertNotIn("library_sha256", json.loads(self.record.read_text()))
        with self.assertRaisesRegex(ValueError, "missing"):
            install_plugin.install(self.venv, self.plugin, self.library.with_name("absent.so"),
                                   self.record.with_name("other.json"))


if __name__ == "__main__":
    unittest.main()
