# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks: the sm_70 build edits apply once, are guarded and refuse other inputs."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import prepare_source


def sample() -> str:
    return "".join(old for old, _ in prepare_source.EDITS)


class PrepareSourceTests(unittest.TestCase):
    def test_every_edit_is_behind_the_option(self) -> None:
        text = prepare_source.apply_edits(sample())
        for _, new in prepare_source.EDITS:
            self.assertIn("VLLM_ECLAIR_SM70_ONLY", new)
            self.assertIn(new, text)
        self.assertIn('set(CUDA_SUPPORTED_ARCHS "7.0")', text)

    def test_flash_attention_import_is_guarded_and_stubs_raise(self) -> None:
        original = prepare_source.FA_EDITS[0][0] + "\nelif current_platform.is_xpu():\n    pass\n"
        text = prepare_source.apply_edits(original, prepare_source.FA_EDITS)
        namespace: dict = {}
        source = ("from typing import Any\n"
                  "class current_platform:\n    is_cuda = staticmethod(lambda: True)\n"
                  "    is_xpu = staticmethod(lambda: False)\n"
                  "import sys, types\n"
                  "sys.modules['vllm._custom_ops'] = types.SimpleNamespace(reshape_and_cache_flash=None)\n")
        exec(compile(source + text.replace("from vllm._custom_ops import reshape_and_cache_flash",
                                           "reshape_and_cache_flash = None"), "fa_utils", "exec"), namespace)
        self.assertIsNone(namespace["compile_flash_attn_varlen_func_from_specs"])
        with self.assertRaisesRegex(ImportError, "not built"):
            namespace["flash_attn_varlen_func"]()
        with self.assertRaisesRegex(ImportError, "not built"):
            namespace["get_scheduler_metadata"]()

    def test_missing_or_repeated_context_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 0"):
            prepare_source.apply_edits(sample().replace("deepgemm", "other"))
        with self.assertRaisesRegex(ValueError, "found 2"):
            prepare_source.apply_edits(sample() + prepare_source.EDITS[0][0])

    def test_unpinned_sdist_is_refused_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "vllm.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                data = sample().encode()
                info = tarfile.TarInfo("vllm-0.32.0/CMakeLists.txt")
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
            with self.assertRaisesRegex(ValueError, "pinned"):
                prepare_source.prepare(archive, root / "out")
            self.assertFalse((root / "out").exists())
            # With the hash check satisfied, a CMakeLists.txt of another release is refused.
            with patch.object(prepare_source, "SDIST_SHA256", prepare_source.sha256(archive)):
                with self.assertRaisesRegex(ValueError, "audited release"):
                    prepare_source.prepare(archive, root / "out")


if __name__ == "__main__":
    unittest.main()
