# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks: serving profiles stay loopback, sm_70-safe and explicit about graphs and state reuse."""

import json
from pathlib import Path
import tempfile
import unittest

import serve_profile


class ServeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.venv, self.model = root / "venv", root / "model"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "bin" / "vllm").write_text("")
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def value(self, argv: list[str], flag: str) -> str:
        return argv[argv.index(flag) + 1]

    def test_default_profile_is_eager_loopback_int8_with_dense_state_checkpoints(self) -> None:
        argv = serve_profile.vllm_argv(self.venv, self.model)
        self.assertEqual(self.value(argv, "--host"), "127.0.0.1")
        self.assertEqual(self.value(argv, "--kv-cache-dtype"), "int8_per_token_head")
        self.assertEqual(self.value(argv, "--quantization"), "eclair-ct-sm70")
        self.assertEqual(self.value(argv, "--attention-backend"), "CUSTOM")
        self.assertEqual(self.value(argv, "--prefix-cache-retention-interval"), "None")
        self.assertIn("--enforce-eager", argv)
        self.assertIn("--language-model-only", argv)
        self.assertNotIn("--speculative-config", argv)

    def test_vision_loads_the_encoder_with_bounded_images_and_no_video(self) -> None:
        argv = serve_profile.vllm_argv(self.venv, self.model, vision=True)
        self.assertNotIn("--language-model-only", argv)
        self.assertEqual(json.loads(self.value(argv, "--limit-mm-per-prompt")), {"image": 4, "video": 0})
        size = json.loads(self.value(argv, "--mm-processor-kwargs"))["size"]
        self.assertLessEqual(size["longest_edge"], 1638400)
        self.assertEqual(self.value(argv, "--mm-encoder-attn-backend"), "TORCH_SDPA")

    def test_decode_graphs_skip_torch_compile_and_boundaries_keep_vllm_default(self) -> None:
        argv = serve_profile.vllm_argv(self.venv, self.model, graphs="decode", mtp=1,
                                       state_checkpoints="boundaries")
        self.assertNotIn("--enforce-eager", argv)
        self.assertIn("-cc.mode=none", argv)
        self.assertIn("-cc.cudagraph_mode=FULL_DECODE_ONLY", argv)
        self.assertNotIn("--prefix-cache-retention-interval", argv)
        self.assertEqual(self.value(argv, "--speculative-config"),
                         '{"method": "mtp", "num_speculative_tokens": 1}')

    def test_budget_phrase_ends_with_the_qwen3_parser_marker(self) -> None:
        self.assertNotIn("--reasoning-config", serve_profile.vllm_argv(self.venv, self.model))
        argv = serve_profile.vllm_argv(self.venv, self.model, budget_phrase=True)
        end = json.loads(self.value(argv, "--reasoning-config"))["reasoning_end_str"]
        self.assertTrue(end.startswith("\n\nConsidering the limited time"))
        self.assertTrue(end.endswith("directly now.\n</think>"))
        self.assertEqual(end.count("</think>"), 1)

    def test_fp16_cache_experiment_keeps_the_thinking_and_graph_settings(self) -> None:
        argv = serve_profile.vllm_argv(self.venv, self.model, kv_cache_dtype="float16",
                                       context=65536, graphs="decode", mtp=2, budget_phrase=True)
        self.assertEqual(self.value(argv, "--kv-cache-dtype"), "float16")
        self.assertEqual(self.value(argv, "--dtype"), "float16")
        self.assertEqual(self.value(argv, "--max-model-len"), "65536")
        self.assertIn("-cc.cudagraph_mode=FULL_DECODE_ONLY", argv)
        self.assertIn("--reasoning-config", argv)

    def test_unsupported_settings_are_refused(self) -> None:
        for settings in ({"graphs": "full"}, {"state_checkpoints": "every"},
                         {"attention": "FLASH_ATTN"}, {"context": 524288}, {"mtp": 5},
                         {"memory_fraction": 0.99}, {"kv_cache_dtype": "fp8"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                serve_profile.vllm_argv(self.venv, self.model, **settings)
        with self.assertRaisesRegex(ValueError, "missing"):
            serve_profile.vllm_argv(self.venv, self.model / "absent")

    def test_finer_matching_and_shared_checkpoints_are_explicit_and_keep_the_mtp_depth(self) -> None:
        argv = serve_profile.vllm_argv(self.venv, self.model, mtp=2, prefix_match_unit=32,
                                       shared_prefix_checkpoint=True)
        self.assertEqual(self.value(argv, "--prefix-match-unit"), "32")
        self.assertIn("--enable-mamba-shared-prefix-checkpoint", argv)
        self.assertEqual(json.loads(self.value(argv, "--speculative-config"))["num_speculative_tokens"], 2)
        for settings in ({"prefix_match_unit": True}, {"prefix_match_unit": 0},
                         {"prefix_match_unit": 8192}, {"shared_prefix_checkpoint": True},
                         {"prefix_match_unit": 32, "shared_prefix_checkpoint": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                serve_profile.vllm_argv(self.venv, self.model, **settings)


if __name__ == "__main__":
    unittest.main()
