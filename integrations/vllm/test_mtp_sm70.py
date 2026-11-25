# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks of draft-only policy, without loading vLLM or CUDA."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, call, patch


class DraftHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        package = ModuleType("eclair_mtp_test")
        head_method = type("NativeLmHeadMethod", (), {})
        fp16 = ModuleType(package.__name__ + ".fp16_sm70")
        fp16.NativeLmHeadMethod = head_method
        wna16 = ModuleType(package.__name__ + ".wna16_sm70")
        wna16.require_sm70 = Mock()
        vllm = ModuleType("vllm")
        vllm.ModelRegistry = Mock()
        model = ModuleType("vllm.model_executor.models.qwen3_5_mtp")
        model.Qwen3_5MTP = type("Qwen3_5MTP", (), {"compute_logits": Mock(return_value="full")})
        self.torch = ModuleType("torch")
        self.torch.nn = SimpleNamespace(functional=SimpleNamespace(pad=Mock(return_value=MagicMock())))
        spec = importlib.util.spec_from_file_location(
            package.__name__ + ".mtp_sm70",
            Path(__file__).parent / "plugin/eclair_vllm/mtp_sm70.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {package.__name__: package, fp16.__name__: fp16,
                                     wna16.__name__: wna16, "vllm": vllm,
                                     model.__name__: model, "torch": self.torch}):
            spec.loader.exec_module(self.module)
        self.method = head_method()
        self.method.apply_vocabulary_range = Mock(return_value="prefix")

    def config(self):
        return SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp", draft_sample_method="greedy",
                                               rejection_sample_method="standard"),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            lora_config=None,
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
                hidden_size=5120, vocab_size=248320, tie_word_embeddings=False)))

    def test_prefix_bounds_and_tile_alignment(self) -> None:
        for value in ("8192", "65536", "131072"):
            self.assertEqual(self.module.vocabulary_prefix(value), int(value))
        for value in ("0", "8191", "8193", "131328", "true", "-8192", "６５５３６"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.vocabulary_prefix(value)

    def test_sampling_and_shared_head_boundaries_fail_closed(self) -> None:
        self.module.validate_config(self.config())
        for owner, key, value in (("speculative_config", "draft_sample_method", "probabilistic"),
                                  ("speculative_config", "rejection_sample_method", "synthetic"),
                                  ("parallel_config", "tensor_parallel_size", 2)):
            config = self.config()
            setattr(getattr(config, owner), key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.validate_config(config)

    def test_draft_projection_preserves_global_ids_and_shared_head(self) -> None:
        draft = object.__new__(self.module.PrefixQwenMTP)
        target_head = SimpleNamespace(weight=object(), quant_method=self.method)
        original_weight = target_head.weight
        draft.lm_head = target_head
        draft.config = SimpleNamespace(vocab_size=248320)
        draft._draft_vocabulary = 65536
        self.assertIs(draft.compute_logits("hidden"), self.torch.nn.functional.pad.return_value)
        self.assertEqual(self.method.apply_vocabulary_range.call_args_list, [
            call(target_head, "hidden", 0, 65536), call(target_head, "hidden", 247808, 248320)])
        self.torch.nn.functional.pad.return_value.__getitem__.assert_called_once_with(
            (slice(None), slice(247808, None)))
        self.torch.nn.functional.pad.return_value.__getitem__.return_value.copy_.assert_called_once_with(
            "prefix")
        self.torch.nn.functional.pad.assert_called_once_with(
            "prefix", (0, 182784), value=float("-inf"))
        self.assertIs(draft.lm_head, target_head)
        self.assertIs(target_head.weight, original_weight)
        self.method.apply_vocabulary_range.return_value = None
        self.assertEqual(draft.compute_logits("hidden"), "full")


if __name__ == "__main__":
    unittest.main()
