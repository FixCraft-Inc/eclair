# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks of the thinking-marker processor, and a CUDA check where vLLM and Triton exist."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).parent / "plugin/eclair_vllm/thinking_markers.py"
START, END = 90, 91


def load_module():
    """The module itself, against the real V2 interface when vLLM is installed."""
    spec = importlib.util.spec_from_file_location("eclair_thinking_markers_test", PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        import vllm.v1.worker.gpu.sample.logits_processor.interface  # noqa: F401
        spec.loader.exec_module(module)
    except ImportError:
        name = "vllm.v1.worker.gpu.sample.logits_processor.interface"
        stub = ModuleType(name)
        stub.LogitsProcessor = type("LogitsProcessor", (), {})
        with patch.dict(sys.modules, {name: stub}):
            spec.loader.exec_module(module)
    return module


markers = load_module()


class HostTests(unittest.TestCase):
    def test_penalty_bounds(self) -> None:
        self.assertEqual(markers.penalty_value(1), 1.0)
        for value in (-0.1, 10.5, float("nan"), True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                markers.penalty_value(value)
        with patch.dict(os.environ, {markers.ENVIRONMENT: "2.5"}):
            self.assertEqual(markers.default_penalty(), 2.5)
        with patch.dict(os.environ, {markers.ENVIRONMENT: "x"}), self.assertRaises(ValueError):
            markers.default_penalty()
        with patch.dict(os.environ, {markers.ENVIRONMENT: ""}):
            self.assertEqual(markers.default_penalty(), 0.0)

    def test_marker_ids_keep_single_token_spellings(self) -> None:
        vocabulary = {" Wait": 1, "Wait": 2, " wait": 3, " Hmm": 4, " but": 5, "But": 6,
                      "<think>": START, "</think>": END}
        encode = lambda text: [vocabulary[text]] if text in vocabulary else [7, 8]
        self.assertEqual(markers.marker_token_ids(encode, vocabulary.get), ([1, 2, 3, 4, 5, 6], START, END))
        with self.assertRaisesRegex(ValueError, "lacks"):
            markers.marker_token_ids(encode, lambda text: None)
        with self.assertRaisesRegex(ValueError, "lacks"):
            markers.marker_token_ids(lambda text: [7, 8], vocabulary.get)


def cuda_ready() -> bool:
    try:
        import torch
        import triton  # noqa: F401
        import vllm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


@unittest.skipUnless(cuda_ready(), "needs vLLM, Triton and a CUDA device")
class CudaTests(unittest.TestCase):
    def test_only_open_thinking_rows_are_penalized(self) -> None:
        import numpy as np
        import torch

        device = torch.device("cuda")
        length, vocabulary, marker_ids = 4096, 128, [3, 7, 11]
        # Slot 0: thinking opened in the prompt, still open. Slot 1: opened and closed in a long
        # committed history (cold scan over several blocks). Slot 2: open but no penalty requested.
        history = torch.zeros((3, length), dtype=torch.int32, device=device)
        history[0, :5] = torch.tensor([1, 2, START, 4, 5])
        history[1, 10], history[1, 3000] = START, END
        history[2, :2] = torch.tensor([START, 9])
        total = torch.tensor([5, 3001, 2], dtype=torch.int32, device=device)
        states = SimpleNamespace(device=device, max_num_reqs=3,
                                 all_token_ids=SimpleNamespace(gpu=history),
                                 total_len=SimpleNamespace(gpu=total))
        processor = markers.ThinkingMarkerPenalty.__new__(markers.ThinkingMarkerPenalty)
        processor.configure(states, marker_ids, START, END)
        request = lambda penalty: SimpleNamespace(extra_args={markers.ARGUMENT: penalty})
        self.assertTrue(processor.add_request(0, request(1.5)))
        self.assertTrue(processor.add_request(1, request(1.5)))
        self.assertFalse(processor.add_request(2, request(0)))
        processor.apply_staged_writes()
        # Slot 0 verifies two drafts: the second draft closes thinking, so its last row is free.
        # Slot 1 verifies one draft that reopens thinking.
        rows_to_slots = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int32, device=device)
        local = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.int32, device=device)
        inputs = torch.tensor([5, 6, END, 0, START, 9], dtype=torch.int32, device=device)
        context = SimpleNamespace(idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
                                  idx_mapping_np=np.array([0, 1, 2]), expanded_idx_mapping=rows_to_slots,
                                  expanded_local_pos=local, input_ids=inputs)
        logits = torch.zeros((6, vocabulary), dtype=torch.float32, device=device)
        result = processor.apply(logits, context).cpu()
        torch.cuda.synchronize()
        expected = torch.zeros((6, vocabulary))
        for row in (0, 1, 4):
            expected[row, marker_ids] = -1.5
        self.assertTrue(torch.equal(result, expected), result[:, marker_ids])
        self.assertEqual(processor.scan_pos.cpu().tolist(), [5, 3001, 0])
        # A recycled slot starts from a fresh scan.
        self.assertTrue(processor.add_request(1, request(2.0)))
        processor.apply_staged_writes()
        self.assertEqual(processor.scan_pos.cpu().tolist()[1], 0)


if __name__ == "__main__":
    unittest.main()
