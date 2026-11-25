# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Dispatch policy at the pinned framework seam, without CUDA or vLLM."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class UpstreamScheme:
    def process_weights_after_loading(self, layer):
        pass

    def apply_weights(self, layer, x, bias):
        return "Exllama"


class W4A16DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        package = ModuleType("eclair_w4_test")
        package.METHOD = "eclair-ct-sm70"
        package.native = SimpleNamespace(w4a16_linear=Mock(return_value="native"),
                                         w4a16_gemm=Mock(return_value="gemm"),
                                         w4a16_gemm_context=Mock(return_value="context"))
        modules = {}
        for name, fields in {
            "torch": {"float16": "half", "int32": "int32", "cuda": SimpleNamespace(
                is_current_stream_capturing=Mock(return_value=False))},
            "vllm.config": {"CUDAGraphMode": SimpleNamespace(FULL="full", FULL_DECODE_ONLY="decode"),
                            "get_current_vllm_config": Mock()},
            "vllm.forward_context": {"get_forward_context": Mock(),
                                     "is_forward_context_available": Mock(return_value=True)},
            "vllm.model_executor.layers.quantization": {"register_quantization_config": Mock()},
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": {
                "CompressedTensorsConfig": type("Config", (), {})},
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16": {
                "CompressedTensorsWNA16": UpstreamScheme},
            "vllm.platforms": {"current_platform": Mock()},
        }.items():
            modules[name] = ModuleType(name)
            modules[name].__dict__.update(fields)
        path = Path(__file__).parent / "plugin/eclair_vllm/wna16_sm70.py"
        spec = importlib.util.spec_from_file_location(package.__name__ + ".wna16_sm70", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {package.__name__: package, **modules}):
            spec.loader.exec_module(self.module)
        self.context = SimpleNamespace(cudagraph_runtime_mode="full",
                                       batch_descriptor=SimpleNamespace(uniform=True))
        self.module.get_forward_context.return_value = self.context
        self.torch = modules["torch"]
        self.native = package.native.w4a16_linear
        package.native.current_library = Mock(return_value=SimpleNamespace(w4a16_linear_ready=True))
        self.binding = package.native
        self.method = self.module.Sm70WNA16()
        self.method._native_library = object()
        self.method._prefill_gemm = None
        self.method._native_shape = (5120, 17408)
        self.layer = SimpleNamespace(weight_packed=SimpleNamespace(device="cuda:0"),
                                     weight_scale=object())

    def input(self, rows: int = 3, **kwargs):
        values = dict(shape=(rows, 17408), is_cuda=True, dtype="half", device="cuda:0",
                      dim=lambda: 2, is_contiguous=lambda: True)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_measured_rows_choose_the_recorded_split(self) -> None:
        for rows, splits in ((2, 8), (3, 16), (4, 8)):
            x = self.input(rows)
            self.assertEqual(self.method.apply_weights(self.layer, x, None), "native")
            self.native.assert_called_with(self.method._native_library, x,
                                           self.layer.weight_packed, self.layer.weight_scale, splits)

    def test_weight_admission_requires_packing_policy_and_image(self) -> None:
        self.method.kernel = type("ExllamaLinearKernel", (), {})()
        self.method.num_bits = 4
        self.method.group_size = 128
        self.method.symmetric = True
        compilation = SimpleNamespace(cudagraph_mode="decode")
        self.module.get_current_vllm_config.return_value = SimpleNamespace(
            compilation_config=compilation)
        self.layer.output_size_per_partition = 5120
        self.layer.input_size_per_partition = 17408
        self.layer.weight_packed = SimpleNamespace(
            shape=(2176, 5120), dtype="int32", is_cuda=True, device="cuda:0",
            is_contiguous=lambda: True)
        self.layer.weight_scale = SimpleNamespace(
            shape=(136, 5120), dtype="half", device="cuda:0", is_contiguous=lambda: True)
        with patch.dict(os.environ, {"ECLAIR_NATIVE_W4A16": "1"}):
            self.method.process_weights_after_loading(self.layer)
            self.assertIs(self.method._native_library, self.binding.current_library.return_value)
            for owner, field, value in (
                    (self.method, "symmetric", False),
                    (self.method, "group_size", 64),
                    (compilation, "cudagraph_mode", "full"),
                    (self.layer.weight_packed, "shape", (5120, 2176)),
                    (self.layer.weight_scale, "dtype", "float32"),
                    (self.binding.current_library.return_value, "w4a16_linear_ready", False)):
                with self.subTest(field=field), patch.object(owner, field, value):
                    self.method.process_weights_after_loading(self.layer)
                    self.assertIsNone(self.method._native_library)
        with patch.dict(os.environ, {"ECLAIR_NATIVE_W4A16": "0"}):
            self.method.process_weights_after_loading(self.layer)
            self.assertIsNone(self.method._native_library)

    def test_m1_prefill_bias_and_unsupported_layouts_keep_exllama(self) -> None:
        for x in (self.input(1), self.input(5), self.input(4096),
                  self.input(dtype="float32"), self.input(is_cuda=False),
                  self.input(is_contiguous=lambda: False), self.input(shape=(3, 5120))):
            self.assertEqual(self.method.apply_weights(self.layer, x, None), "Exllama")
        self.assertEqual(self.method.apply_weights(self.layer, self.input(), object()), "Exllama")
        self.context.batch_descriptor.uniform = False
        self.assertEqual(self.method.apply_weights(self.layer, self.input(), None), "Exllama")
        self.context.batch_descriptor.uniform = True
        self.context.cudagraph_runtime_mode = "none"
        self.assertEqual(self.method.apply_weights(self.layer, self.input(), None), "Exllama")
        self.native.assert_not_called()
        self.context.batch_descriptor = None
        self.assertEqual(self.method.apply_weights(self.layer, self.input(), None), "Exllama")
        self.torch.cuda.is_current_stream_capturing.return_value = True
        self.assertEqual(self.method.apply_weights(self.layer, self.input(), None), "native")

    def admit_layer(self) -> SimpleNamespace:
        """A measured down_proj layout whose library offers both native paths."""
        self.method.kernel = type("ExllamaLinearKernel", (), {})()
        self.method.num_bits = 4
        self.method.group_size = 128
        self.method.symmetric = True
        self.module.get_current_vllm_config.return_value = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode="decode"))
        self.layer.output_size_per_partition = 5120
        self.layer.input_size_per_partition = 17408
        self.layer.weight_packed = SimpleNamespace(
            shape=(2176, 5120), dtype="int32", is_cuda=True, device="cuda:0",
            is_contiguous=lambda: True)
        self.layer.weight_scale = SimpleNamespace(
            shape=(136, 5120), dtype="half", device="cuda:0", is_contiguous=lambda: True)
        library = SimpleNamespace(w4a16_linear_ready=True, w4a16_gemm_ready=True)
        self.binding.current_library.return_value = library
        return library

    def test_prefill_admission_is_separate_and_needs_image_and_context(self) -> None:
        library = self.admit_layer()
        with patch.dict(os.environ, {"ECLAIR_NATIVE_W4A16_PREFILL": "1",
                                     "ECLAIR_NATIVE_W4A16": "0"}):
            self.method.process_weights_after_loading(self.layer)
            self.assertEqual(self.method._prefill_gemm, (library, "context", (32, 64)))
            self.assertIsNone(self.method._native_library)
            for owner, field, value in ((library, "w4a16_gemm_ready", False),
                                        (self.method, "group_size", 64),
                                        (self.layer, "input_size_per_partition", 4096),
                                        (self.layer.weight_packed, "is_cuda", False)):
                with self.subTest(field=field), patch.object(owner, field, value):
                    self.method.process_weights_after_loading(self.layer)
                    self.assertIsNone(self.method._prefill_gemm)
            self.binding.w4a16_gemm_context.return_value = None
            self.method.process_weights_after_loading(self.layer)
            self.assertIsNone(self.method._prefill_gemm)
        with patch.dict(os.environ, {"ECLAIR_NATIVE_W4A16": "1"}):
            os.environ.pop("ECLAIR_NATIVE_W4A16_PREFILL", None)
            self.method.process_weights_after_loading(self.layer)
            self.assertIsNone(self.method._prefill_gemm)
            self.assertIs(self.method._native_library, library)

    def test_eager_calls_within_the_measured_rows_take_the_gemm(self) -> None:
        self.method._prefill_gemm = ("library", "context", (32, 64))
        for rows in (32, 50, 51, 64):
            x = self.input(rows)
            self.assertEqual(self.method.apply_weights(self.layer, x, None), "gemm")
            self.binding.w4a16_gemm.assert_called_with("library", "context", x,
                                                       self.layer.weight_packed,
                                                       self.layer.weight_scale)
        self.binding.w4a16_gemm.reset_mock()
        self.context.cudagraph_runtime_mode = "none"
        for x in (self.input(31), self.input(65), self.input(1914), self.input(5),
                  self.input(64, dtype="float32"), self.input(64, is_cuda=False),
                  self.input(64, device="cuda:1"), self.input(64, is_contiguous=lambda: False),
                  self.input(shape=(64, 5120)), self.input(64, dim=lambda: 3)):
            self.assertEqual(self.method.apply_weights(self.layer, x, None), "Exllama")
        self.assertEqual(self.method.apply_weights(self.layer, self.input(64), object()),
                         "Exllama")
        self.torch.cuda.is_current_stream_capturing.return_value = True
        self.assertEqual(self.method.apply_weights(self.layer, self.input(64), None), "Exllama")
        self.binding.w4a16_gemm.assert_not_called()
        self.method._prefill_gemm = None
        self.torch.cuda.is_current_stream_capturing.return_value = False
        self.assertEqual(self.method.apply_weights(self.layer, self.input(64), None), "Exllama")

    def test_every_routed_shape_is_a_measured_layout_with_a_valid_range(self) -> None:
        self.assertLessEqual(set(self.module.PREFILL_ROWS), set(self.module.NATIVE_SPLITS))
        for low, high in self.module.PREFILL_ROWS.values():
            self.assertTrue(4 < low <= high)


if __name__ == "__main__":
    unittest.main()
