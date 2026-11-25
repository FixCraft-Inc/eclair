# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks of native-kernel admission, without importing CUDA or vLLM."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fp16_checks import explained_half_tie


class Fp16RoundingTests(unittest.TestCase):
    def test_independently_rounded_fp64_values_explain_the_observed_tie(self) -> None:
        self.assertTrue(explained_half_tie([4.49609375, 4.49609375],
                                           [4.4948880920, 4.4945529393]))

    def test_an_unexplained_difference_or_nonfinite_value_fails(self) -> None:
        for native, exact in (([4.49609375, 4.4921875], [4.4948880920, 4.4945529393]),
                              ([4.4921875, 4.4921875], [4.4948880920, 4.4945529393]),
                              ([float("inf")] * 2, [float("inf")] * 2),
                              ([0.0, 0.0], [0.0, 1e10]),
                              ([0.0], [0.0])):
            with self.subTest(native=native, exact=exact):
                self.assertFalse(explained_half_tie(native, exact))


def load_binding() -> ModuleType:
    path = Path(__file__).parent / "plugin/eclair_vllm/native.py"
    spec = importlib.util.spec_from_file_location("eclair_native_loading_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Loading the binding only declares functions; tensor calls are not under test.
    with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
        spec.loader.exec_module(module)
    return module


class NativeLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.native = load_binding()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "libeclair_kernels.so"
        self.path.touch()
        self.library = SimpleNamespace(**{name: Mock(return_value=0) for name in (
            "eclair_paged_prefill_int8", "eclair_gated_delta_states",
            "eclair_paged_prefill_heads_supported", "eclair_paged_prefill_image",
            "eclair_gated_delta_image", "eclair_gated_delta_output",
            "eclair_gated_delta_output_image")})
        self.environment = {"ECLAIR_KERNELS_LIBRARY": str(self.path)}

    def admit(self, **environment: str) -> SimpleNamespace | None:
        with patch.dict(os.environ, {**self.environment, **environment}, clear=True), \
                patch.object(self.native.ctypes, "CDLL", return_value=self.library):
            return self.native.load(0)

    def test_output_stays_off_without_explicit_opt_in(self) -> None:
        for environment in ({}, {"ECLAIR_NATIVE_GDN_OUTPUT": "0"},
                            {"ECLAIR_NATIVE_GDN_OUTPUT": "true"}):
            with self.subTest(environment=environment):
                library = self.admit(**environment)
                self.assertIs(library, self.library)
                self.assertFalse(library.gated_delta_output_ready)
        self.library.eclair_gated_delta_output_image.assert_not_called()

    def test_output_opt_in_still_requires_a_loadable_image(self) -> None:
        self.assertTrue(self.admit(ECLAIR_NATIVE_GDN_OUTPUT="1").gated_delta_output_ready)
        self.library.eclair_gated_delta_output_image.return_value = 209
        library = self.admit(ECLAIR_NATIVE_GDN_OUTPUT="1")
        self.assertIs(library, self.library)
        self.assertFalse(library.gated_delta_output_ready)

    def test_old_library_keeps_state_and_prefill_available(self) -> None:
        del self.library.eclair_gated_delta_output
        del self.library.eclair_gated_delta_output_image
        library = self.admit(ECLAIR_NATIVE_GDN_OUTPUT="1")
        self.assertIs(library, self.library)
        self.assertFalse(library.gated_delta_output_ready)

    def test_global_disable_takes_precedence_over_output_opt_in(self) -> None:
        self.assertIsNone(self.admit(ECLAIR_NATIVE_KERNELS="0", ECLAIR_NATIVE_GDN_OUTPUT="1"))
        self.library.eclair_paged_prefill_image.assert_not_called()

    def test_w4a16_requires_opt_in_symbols_and_image(self) -> None:
        self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16="1").w4a16_linear_ready)
        self.library.eclair_w4a16_linear = Mock(return_value=0)
        self.library.eclair_w4a16_linear_image = Mock(return_value=0)
        for value in ("0", "true"):
            self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16=value).w4a16_linear_ready)
        self.library.eclair_w4a16_linear_image.assert_not_called()
        self.assertTrue(self.admit(ECLAIR_NATIVE_W4A16="1").w4a16_linear_ready)
        self.library.eclair_w4a16_linear_image.return_value = 209
        self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16="1").w4a16_linear_ready)
        self.assertIsNone(self.admit(ECLAIR_NATIVE_KERNELS="0", ECLAIR_NATIVE_W4A16="1"))

    def test_w4a16_gemm_requires_its_own_opt_in_symbols_and_image(self) -> None:
        self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16_PREFILL="1").w4a16_gemm_ready)
        self.library.eclair_w4a16_gemm = Mock(return_value=0)
        self.library.eclair_w4a16_gemm_image = Mock(return_value=0)
        # A library without the create symbol cannot own a context.
        self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16_PREFILL="1").w4a16_gemm_ready)
        self.library.eclair_w4a16_gemm_create = Mock(return_value=0)
        for value in (None, "0", "true"):
            environment = {} if value is None else {"ECLAIR_NATIVE_W4A16_PREFILL": value}
            self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16="1", **environment).w4a16_gemm_ready)
        self.library.eclair_w4a16_gemm_image.assert_not_called()
        self.assertTrue(self.admit(ECLAIR_NATIVE_W4A16_PREFILL="1").w4a16_gemm_ready)
        self.library.eclair_w4a16_gemm_image.return_value = 209
        self.assertFalse(self.admit(ECLAIR_NATIVE_W4A16_PREFILL="1").w4a16_gemm_ready)


class Fp16SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        package = ModuleType("eclair_fp16_selection_test")
        package.native = SimpleNamespace(current_library=Mock(return_value=None))
        embedding = ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")
        embedding.ParallelLMHead = type("ParallelLMHead", (), {})
        embedding.UnquantizedEmbeddingMethod = type("UnquantizedEmbeddingMethod", (), {})
        path = Path(__file__).parent / "plugin/eclair_vllm/fp16_sm70.py"
        spec = importlib.util.spec_from_file_location(package.__name__ + ".fp16_sm70", path)
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {package.__name__: package,
                                     embedding.__name__: embedding,
                                     "torch": ModuleType("torch")}):
            spec.loader.exec_module(self.module)
        self.head = embedding.ParallelLMHead()

    def select(self, layer, method, value: str | None):
        environment = {} if value is None else {"ECLAIR_NATIVE_FP16_LINEAR": value}
        with patch.dict(os.environ, environment, clear=True):
            return self.module.select_lm_head(layer, method)

    def test_fp16_default_keeps_upstream_selection(self) -> None:
        for value in (None, "0", "true"):
            with self.subTest(value=value):
                self.assertIsNone(self.select(self.head, None, value))

    def test_opt_in_selects_only_an_unquantized_lm_head(self) -> None:
        self.assertIsInstance(self.select(self.head, None, "1"), self.module.NativeLmHeadMethod)
        upstream_method = object()
        self.assertIs(self.select(self.head, upstream_method, "1"), upstream_method)
        self.assertIsNone(self.select(object(), None, "1"))


if __name__ == "__main__":
    unittest.main()
