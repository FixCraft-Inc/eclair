#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Operators-only W8A16 contract tests on the documented V100 environment."""

import argparse
import os
import unittest
from unittest.mock import patch

import torch


def assert_same_half(test: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require identical binary16 bits, comparing NaNs by class only."""
    test.assertEqual(actual.dtype, torch.float16)
    nan = torch.isnan(actual)
    test.assertTrue(bool(torch.equal(nan, torch.isnan(expected))))
    test.assertTrue(bool(torch.equal(actual[~nan].view(torch.int16),
                                     expected[~nan].view(torch.int16))))


class W8A16Tests(unittest.TestCase):
    def test_formats_shapes_streams_and_workspace(self) -> None:
        torch.manual_seed(71)
        stream = torch.cuda.Stream()
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for m, n, k in ((1, 19, 33), (4, 17, 31), (17, 19, 33), (32, 48, 64)):
                a = torch.randn(m, k).half()
                weights = (torch.randn(n, k) / 4).to(dtype)
                bias = torch.arange(n).float() / 16
                expected = (a.double() @ weights.double().t()).float() * 0.007 + bias
                with torch.cuda.stream(stream), torch.inference_mode():
                    da, dw = a.cuda(), weights.cuda()
                    scale = torch.tensor(0.007, device="cuda")
                    db = bias.cuda()
                    for budget in (0, 64 * 1024 * 1024):
                        actual = torch.ops.eclair.w8a16_mm(da, dw, scale, db, budget)
                        stream.synchronize()
                        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)

    def test_fp16_output_is_the_fp32_result_rounded_once(self) -> None:
        # The fused epilogue replaces mul_, add_ and .half(); results must not move.
        torch.manual_seed(91)
        stream = torch.cuda.Stream()
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for m, n, k in ((1, 19, 33), (4, 16, 64), (17, 19, 33), (32, 48, 64), (300, 1028, 96)):
                a = torch.randn(m, k).half()
                weights = (torch.randn(n, k) / 4).to(dtype)
                bias_cpu = torch.randn(n)
                with torch.cuda.stream(stream), torch.inference_mode():
                    da, dw, db = a.cuda(), weights.cuda(), bias_cpu.cuda()
                    # 5e4 drives some outputs past the binary16 range.
                    for scale_value in (0.007, 5.0e4):
                        scale = torch.tensor(scale_value, device="cuda")
                        for bias in (None, db):
                            for budget in (0, 64 * 1024 * 1024):
                                wide = torch.ops.eclair.w8a16_mm(da, dw, scale, bias, budget)
                                half = torch.ops.eclair.w8a16_mm(da, dw, scale, bias, budget,
                                                                 out_dtype=torch.float16)
                                stream.synchronize()
                                assert_same_half(self, half, wide.half())
                    explicit = torch.ops.eclair.w8a16_mm(da, dw, torch.tensor(0.007, device="cuda"),
                                                         db, out_dtype=torch.float32)
                    stream.synchronize()
                    self.assertEqual(explicit.dtype, torch.float32)
                # Independent FP64 reference for the FP16 result.
                with torch.inference_mode():
                    actual = torch.ops.eclair.w8a16_mm(
                        a.cuda(), weights.cuda(), torch.tensor(0.007, device="cuda"),
                        bias_cpu.cuda(), out_dtype=torch.float16)
                    torch.cuda.synchronize()
                expected = ((a.double() @ weights.double().t()) * 0.007 + bias_cpu.double()).half()
                torch.testing.assert_close(actual.cpu().float(), expected.float(),
                                           atol=2e-3, rtol=2e-3)

    def test_fp16_output_nonfinite_values_and_rejection(self) -> None:
        nan, inf = float("nan"), float("inf")
        for m in (2, 5):
            a = torch.ones(m, 8, device="cuda", dtype=torch.float16)
            a[0, 0] = inf
            a[1, 0] = nan
            w = torch.ones(4, 8, device="cuda").to(torch.float8_e4m3fn)
            bias = torch.tensor([0.0, -inf, 1.0, nan], device="cuda")
            for scale_value in (1.0, -0.0, nan):
                scale = torch.tensor(scale_value, device="cuda")
                for budget in (0, 64 * 1024 * 1024):
                    for b in (None, bias):
                        wide = torch.ops.eclair.w8a16_mm(a, w, scale, b, budget)
                        half = torch.ops.eclair.w8a16_mm(a, w, scale, b, budget,
                                                         out_dtype=torch.float16)
                        torch.cuda.synchronize()
                        assert_same_half(self, half, wide.half())
        with self.assertRaisesRegex(RuntimeError, "out_dtype must be FP32 or FP16"):
            torch.ops.eclair.w8a16_mm(a, w, torch.ones((), device="cuda"),
                                      out_dtype=torch.bfloat16)

    def test_all_representations(self) -> None:
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            w = torch.arange(256, dtype=torch.uint8).view(dtype).reshape(256, 1)
            for m in (1, 16):
                a = torch.ones(m, 1).half()
                expected = (a.double() @ w.double().t()).float()
                actual = torch.ops.eclair.w8a16_mm(a.cuda(), w.cuda(), torch.ones((), device="cuda"))
                torch.cuda.synchronize()
                torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4,
                                           equal_nan=True)

    def test_invalid_input_policy_and_determinism(self) -> None:
        a = torch.ones(16, 32, device="cuda", dtype=torch.float16)
        w = a.to(torch.float8_e4m3fn)
        scale = torch.ones((), device="cuda")
        cases = (
            ((a.float(), w, scale), "FP16 activations"),
            ((a, w.float(), scale), "E4M3FN or E5M2"),
            ((a, w, scale.expand(2).contiguous()), "scalar FP32"),
            ((a, w, scale, None, -1), "nonnegative"),
            ((a, w, scale, torch.ones(2, device="cuda")), "bias must be"),
            ((a[:, ::2], w, scale), "contiguous"),
            ((a, w[:, :16].contiguous(), scale), "A\\[M,K\\]"),
            ((a[:0], w, scale), "nonempty"),
        )
        for args, message in cases:
            with self.assertRaisesRegex(RuntimeError, message):
                torch.ops.eclair.w8a16_mm(*args)
        with patch.dict(os.environ, {"ECLAIR_DISABLE": "fp8.gemm"}):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                torch.ops.eclair.w8a16_mm(a, w, scale)
        with self.assertRaisesRegex(RuntimeError, "inference only"):
            torch.ops.eclair.w8a16_mm(a.requires_grad_(), w, scale)
        a = a.detach()
        original = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with self.assertRaisesRegex(RuntimeError, "deterministic algorithms"):
                torch.ops.eclair.w8a16_mm(a, w, scale)
        finally:
            torch.use_deterministic_algorithms(original)
        torch.cuda.synchronize()

    def test_no_aten_override(self) -> None:
        self.assertEqual(torch.ops.eclair.adapter_info(), "eclair.pytorch/3;operators-only")
        a = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.ones((), device="cuda")
        with self.assertRaisesRegex(RuntimeError, "only supported on CUDA devices"):
            torch._scaled_mm(a, a.t(), scale, scale, out_dtype=torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    if torch.__version__ != "2.7.1+cu126" or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("These integration tests require torch 2.7.1+cu126 and sm_70")
    torch.ops.load_library(args.library)
    unittest.main(argv=[__file__])
