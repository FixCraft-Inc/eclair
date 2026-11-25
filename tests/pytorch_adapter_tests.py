#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Opt-in CUDA integration tests; requires the documented isolated environment."""

import argparse
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/pytorch"))
from eclair_torch import load


class AdapterTests(unittest.TestCase):
    def test_scaled_mm_formats_streams_scales_and_outputs(self) -> None:
        torch.manual_seed(17)
        stream = torch.cuda.Stream()
        for dtype_a, dtype_b in ((torch.float8_e4m3fn, torch.float8_e4m3fn),
                                 (torch.float8_e5m2, torch.float8_e5m2),
                                 (torch.float8_e4m3fn, torch.float8_e5m2)):
            for m, n, k in ((1, 19, 33), (4, 17, 31), (17, 19, 33), (32, 48, 64)):
                a = (torch.randn(m, k) / 4).to(dtype_a)
                w = (torch.randn(n, k) / 4).to(dtype_b)
                expected = (a.double() @ w.double().t()).float() * 0.7 * 1.3
                bias = torch.arange(n, dtype=torch.float32) / 8
                expected += bias
                with torch.cuda.stream(stream), torch.inference_mode():
                    da, dw = a.cuda(), w.cuda()
                    sa = torch.tensor(0.7, device="cuda")
                    sb = torch.tensor([1.3], device="cuda")
                    for output_dtype in (torch.float32, torch.float16, torch.bfloat16):
                        actual = torch._scaled_mm(da, dw.t(), sa, sb, bias=bias.cuda(),
                                                  out_dtype=output_dtype)
                        stream.synchronize()
                        self.assertEqual(actual.dtype, output_dtype)
                        tolerance = 2e-4 if output_dtype == torch.float32 else (
                            2e-3 if output_dtype == torch.float16 else 2e-2)
                        torch.testing.assert_close(actual.cpu().float(), expected,
                                                   atol=tolerance, rtol=tolerance)

    def test_block_scaled_linear(self) -> None:
        torch.manual_seed(23)
        a = torch.randn(17, 33).half()
        weights = (torch.randn(19, 33) / 4).to(torch.float8_e4m3fn)
        scales = torch.tensor([[0.7, 1.3, -0.5], [0.25, 2.0, 1.0], [1.5, 0.9, 0.75]])
        expanded = weights.float()
        for n in range(19):
            for k in range(33):
                expanded[n, k] *= scales[n // 7, k // 13]
        expected = (a.double() @ expanded.half().double().t()).float()
        actual = torch.ops.eclair.fp8_linear(a.cuda(), weights.cuda(), scales.cuda(), 7, 13)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)

    def test_nonfinite_and_representations(self) -> None:
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            weights = torch.arange(256, dtype=torch.uint8).view(dtype).reshape(256, 1)
            for m in (1, 16):
                a = torch.ones(m, 1).to(dtype)
                expected = (a.double() @ weights.double().t()).float()
                scale = torch.ones((), device="cuda")
                actual = torch._scaled_mm(a.cuda(), weights.cuda().t(), scale, scale,
                                          out_dtype=torch.float32)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4,
                                           equal_nan=True)

    def test_fail_closed(self) -> None:
        a = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn)
        b = a.t()
        scale = torch.ones((), device="cuda")
        cases = (
            ({"use_fast_accum": True}, "fast accumulation"),
            ({"scale_result": scale}, "scale_result"),
            ({"out_dtype": None}, "explicit FP32"),
            ({"out_dtype": torch.float64}, "explicit FP32"),
            ({"bias": torch.ones(2, device="cuda")}, "bias must be"),
        )
        for changes, message in cases:
            options = {"out_dtype": torch.float32, **changes}
            with self.assertRaisesRegex(RuntimeError, message):
                torch._scaled_mm(a, b, scale, scale, **options)
        with self.assertRaisesRegex(RuntimeError, "scalar FP32 scales"):
            torch._scaled_mm(a, b, torch.ones(16, device="cuda"), scale,
                             out_dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch._scaled_mm(a, b.contiguous(), scale, scale, out_dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "inference only"):
            torch._scaled_mm(a, b, scale.requires_grad_(), scale, out_dtype=torch.float32)
        torch.cuda.synchronize()

    def test_policy(self) -> None:
        a = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.ones((), device="cuda")
        original = os.environ.get("ECLAIR_DISABLE")
        try:
            for value, message in (("fp8.gemm", "disabled"), ("unknown.feature", "policy")):
                os.environ["ECLAIR_DISABLE"] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    torch._scaled_mm(a, a.t(), scale, scale, out_dtype=torch.float32)
        finally:
            if original is None:
                os.environ.pop("ECLAIR_DISABLE", None)
            else:
                os.environ["ECLAIR_DISABLE"] = original

    def test_workspace_and_determinism_configuration(self) -> None:
        a = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.ones((), device="cuda")
        original = os.environ.get("ECLAIR_TORCH_WORKSPACE_BYTES")
        try:
            for value in ("-1", "1x", "", "18446744073709551616"):
                os.environ["ECLAIR_TORCH_WORKSPACE_BYTES"] = value
                with self.assertRaisesRegex(RuntimeError, "nonnegative byte count"):
                    torch._scaled_mm(a, a.t(), scale, scale, out_dtype=torch.float32)
        finally:
            if original is None:
                os.environ.pop("ECLAIR_TORCH_WORKSPACE_BYTES", None)
            else:
                os.environ["ECLAIR_TORCH_WORKSPACE_BYTES"] = original
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            with self.assertRaisesRegex(RuntimeError, "deterministic algorithms"):
                torch._scaled_mm(a, a.t(), scale, scale, out_dtype=torch.float32)
        finally:
            torch.use_deterministic_algorithms(deterministic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    # Preserve the native failure before loading. A passing result after loading
    # cannot otherwise prove that the original failing operator was replaced.
    a = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones((), device="cuda")
    try:
        torch._scaled_mm(a, a.t(), scale, scale, out_dtype=torch.float32)
    except RuntimeError as error:
        if "compute capability" not in str(error):
            raise
        print("Native failure:", str(error), flush=True)
    else:
        raise RuntimeError("negative control failed: native scaled_mm unexpectedly succeeded")
    print("torch", torch.__version__, "CUDA", torch.version.cuda,
          "device", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
    load(args.library)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
