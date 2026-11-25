#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""A plain PyTorch script: the Eclair launcher selects the native adapter."""

import torch


def main() -> None:
    a = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn)
    b = torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn).t()
    scale = torch.ones((), device="cuda")
    output = torch._scaled_mm(a, b, scale, scale, out_dtype=torch.float32)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu(), torch.full((16, 16), 32.0))
    print("FP8 scaled_mm passed: 16x32 times 32x16; FP32 output; exact result 32")


if __name__ == "__main__":
    main()
