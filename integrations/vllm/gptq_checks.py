#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Check real sequential symmetric GPTQ layers against independent FP64 sums.

Run in the pinned vLLM venv on an idle V100. GPTQv1 packs eight input values
in each int32, least significant nibble first. Symmetric zero points store 7
(the format's offset makes the effective zero 8). Act-order is not supported.
The loader and arithmetic under test are vLLM's, admitted by eclair-gptq-sm70.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from v100_checks import compare, COSINE_LIMIT, REL_FRO_LIMIT

LAYERS = ("model.language_model.layers.3.self_attn.q_proj",
          "model.language_model.layers.3.mlp.down_proj",
          "model.language_model.layers.0.linear_attn.in_proj_qkv")


def reference_weight(torch, tensors: dict):
    """FP64 [K,N] weights, rejecting a different GPTQ representation."""
    packed, scales, zeros, groups = (tensors[k] for k in ("qweight", "scales", "qzeros", "g_idx"))
    k, n = packed.shape[0] * 8, packed.shape[1]
    if k % 128 or n % 8 or scales.shape != (k // 128, n) or zeros.shape != (k // 128, n // 8):
        raise ValueError("unexpected GPTQ tensor shapes")
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    if not bool((((zeros.to(torch.int64).unsqueeze(-1) >> shifts) & 15) == 7).all()):
        raise ValueError("checkpoint is not symmetric GPTQv1 (stored zero must be 7)")
    if groups.shape != (k,) or not torch.equal(groups.to(torch.int64), torch.arange(k) // 128):
        raise ValueError("checkpoint uses a nonsequential group index")
    # [K/8,8,N], then flatten the first two axes to restore input order.
    values = ((packed.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & 15).reshape(k, n)
    return (values.to(torch.float64) - 8) * scales.half().double().repeat_interleave(128, dim=0)


def layer_check(torch, config, tensors: dict, prefix: str, stream) -> list[dict]:
    from eclair_vllm.gptq_sm70 import Sm70GPTQLinear

    weight = reference_weight(torch, tensors)
    k, n = weight.shape
    method = Sm70GPTQLinear(config)
    layer = torch.nn.Module()
    with torch.device("cuda"):
        method.create_weights(layer, input_size_per_partition=k, output_partition_sizes=[n],
                              input_size=k, output_size=n, params_dtype=torch.float16,
                              weight_loader=lambda *a, **kw: None)
    with torch.no_grad():
        for name in ("qweight", "scales", "qzeros"):
            getattr(layer, name).copy_(tensors[name])
    method.process_weights_after_loading(layer)
    results = []
    generator = torch.Generator().manual_seed(1234)
    for rows in (1, 3, 64):
        x = torch.randn(rows, k, generator=generator).half()
        expected = x.double() @ weight
        with torch.cuda.stream(stream):
            device_x = x.cuda()
            actual = method.apply(layer, device_x)
        stream.synchronize()
        result = compare(actual.cpu(), expected)
        result.update(layer=prefix, shape=[rows, n, k], kernel=type(method.kernel).__name__)
        result["passed"] = (result["finite"] and result["rel_fro"] <= REL_FRO_LIMIT
                            and result["cosine"] >= COSINE_LIMIT and actual.dtype == torch.float16
                            and tuple(actual.shape) == (rows, n))
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    import torch
    import vllm
    from safetensors import safe_open
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from eclair_vllm.gptq_sm70 import EclairGPTQSm70Config, METHOD, Sm70GPTQLinear

    record: dict = {"schema": "eclair.vllm-gptq-checks/1", "passed": False,
                    "vllm": vllm.__version__, "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(0), "results": [],
                    "reference": "independently unpacked GPTQv1, FP64 products and sums",
                    "synchronization": "nondefault stream, stream.synchronize after each call",
                    "tolerance": {"rel_fro": REL_FRO_LIMIT, "cosine": COSINE_LIMIT}}
    try:
        with set_current_vllm_config(VllmConfig()):
            raw = json.loads((args.checkpoint / "config.json").read_text())["quantization_config"]
            config = EclairGPTQSm70Config.from_config(raw)
            if EclairGPTQSm70Config.override_quantization_method(raw, METHOD) != METHOD:
                raise RuntimeError("explicit GPTQ profile was not admitted")
            if EclairGPTQSm70Config.override_quantization_method(raw, None) is not None:
                raise RuntimeError("GPTQ adapter claimed an implicit launch")
            # A changed format must fail before parameter creation or a kernel launch.
            for key, value in (("weight_bits", 8), ("group_size", 32),
                               ("is_sym", False), ("desc_act", True)):
                old = getattr(config, key)
                setattr(config, key, value)
                try:
                    Sm70GPTQLinear(config)
                except ValueError:
                    pass
                else:
                    raise RuntimeError("unsupported GPTQ setting admitted: " + key)
                finally:
                    setattr(config, key, old)
            index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
            torch.cuda.set_device(0)
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                         distributed_init_method=f"file://{args.output}.rendezvous")
            initialize_model_parallel(1, 1)
            stream = torch.cuda.Stream()
            with set_current_vllm_config(VllmConfig()):
                for prefix in LAYERS:
                    tensors = {}
                    for suffix in ("qweight", "scales", "qzeros", "g_idx"):
                        name = prefix + "." + suffix
                        with safe_open(args.checkpoint / index[name], framework="pt") as source:
                            tensors[suffix] = source.get_tensor(name)
                    record["results"] += layer_check(torch, config, tensors, prefix, stream)
            record["passed"] = len(record["results"]) == 9 and all(r["passed"] for r in record["results"])
    except Exception as error:  # preserve setup/numerical failures as evidence
        record["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("passed", "error", "vllm")}))
    raise SystemExit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
