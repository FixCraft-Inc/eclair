#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Profile one prefill of a serving profile with vLLM's torch profiler and print top kernels.

Usage: prefill_profile.py PROFILE.json OUTPUT_DIR [PROMPT_TOKENS]. Starts the
engine with vLLM's own --profiler-config, sends one warmup request, then profiles
a single request of about PROMPT_TOKENS tokens (default 12288) with one output
token, and prints the worker's CUDA-time table. The prompt is local filler text.
Like generation_probe.py it owns no GPU preflight or cooling monitor, so run it
only on an idle V100 for bounded periods.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 18080


def post(path: str, body: dict | None = None, timeout: float = 1800) -> dict:
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def prompt(tokens: int, salt: str) -> str:
    # About 1.3 words per token. The salt keeps each request out of the prefix cache.
    words = [f"{salt}-item{i % 997} value{i % 89} note{i % 13}" for i in range(tokens // 4)]
    return "Summarize the following list in one word.\n" + " ".join(words)


def chat(text: str) -> dict:
    started = time.monotonic()
    reply = post("/v1/chat/completions", {
        "model": "eclair-qwen", "max_tokens": 1, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": text}]})
    return {"seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": reply.get("usage", {}).get("prompt_tokens")}


def main() -> None:
    profile, output = sys.argv[1], Path(sys.argv[2])
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 12288
    output.mkdir(parents=True, exist_ok=False)
    data = json.load(open(profile))
    config = {"profiler": "torch", "torch_profiler_dir": str(output.resolve()),
              "torch_profiler_with_stack": False, "torch_profiler_use_gzip": False}
    argv = data["argv"] + ["--profiler-config", json.dumps(config)]
    with open(output / "engine.log", "w") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, **data["env"]}, cwd=str(output))
    try:
        started = time.monotonic()
        while True:
            if process.poll() is not None:
                raise SystemExit(f"engine exited {process.returncode}")
            if time.monotonic() - started > 900:
                raise SystemExit("startup timeout")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                    break
            except OSError:
                time.sleep(2)
        print(json.dumps({"warmup": chat(prompt(tokens, "warm"))}))
        post("/start_profile")
        print(json.dumps({"profiled": chat(prompt(tokens, "cold"))}))
        post("/stop_profile")
        time.sleep(5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(10)
    for table in sorted(output.rglob("*.txt")):
        print(f"== {table.relative_to(output)}")
        print(table.read_text()[:12000])


if __name__ == "__main__":
    main()
