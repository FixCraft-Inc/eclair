#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Start one serving profile, send fixed short greedy chat requests, print replies, and stop.

Usage: generation_probe.py PROFILE.json LOG [ENV=VALUE ...]. A bring-up diagnostic for
comparing engine variants on identical prompts. Like startup_probe.py it owns no GPU
preflight or cooling monitor, so run it only on an idle V100 for bounded periods.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PROMPTS = ("Write a Python function that returns the sum of a list of integers. Reply with code only.",
           "What is the capital of France? Answer in one word.")


def chat(content: str) -> dict:
    body = json.dumps({"model": "eclair-qwen", "temperature": 0, "max_tokens": 48,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": content}]}).encode()
    request = urllib.request.Request("http://127.0.0.1:18080/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    return {"seconds": round(time.monotonic() - started, 2), "usage": reply.get("usage"),
            "content": reply["choices"][0]["message"].get("content")}


def main() -> None:
    profile, log_path = sys.argv[1], sys.argv[2]
    data = json.load(open(profile))
    extra = dict(item.split("=", 1) for item in sys.argv[3:])
    with open(log_path, "w") as log:
        process = subprocess.Popen(data["argv"], stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, **data["env"], **extra},
                                   cwd=os.path.dirname(log_path))
    try:
        started = time.monotonic()
        while True:
            if process.poll() is not None:
                print("engine exited", process.returncode)
                return
            if time.monotonic() - started > 600:
                print("startup timeout")
                return
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2):
                    break
            except OSError:
                time.sleep(2)
        for prompt in PROMPTS:
            print(json.dumps(chat(prompt), ensure_ascii=False))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(10)


if __name__ == "__main__":
    main()
