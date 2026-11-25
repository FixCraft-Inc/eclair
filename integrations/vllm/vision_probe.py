#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Start one vision serving profile, ask about locally drawn images, score the replies, and stop.

Usage: vision_probe.py PROFILE.json LOG. Images are drawn with Pillow (vLLM's
own dependency): numbers and words at large and small sizes, and two colored
shapes. They are sent inline as data URLs, so nothing is fetched. A text reply
is correct when it equals the drawn text, ignoring case, periods and backticks.
A shapes reply is correct when it names both colored shapes. Like
generation_probe.py, this owns no GPU preflight or cooling monitor. Run it only
on an idle V100 for bounded periods.
"""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

# (kind, text drawn, image size, font size). Numbers and words at a large and a
# small size, then shapes. The expected answer is the drawn text or shape list.
CASES = (("number", "4718", (960, 540), 180), ("number", "7351", (960, 540), 180),
         ("number", "90264", (640, 360), 120), ("number", "3815", (480, 270), 64),
         ("word", "LANTERN", (960, 540), 150), ("word", "orbit", (640, 360), 90),
         ("shapes", "red square, blue circle", (960, 540), 0))
QUESTIONS = {"number": "Which number is written in the image? Reply with digits only.",
             "word": "Which word is written in the image? Reply with the word only.",
             "shapes": "Name the shape on the left and the shape on the right with their colors, "
                       "as 'color shape, color shape'."}


def image_url(kind: str, text: str, size: tuple[int, int], font: int) -> str:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    if kind == "shapes":
        side = height // 3
        draw.rectangle((width // 8, height // 3, width // 8 + side, height // 3 + side), fill="red")
        draw.ellipse((width * 5 // 8, height // 3, width * 5 // 8 + side, height // 3 + side),
                     fill="blue")
    else:
        draw.text((width // 12, height // 3), text, fill="black",
                  font=ImageFont.load_default(size=font))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def ask(kind: str, text: str, size: tuple[int, int], font: int) -> dict:
    content = [{"type": "image_url", "image_url": {"url": image_url(kind, text, size, font)}},
               {"type": "text", "text": QUESTIONS[kind]}]
    body = json.dumps({"model": "eclair-qwen", "temperature": 0, "max_tokens": 32,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "user", "content": content}]}).encode()
    request = urllib.request.Request("http://127.0.0.1:18080/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.load(response)
    answer = (reply["choices"][0]["message"].get("content") or "").strip()
    normalized = answer.lower().replace(".", "").replace("`", "").strip()
    correct = (all(part in normalized for part in text.split(", ")) if kind == "shapes"
               else normalized == text.lower())
    return {"kind": kind, "expected": text, "size": size, "answer": answer,
            "correct": correct, "seconds": round(time.monotonic() - started, 2),
            "prompt_tokens": reply.get("usage", {}).get("prompt_tokens")}


def main() -> None:
    profile, log_path = sys.argv[1], sys.argv[2]
    data = json.load(open(profile))
    with open(log_path, "w") as log:
        process = subprocess.Popen(data["argv"], stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, **data["env"]}, cwd=os.path.dirname(log_path))
    try:
        started = time.monotonic()
        while True:
            if process.poll() is not None:
                print("engine exited", process.returncode)
                return
            if time.monotonic() - started > 900:
                print("startup timeout")
                return
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2):
                    break
            except OSError:
                time.sleep(2)
        print(json.dumps({"startup_seconds": round(time.monotonic() - started, 1)}))
        results = [ask(*case) for case in CASES]
        for result in results:
            print(json.dumps(result, ensure_ascii=False))
        print(json.dumps({"correct": sum(r["correct"] for r in results), "cases": len(results)}))
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
