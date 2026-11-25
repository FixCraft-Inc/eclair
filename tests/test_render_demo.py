#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.

"""Bounded subprocess checks for frame output, control and renderer ownership."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from collections.abc import Iterator


class RenderDemoTests(unittest.TestCase):
    binary: Path
    cuda_worker: Path | None = None

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="eclair-rt-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = dict(os.environ, XDG_RUNTIME_DIR=str(self.root))

    def execute(self, *arguments: str, success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [str(self.binary), *arguments], env=self.env, text=True,
            capture_output=True, timeout=25, check=False,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def control(self, command: str) -> dict:
        return json.loads(self.execute("--control", command).stdout)

    def gpu_memory(self) -> int:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(result.stdout.strip())

    def assert_gpu_released(self, pid: int, baseline: int) -> None:
        deadline = time.monotonic() + 5
        while True:
            used = self.gpu_memory()
            if used <= baseline:
                break
            self.assertLess(time.monotonic(), deadline, f"VRAM remained allocated: {used} MiB")
            time.sleep(0.1)
        self.assertFalse(Path(f"/proc/{pid}").exists(), "CUDA worker was not reaped")
        print(f"CUDA worker {pid} reaped; NVML memory returned to {used} MiB (baseline {baseline})")

    @contextlib.contextmanager
    def running(self, *arguments: str) -> Iterator[subprocess.Popen]:
        with (self.root / "process.log").open("w+") as log:
            process = subprocess.Popen(
                [str(self.binary), "--headless", "--listen", "--width", "64",
                 "--height", "64", "--seconds", "60", *arguments],
                env=self.env, stdout=log, stderr=log, start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 20
                while not (self.root / "eclair-rt/control.sock").is_socket():
                    if process.poll() is not None:
                        log.seek(0)
                        self.fail("preview exited during startup: " + log.read(8192))
                    if time.monotonic() >= deadline:
                        self.fail("preview socket startup timed out")
                    time.sleep(0.02)
                yield process
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                        self.fail("preview did not terminate and reap its worker")

    def test_cpu_and_raster_frames_are_distinct_and_animated(self) -> None:
        images = []
        for mode, frames in [("cpu-ray", "1"), ("raster", "1"), ("raster", "12")]:
            path = self.root / f"{mode}-{frames}.ppm"
            result = self.execute("--headless", "--mode", mode, "--width", "64",
                                  "--height", "64", "--frames", frames, "--output", str(path))
            report = json.loads(result.stdout)
            self.assertEqual(report["frames"], int(frames))
            self.assertTrue(report["cuda_worker_reaped"])
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"P6\n64 64\n255\n"))
            self.assertEqual(len(data), len(b"P6\n64 64\n255\n") + 64 * 64 * 3)
            images.append(data)
        self.assertNotEqual(images[0], images[1])
        self.assertNotEqual(images[1], images[2])

    def test_invalid_configuration_and_existing_output_fail(self) -> None:
        for arguments in [("--width", "4294967295"), ("--samples", "0"),
                          ("--bounces", "9"), ("--mode", "unknown"), ("--scene", "unknown")]:
            self.execute("--headless", *arguments, success=False)
        path = self.root / "keep.ppm"
        path.write_bytes(b"keep")
        self.execute("--headless", "--mode", "raster", "--width", "16", "--height",
                     "16", "--frames", "1", "--output", str(path), success=False)
        self.assertEqual(path.read_bytes(), b"keep")

    def test_materials_scene_in_cpu_modes(self) -> None:
        frames = []
        for mode in ["cpu-ray", "raster"]:
            path = self.root / f"materials-{mode}.ppm"
            result = self.execute(
                "--headless", "--mode", mode, "--scene", "materials", "--width", "64",
                "--height", "64", "--samples", "4", "--bounces", "6", "--frames", "1",
                "--output", str(path),
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["scene"], "materials")
            self.assertEqual(report["mode"], mode)
            self.assertEqual(report["bounces"], 6)
            frames.append(path.read_bytes())
        self.assertNotEqual(*frames)

    def test_controls_pause_snapshot_and_stop(self) -> None:
        with self.running("--mode", "raster") as process:
            self.assertTrue(self.control("status")["animated"])
            self.assertFalse(self.control("still")["animated"])
            self.assertTrue(self.control("animate")["animated"])
            status = self.control("cpu-ray")
            self.assertEqual(status["mode"], "cpu-ray")
            self.assertEqual(status["cuda_worker_pid"], 0)
            time.sleep(0.05)
            snapshot = self.control("snapshot")
            self.assertTrue(Path(snapshot["snapshot"]).is_file())
            status = self.control("pause")
            for field in ["cuda_worker_pid", "explicit_vram_bytes", "pinned_ram_bytes", "cpu_frame_bytes"]:
                self.assertEqual(status[field], 0, field)
            result = self.execute("--control", "bad-command", success=False)
            self.assertIn("error", json.loads(result.stdout))
            self.assertEqual(self.control("status")["mode"], "pause")
            self.control("stop")
            self.assertEqual(process.wait(timeout=8), 0)
        self.assertFalse((self.root / "eclair-rt/control.sock").exists())

    def test_stale_worker_frame_falls_back_and_reaps(self) -> None:
        worker = self.root / "bad-worker"
        worker.write_text(
            "#!/usr/bin/python3\nimport socket,struct\n"
            "s=socket.socket(fileno=198)\n"
            "s.send(struct.pack('<8IfI3Q',0x54524345,2,1,0,0,1,1,2,0.0,0,0,0,0))\n"
            "p=bytearray(s.recv(64))\n"
            "struct.pack_into('<I',p,16,struct.unpack_from('<I',p,16)[0]+1)\n"
            "s.send(p)\n",
            encoding="utf-8",
        )
        worker.chmod(0o700)
        with self.running("--mode", "cuda-ray", "--cuda-worker", str(worker)) as process:
            deadline = time.monotonic() + 5
            while True:
                status = self.control("status")
                if status["mode"] == "cpu-ray":
                    break
                self.assertLess(time.monotonic(), deadline, "CUDA failure did not fall back")
                time.sleep(0.02)
            self.assertEqual(status["cuda_worker_pid"], 0)
            self.assertEqual(status["pinned_ram_bytes"], 0)
            self.assertIn("stale", status["last_error"])
            self.control("stop")
            self.assertEqual(process.wait(timeout=8), 0)

    def test_previous_worker_protocol_is_rejected(self) -> None:
        worker = self.root / "old-worker"
        worker.write_text(
            "#!/usr/bin/python3\nimport socket,struct\n"
            "s=socket.socket(fileno=198)\n"
            "s.send(struct.pack('<8IfI3Q',0x54524345,1,1,0,0,1,1,2,0.0,0,0,0,0))\n"
            "s.recv(64)\n", encoding="utf-8",
        )
        worker.chmod(0o700)
        result = self.execute(
            "--headless", "--mode", "cuda-ray", "--width", "64", "--height", "64",
            "--frames", "1", "--cuda-worker", str(worker), success=False,
        )
        self.assertIn("invalid renderer packet", result.stderr)

    def test_real_cuda_transitions_and_failure(self) -> None:
        if self.cuda_worker is None:
            self.skipTest("explicit --cuda-worker required for GPU lifecycle checks")
        baseline = self.gpu_memory()
        with self.running("--mode", "raster", "--scene", "materials", "--bounces", "6",
                          "--denoise", "--cuda-worker", str(self.cuda_worker)) as process:
            for _ in range(3):
                active = self.control("cuda-ray")
                pid = active["cuda_worker_pid"]
                self.assertGreater(pid, 0)
                self.assertGreater(active["pinned_ram_bytes"], 0)
                self.assertTrue(active["denoise"])
                self.assertGreater(active["explicit_vram_bytes"], 64 * 64 * 4)
                status = self.control("cpu-ray")
                self.assertEqual(status["cuda_worker_pid"], 0)
                self.assertEqual(status["explicit_vram_bytes"], 0)
                self.assertEqual(status["pinned_ram_bytes"], 0)
                self.assert_gpu_released(pid, baseline)
            active = self.control("cuda-ray")
            os.kill(active["cuda_worker_pid"], signal.SIGKILL)
            deadline = time.monotonic() + 5
            while True:
                status = self.control("status")
                if status["mode"] == "cpu-ray":
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
            self.assertEqual(status["pinned_ram_bytes"], 0)
            self.assert_gpu_released(active["cuda_worker_pid"], baseline)
            self.control("stop")
            self.assertEqual(process.wait(timeout=8), 0)
        self.env["ECLAIR_DISABLE"] = "rt.emulation"
        disabled = self.execute(
            "--headless", "--mode", "cuda-ray", "--width", "64", "--height", "64",
            "--frames", "1", "--cuda-worker", str(self.cuda_worker), success=False,
        )
        self.assertIn("disabled by policy", disabled.stderr)
        self.assertEqual(self.gpu_memory(), baseline)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cuda-worker", type=Path)
    options, remaining = parser.parse_known_args()
    RenderDemoTests.binary = options.binary.resolve(strict=True)
    RenderDemoTests.cuda_worker = options.cuda_worker.resolve(strict=True) if options.cuda_worker else None
    program = unittest.main(argv=[__file__, *remaining], exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
