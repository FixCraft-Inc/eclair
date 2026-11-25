#!/usr/bin/env python3
# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Start one reviewed serving profile, wait for /health or exit, report, and always stop it.

Usage: startup_probe.py PROFILE.json LOG DEADLINE_SECONDS. A quick bring-up loop only:
it owns no GPU preflight or cooling monitor, so run it on an idle V100 for bounded periods.
"""
import json, os, signal, subprocess, sys, time, urllib.request
profile, log_path, deadline = sys.argv[1], sys.argv[2], float(sys.argv[3])
data = json.load(open(profile))
with open(log_path, "w") as log:
    process = subprocess.Popen(data["argv"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               env={**os.environ, **data["env"]}, cwd=os.path.dirname(log_path),
                               start_new_session=True)
started, state = time.monotonic(), "timeout"
try:
    while time.monotonic() - started < deadline:
        if process.poll() is not None:
            state = f"exited {process.returncode}"
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as r:
                if r.status == 200:
                    state = "healthy"
                    break
        except OSError:
            pass
        time.sleep(2)
    print(state, round(time.monotonic() - started, 1))
finally:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(10)
