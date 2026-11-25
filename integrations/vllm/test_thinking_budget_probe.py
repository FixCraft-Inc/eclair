# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Host checks of probe exit status and evidence preservation, without vLLM."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import thinking_budget_probe


class ThinkingBudgetProbeTests(unittest.TestCase):
    def invoke(self, arguments: list[str]) -> int:
        with patch("sys.argv", ["thinking_budget_probe.py", *arguments]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return thinking_budget_probe.main()

    def test_unmet_contract_is_nonzero_and_retains_the_failure(self) -> None:
        result = {"passed": False, "cases": [{"case": "natural_end", "passed": False}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            with patch.object(thinking_budget_probe, "probe", return_value=result) as probe:
                self.assertEqual(self.invoke(["--output", str(output)]), 1)
            probe.assert_called_once_with("phrase")
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_bare_control_passes_only_with_a_passing_result(self) -> None:
        with patch.object(thinking_budget_probe, "probe", return_value={"passed": True}) as probe:
            self.assertEqual(self.invoke(["--end-mode", "bare"]), 0)
        probe.assert_called_once_with("bare")

    def test_existing_evidence_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prior.json"
            output.write_text("retained failure\n")
            with patch.object(thinking_budget_probe, "probe", return_value={"passed": True}):
                self.assertEqual(self.invoke(["--output", str(output)]), 2)
            self.assertEqual(output.read_text(), "retained failure\n")

    def test_missing_dependency_is_an_error_without_a_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent.json"
            with patch.object(thinking_budget_probe, "probe", side_effect=ImportError("vllm")):
                self.assertEqual(self.invoke(["--output", str(output)]), 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
