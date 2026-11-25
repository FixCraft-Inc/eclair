# ECLAIR - Emulation and Compatibility Layer for Accelerated Inference and Rendering
# Copyright (C) 2026 FixCraft Inc.
# Licensed under the GNU Affero General Public License v3.0 or later.
"""Independent timelines check step boundaries and overlapping device activity."""

import unittest

import decode_trace


def event(name, start, duration, correlation, cat="kernel", graph=1):
    return {"name": name, "ts": start, "dur": duration, "cat": cat,
            "args": {"correlation": correlation, "graph id": graph, "grid": [10, 1, 48]}}


W4 = "void gemm_half_q_half_gptq_4bit_kernel<true, 3>()"


class DecodeTraceTests(unittest.TestCase):
    def test_complete_step_includes_drafts_copies_and_idle_without_double_counting(self):
        events = [event("target start", 100, 10, 1), event(W4, 110, 20, 1),
                  event("draft", 140, 30, 2), event("copy", 150, 30, 3, "gpu_memcpy"),
                  event("host", 180, 10, 3, "cpu_op"),
                  event("target start", 200, 10, 4), event(W4, 210, 20, 4)]
        result = decode_trace.summarize(events[::-1])
        self.assertEqual(result["complete_steps"], 1)
        row = result["steps"][0]
        self.assertAlmostEqual(row["window_ms"], 0.1)
        self.assertAlmostEqual(row["busy_ms"], 0.07)
        self.assertAlmostEqual(row["idle_ms"], 0.03)
        self.assertAlmostEqual(row["tail_idle_ms"], 0.02)
        self.assertEqual(row["kernel_ms"], {"other": 0.04, "w4a16": 0.02})
        self.assertEqual(result["w4a16_shapes"], [
            {"m": 3, "grid": [10, 1, 48], "calls": 1, "calls_per_step": 1}])
        self.assertEqual(len(result["w4a16_calls"]), 1)

    def test_refuses_missing_single_or_eager_target_graphs(self):
        for events in ([], [event(W4, 0, 1, 1)],
                       [event(W4, 0, 1, 1, graph=0), event(W4, 2, 1, 2)]):
            with self.subTest(events=events), self.assertRaises(ValueError):
                decode_trace.summarize(events)

    def test_union_clips_activity_at_both_step_boundaries(self):
        self.assertEqual(decode_trace.intervals([
            event("a", 0, 15, 0), event("b", 12, 20, 0), event("c", 40, 5, 0)], 10, 25),
            [(10, 25)])

    def test_native_partial_identifies_target_and_reduction_is_included(self):
        name = "void eclair::cuda::(anonymous namespace)::w4a16_partial<(unsigned int)3>()"
        reduce = "void eclair::cuda::(anonymous namespace)::w4a16_reduce()"
        result = decode_trace.summarize([
            event(name, 0, 10, 1), event(reduce, 10, 2, 1),
            event("draft", 15, 5, 2), event(name, 30, 10, 3)])
        self.assertEqual(result["steps"][0]["kernel_ms"]["w4a16"], 0.012)
        self.assertEqual(result["w4a16_calls"][0]["m"], 3)
        self.assertEqual(result["w4a16_calls"][0]["implementation"], "eclair")


if __name__ == "__main__":
    unittest.main()
