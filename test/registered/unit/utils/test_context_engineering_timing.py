import ast
import math
import unittest
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
)


class TestContextEngineeringTiming(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        reporter_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SchedulerMetricsReporter"
        )
        self.methods = {
            node.name: node
            for node in reporter_class.body
            if isinstance(node, ast.FunctionDef)
        }
        method_names = (
            "_reset_context_engineering_timing",
            "_context_engineering_timing_suffix",
            "_context_engineering_check_timing_suffix",
            "record_context_engineering_check_timing",
        )
        namespace = {"defaultdict": defaultdict, "math": math}
        exec(
            compile(
                ast.Module(
                    body=[self.methods[name] for name in method_names],
                    type_ignores=[],
                ),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
        reporter_type = type(
            "TimingReporter", (), {name: namespace[name] for name in method_names}
        )
        self.reporter = reporter_type()
        self.reporter.is_stats_logging_rank = True
        self.reporter._reset_context_engineering_timing()

    def test_decode_log_preserves_checks_without_compact_events(self):
        self.reporter.record_context_engineering_check_timing(
            "pair_resume_check", 0.005
        )
        self.reporter.record_context_engineering_check_timing(
            "pair_retract_check", 0.02
        )
        block = next(
            node
            for node in self.methods["report_decode_stats"].body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.is_stats_logging_rank"
        )
        messages = []
        self.reporter._get_graph_replay_timer = lambda: None
        namespace = {
            "self": self.reporter,
            "msg": "Decode batch, #ce-compact: 0",
            "datetime": datetime,
            "logger": SimpleNamespace(info=messages.append),
            "log_time_delta": 0.01,
        }
        exec(
            compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"),
            namespace,
        )
        self.assertIn("ce check timing:", messages[0])
        self.assertIn("pair_resume_check: count=1, avg_ms=0.005", messages[0])
        self.assertIn("pair_retract_check: count=1, avg_ms=0.020", messages[0])
        self.assertEqual(self.reporter._context_engineering_check_timing, {})

    def test_event_and_check_windows_are_independent(self):
        self.reporter._context_engineering_timing["radix_pause_local"].append(1.0)
        self.reporter.record_context_engineering_check_timing(
            "pair_resume_check", 0.005
        )
        self.assertIn(
            "radix_pause_local: count=1",
            self.reporter._context_engineering_timing_suffix(),
        )
        self.assertIn(
            "pair_resume_check: count=1",
            self.reporter._context_engineering_check_timing_suffix(),
        )
        self.assertEqual(self.reporter._context_engineering_timing_suffix(), "")
        self.assertEqual(
            self.reporter._context_engineering_check_timing_suffix(), ""
        )

    def test_non_logging_rank_does_not_collect_checks(self):
        self.reporter.is_stats_logging_rank = False
        self.reporter.record_context_engineering_check_timing(
            "pair_resume_check", 0.005
        )
        self.assertEqual(self.reporter._context_engineering_check_timing, {})


if __name__ == "__main__":
    unittest.main()
