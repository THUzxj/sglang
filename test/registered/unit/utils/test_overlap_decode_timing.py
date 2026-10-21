import ast
import math
import unittest
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import Mock


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
)


class TestOverlapDecodeTiming(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        reporter = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SchedulerMetricsReporter"
        )
        methods = {
            node.name: node
            for node in reporter.body
            if isinstance(node, ast.FunctionDef)
        }
        self.methods = methods
        names = (
            "_reset_context_engineering_timing",
            "record_overlap_disagg_decode_timing",
            "_overlap_disagg_decode_timing_suffix",
        )
        namespace = {"defaultdict": defaultdict, "math": math}
        exec(
            compile(
                ast.Module(body=[methods[name] for name in names], type_ignores=[]),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
        reporter_type = type(
            "TimingReporter", (), {name: namespace[name] for name in names}
        )
        self.reporter = reporter_type()
        self.reporter.is_stats_logging_rank = True
        self.reporter._reset_context_engineering_timing()

    def test_suffix_aggregates_and_clears_window(self):
        self.reporter.record_overlap_disagg_decode_timing("run_batch", 1.0)
        self.reporter.record_overlap_disagg_decode_timing("run_batch", 3.0)
        suffix = self.reporter._overlap_disagg_decode_timing_suffix()
        self.assertIn("run_batch: count=2, total_ms=4.000", suffix)
        self.assertIn("avg_ms=2.000, max_ms=3.000", suffix)
        self.assertEqual(self.reporter._overlap_disagg_decode_timing_suffix(), "")

    def test_non_logging_rank_does_not_collect(self):
        self.reporter.is_stats_logging_rank = False
        self.reporter.record_overlap_disagg_decode_timing("run_batch", 1.0)
        self.assertEqual(self.reporter._overlap_disagg_decode_timing, {})

    def test_decode_log_includes_and_drains_timings(self):
        self.reporter.record_overlap_disagg_decode_timing("run_batch", 1.25)
        self.reporter._get_graph_replay_timer = lambda: None
        block = next(
            node
            for node in self.methods["report_decode_stats"].body
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.is_stats_logging_rank"
        )
        messages = []
        namespace = {
            "self": self.reporter,
            "msg": "Decode batch",
            "datetime": datetime,
            "logger": SimpleNamespace(info=messages.append),
            "log_time_delta": 0.01,
        }
        exec(
            compile(ast.Module(body=[block], type_ignores=[]), str(SOURCE), "exec"),
            namespace,
        )
        self.assertIn("overlap disagg decode timing:", messages[0])
        self.assertIn("run_batch: count=1, total_ms=1.250", messages[0])
        self.assertEqual(self.reporter._overlap_disagg_decode_timing, {})

    def test_loop_preserves_overlap_order_and_reports_completed_calls(self):
        decode_source = SOURCE.parents[2] / "disaggregation/decode.py"
        tree = ast.parse(decode_source.read_text())
        mixin = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SchedulerDisaggregationDecodeMixin"
        )
        method = next(
            node for node in mixin.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "event_loop_overlap_disagg_decode"
        )
        method.decorator_list = []
        namespace = {
            "Scheduler": object,
            "ScheduleBatch": object,
            "Optional": Optional,
            "Callable": Callable,
            "deque": deque,
            "time": SimpleNamespace(perf_counter=Mock(side_effect=range(1000))),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(decode_source), "exec"),
            namespace,
        )
        for disable_overlap in (False, True):
            with self.subTest(disable_overlap=disable_overlap):
                events = []
                batch = SimpleNamespace(copy=lambda: batch)
                batches = iter([batch, batch, None])
                receives = iter([[], [], []])

                def stage(name, result=None):
                    def run(*args, **kwargs):
                        events.append(name)
                        return result
                    return run

                scheduler = SimpleNamespace(
                    metrics_reporter=self.reporter,
                    running_batch=None,
                    chunked_req=None,
                    _engine_paused=False,
                    request_receiver=SimpleNamespace(recv_requests=lambda: next(receives)),
                    process_input_requests=stage("input"),
                    process_decode_queue=stage("queue"),
                    get_next_disagg_decode_batch_to_run=lambda **kwargs: SimpleNamespace(
                        running_batch=batch, batch_to_run=next(batches)
                    ),
                    ngram_embedding_manager=SimpleNamespace(
                        prepare_for_forward=lambda batch, **kwargs: batch
                    ),
                    is_disable_overlap_for_batch=lambda *args, **kwargs: disable_overlap,
                    run_batch=stage("run", object()),
                    _apply_war_barrier=stage("barrier"),
                    process_batch_result=stage("process"),
                    _finalize_pending_compact_pauses=stage("finalize"),
                    launch_batch_sample_if_needed=stage("sample"),
                    on_idle=stage("idle"),
                )
                self.reporter._overlap_disagg_decode_timing.clear()
                with self.assertRaises(StopIteration):
                    namespace[method.name](scheduler)
                stats = self.reporter._overlap_disagg_decode_timing
                self.assertEqual(stats["recv_requests"][0], 3)
                self.assertEqual(stats["run_batch"][0], 2)
                self.assertEqual(stats["process_batch_result"][0], 2)
                self.assertEqual(stats["_finalize_pending_compact_pauses"][0], 2)
                self.assertEqual(stats["batch.copy"][0], 2)
                second_run = events.index("run", events.index("run") + 1)
                first_process = events.index("process")
                if disable_overlap:
                    self.assertLess(first_process, second_run)
                else:
                    self.assertGreater(first_process, second_run)
                self.assertEqual(events.count("sample"), 3)


if __name__ == "__main__":
    unittest.main()
