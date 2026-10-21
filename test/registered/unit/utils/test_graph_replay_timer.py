"""CPU-only tests: no CUDA context, server, or model required."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"
# Avoid initializing model dependencies through sglang.__init__ in CPU tests.
ci_spec = importlib.util.spec_from_file_location(
    "graph_timer_ci_register", ROOT.parent / "test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(ci_spec)
sys.modules[ci_spec.name] = ci
ci_spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "graph_timer_under_test", ROOT / "utils/device_timer.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeCUDA:
    def __init__(self):
        self.now = 0.0
        self.complete = -1.0
        self.stream = SimpleNamespace(device="cuda:0", cuda_stream=13)

    def current_stream(self):
        return self.stream

    def Event(self, *, enable_timing):
        assert enable_timing
        clock = self

        class Event:
            def record(self, stream):
                self.time = clock.now

            def query(self):
                return self.time <= clock.complete

            def elapsed_time(self, other):
                assert other.query(), "Must never read an incomplete event"
                return other.time - self.time

            def synchronize(self):
                raise AssertionError("Timing must never synchronize")

        return Event()


class TestGraphReplayTimer(unittest.TestCase):
    def setUp(self):
        self.cuda = FakeCUDA()
        self.patch = patch.object(module.torch, "cuda", self.cuda)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.timer = module.GraphReplayTimer()

    def replay(self, start, end, bs=16):
        self.cuda.now = start
        with self.timer.wrap(batch_size=bs):
            self.cuda.now = end

    def test_async_samples_and_gap_across_log_windows(self):
        self.replay(0, 4)
        pending = self.timer.take_log_suffix()
        self.assertIn("cuda graph samples: 0", pending)
        self.assertIn("cuda graph ms avg: nan", pending)
        self.assertIn("cuda graph pending: 1", pending)
        self.cuda.complete = 4
        first = self.timer.take_log_suffix()
        self.assertIn("cuda graph ms avg: 4.000", first)
        self.assertIn("cuda graph gap samples: 0", first)
        self.replay(9, 15, bs=32)
        self.cuda.complete = 15
        second = self.timer.take_log_suffix()
        self.assertIn("cuda graph ms avg: 6.000", second)
        self.assertIn("cuda graph gap ms avg: 5.000", second)
        self.assertIn("cuda graph last sample: 2", second)
        self.assertIn("cuda graph last bs: 32", second)
        self.assertIn("cuda graph samples: 0", self.timer.take_log_suffix())

    def test_aggregate_completed_samples(self):
        self.replay(0, 4)
        self.replay(9, 15)
        self.replay(17, 20)
        self.cuda.complete = 20
        result = self.timer.take_log_suffix()
        self.assertIn("cuda graph samples: 3", result)
        self.assertIn("cuda graph ms avg: 4.333", result)
        self.assertIn("cuda graph ms max: 6.000", result)
        self.assertIn("cuda graph gap samples: 2", result)
        self.assertIn("cuda graph gap ms avg: 3.500", result)

    def test_idle_or_eager_break_discards_only_gap(self):
        self.replay(0, 4)
        self.timer.break_sequence()
        self.replay(1000, 1005)
        self.cuda.complete = 1005
        result = self.timer.take_log_suffix()
        self.assertIn("cuda graph samples: 2", result)
        self.assertIn("cuda graph gap samples: 0", result)

    def test_stream_change_does_not_make_cross_stream_gap(self):
        self.replay(0, 4)
        self.cuda.stream = SimpleNamespace(device="cuda:0", cuda_stream=45)
        self.replay(10, 15)
        self.cuda.complete = 15
        self.assertIn("cuda graph gap samples: 0", self.timer.take_log_suffix())

    def test_bounded_backlog_drops_without_waiting(self):
        self.timer = module.GraphReplayTimer(max_pending=1)
        self.replay(0, 4)
        self.replay(8, 12)
        self.cuda.complete = 12
        result = self.timer.take_log_suffix()
        self.assertIn("cuda graph samples: 1", result)
        self.assertIn("cuda graph dropped: 1", result)
        self.replay(16, 20)
        self.cuda.complete = 20
        self.assertIn("cuda graph gap samples: 0", self.timer.take_log_suffix())

    def test_replay_exception_does_not_keep_gap(self):
        self.replay(0, 4)
        with self.assertRaises(ValueError):
            with self.timer.wrap(batch_size=16):
                raise ValueError("failed replay")
        self.replay(10, 15)
        self.cuda.complete = 15
        self.assertIn("cuda graph gap samples: 0", self.timer.take_log_suffix())

    def reporter_method(self, name):
        tree = ast.parse(
            (ROOT / "managers/scheduler_components/metrics_reporter.py").read_text()
        )
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "SchedulerMetricsReporter"
        )
        return next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )

    def test_decode_log_consumes_ready_samples(self):
        # Run the production log block without importing GPU-only scheduler code.
        from datetime import datetime

        method = self.reporter_method("report_decode_stats")
        block = next(
            n for n in method.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "self.is_stats_logging_rank"
        )
        messages = []
        namespace = dict(
            self=SimpleNamespace(
                is_stats_logging_rank=True,
                _get_graph_replay_timer=lambda: self.timer,
            ),
            msg="Decode batch, #running-req: 16",
            datetime=datetime,
            logger=SimpleNamespace(info=messages.append),
            log_time_delta=0.01,
        )
        self.replay(0, 4)
        self.cuda.complete = 4
        exec(
            compile(ast.Module(body=[block], type_ignores=[]), "decode_log", "exec"),
            namespace,
        )
        self.assertIn("Decode batch, #running-req: 16", messages[0])
        self.assertIn("cuda graph ms avg: 4.000", messages[0])
        self.assertIn("log interval (ms): 10.000", messages[0])

    def test_actual_idle_hook_breaks_gap_without_metrics(self):
        import time

        method = self.reporter_method("_maybe_log_idle_metrics")
        namespace = dict(time=time, ENABLE_METRICS_DEVICE_TIMER=False)
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "idle_hook", "exec"),
            namespace,
        )
        reporter = SimpleNamespace(
            _get_graph_replay_timer=lambda: self.timer,
            _reset_device_timer_window=lambda: None,
            current_scheduler_metrics_enabled=False,
        )
        self.replay(0, 4)
        namespace["_maybe_log_idle_metrics"](reporter)
        self.replay(1000, 1005)
        self.cuda.complete = 1005
        self.assertIn("cuda graph gap samples: 0", self.timer.take_log_suffix())

    def test_actual_backend_replay_hook(self):
        # Execute the production method without importing GPU-only backends.
        tree = ast.parse((ROOT / "model_executor/runner_backend/full_cuda_graph_backend.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "replay")
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "backend_replay", "exec"), namespace)
        calls = []
        def graph_replay():
            calls.append(True)
            self.cuda.now += 7
        backend = SimpleNamespace(
            _model_runner=SimpleNamespace(graph_replay_timer=self.timer),
            _graphs={"key": SimpleNamespace(replay=graph_replay)},
            _outputs={"key": "output"},
        )
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True), batch_size=12)
        self.assertEqual(namespace["replay"](backend, "key", batch), "output")
        self.cuda.complete = 7
        self.assertIn("cuda graph ms avg: 7.000", self.timer.take_log_suffix())
        backend._model_runner.graph_replay_timer = None
        namespace["replay"](backend, "key", batch)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
