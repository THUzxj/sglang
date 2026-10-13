import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger


class TestSchedulerStatusLoggerTokens(unittest.TestCase):
    def test_dump_includes_running_request_token_usage(self):
        logger = SchedulerStatusLogger.__new__(SchedulerStatusLogger)
        logger.loggers = [object()]
        logger.rank = 0
        logger.dump_interval = 0
        logger.last_dump_time = 0
        req = SimpleNamespace(
            rid="running",
            seqlen=12,
            kv=SimpleNamespace(kv_committed_len=11, kv_allocated_len=16),
        )

        with patch("sglang.srt.utils.scheduler_status_logger.log_json") as log_json:
            logger.maybe_dump(
                SimpleNamespace(reqs=[req]),
                [SimpleNamespace(rid="waiting")],
            )

        self.assertEqual(log_json.call_args.args[1], "scheduler.status")
        payload = log_json.call_args.args[2]
        self.assertEqual(payload["running_rids"], ["running"])
        self.assertEqual(payload["queued_rids"], ["waiting"])
        self.assertEqual(
            payload["running_reqs"],
            {
                "running": {
                    "seqlen": 12,
                    "kv_committed_len": 11,
                    "kv_allocated_len": 16,
                }
            },
        )
