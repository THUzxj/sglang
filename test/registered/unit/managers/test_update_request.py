import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import UpdateRequestReqInput
from sglang.srt.managers.scheduler import Scheduler


class TestUpdateRequest(unittest.TestCase):
    def test_waiting_request_priority_and_labels_are_updated(self):
        req = SimpleNamespace(
            rid="request-1",
            priority=1,
            custom_labels={},
            context_engineering_kind="",
            context_engineering_pair_key=None,
            allow_compact_drain=False,
            _derive_context_engineering_kind=lambda: "compact",
            _derive_context_engineering_pair_key=lambda: "pair-1",
            _derive_allow_compact_drain=lambda: False,
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.chunked_req = None
        scheduler.waiting_queue = [req]
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.running_batch = None
        scheduler.last_batch = None
        scheduler.policy = MagicMock()
        scheduler.processed_tokens_counter = 0

        with patch("sglang.srt.managers.scheduler.get_parallel") as parallel, patch(
            "sglang.srt.managers.scheduler.get_schedule"
        ) as schedule:
            parallel.return_value.pp_size = 1
            schedule.return_value.enable_pair_scheduler = False
            result = scheduler.update_request(
                UpdateRequestReqInput(
                    rid="request-1",
                    priority=7,
                    custom_labels={"request_class": "compact", "pair_key": "pair-1"},
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(result.location, "waiting_queue")
        self.assertEqual(req.priority, 7)
        self.assertEqual(req.context_engineering_kind, "compact")
        self.assertEqual(req.context_engineering_pair_key, "pair-1")
        scheduler.policy.calc_priority.assert_called_once()

    def test_running_kind_update_is_rejected_without_override(self):
        req = SimpleNamespace(
            rid="request-2",
            priority=1,
            custom_labels={"request_class": "main"},
            context_engineering_kind="main",
        )
        scheduler = Scheduler.__new__(Scheduler)
        updated, skipped = scheduler._apply_request_update(
            req,
            UpdateRequestReqInput(
                rid="request-2",
                priority=5,
                context_engineering_kind="compact",
            ),
            running=True,
        )

        self.assertEqual(updated, ["priority"])
        self.assertTrue(skipped)
        self.assertEqual(req.priority, 5)
        self.assertEqual(req.context_engineering_kind, "main")
