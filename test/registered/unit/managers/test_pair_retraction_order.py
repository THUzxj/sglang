import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import ScheduleBatch


class TestPairRetractionOrder(unittest.TestCase):
    def test_compact_is_retracted_before_paired_main(self):
        def req(rid, kind):
            return SimpleNamespace(
                rid=rid,
                beam_group=None,
                context_engineering_pair_key="pair",
                is_context_engineering_request=lambda: True,
                is_context_engineering_compact=lambda: kind == "compact",
            )

        compact = req("compact", "compact")
        main = req("main", "main")
        batch = SimpleNamespace(reqs=[compact, main])
        batch._get_decode_retraction_order = Mock(return_value=[0, 1])
        batch.check_decode_mem = Mock(return_value=True)
        batch.release_req = Mock(return_value=True)
        batch.filter_batch = Mock(side_effect=lambda keep_indices: setattr(
            batch, "reqs", [batch.reqs[i] for i in keep_indices]
        ))

        with patch(
            "sglang.srt.managers.schedule_batch.beam_retraction_order",
            side_effect=lambda indices, reqs: indices,
        ), patch(
            "sglang.srt.managers.schedule_batch.get_schedule",
            return_value=SimpleNamespace(enable_pair_scheduler=True),
        ), patch(
            "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
            return_value=0.5,
        ):
            retracted, ratio, aborted = ScheduleBatch.retract_decode(batch)

        self.assertEqual(retracted, [compact])
        self.assertEqual(batch.reqs, [main])
        self.assertEqual(ratio, 0.5)
        self.assertEqual(aborted, [])
        batch.release_req.assert_called_once_with(0, 1)
