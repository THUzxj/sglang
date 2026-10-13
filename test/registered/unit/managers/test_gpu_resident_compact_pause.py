import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler import Scheduler


def make_req(rid, kind, seqlen):
    return SimpleNamespace(
        rid=rid,
        context_engineering_kind=kind,
        context_engineering_pair_key="pair",
        is_context_engineering_main=lambda: kind == "main",
        is_context_engineering_compact=lambda: kind == "compact",
        is_context_engineering_paused=False,
        allow_compact_drain=False,
        seqlen=seqlen,
        kv=SimpleNamespace(holds_kv=True),
    )


class TestGpuResidentCompactPause(unittest.TestCase):
    def test_budget_pause_keeps_compact_kv_and_resume_rejoins_batch(self):
        main = make_req("main", "main", 10)
        compact = make_req("compact", "compact", 20)
        batch = SimpleNamespace(reqs=[main, compact], batch_is_full=True)
        batch.is_empty = lambda: not batch.reqs
        batch.filter_batch = Mock(side_effect=lambda keep_indices: setattr(
            batch, "reqs", [batch.reqs[i] for i in keep_indices]
        ))
        batch.release_req = Mock()
        batch.merge_batch = Mock(side_effect=lambda other: batch.reqs.extend(other.reqs))

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.paused_compact_queue = []
        scheduler.waiting_queue = []
        scheduler.metrics_reporter = SimpleNamespace(num_paused_reqs=0)
        scheduler.max_running_requests = 4
        scheduler._build_gpu_resident_decode_batch = Mock(
            return_value=SimpleNamespace(reqs=[compact])
        )
        config = SimpleNamespace(
            enable_pair_scheduler=True,
            context_engineering_compact_pause_mode="gpu_resident",
            context_engineering_main_turn_decode_max_batch_size=None,
            context_engineering_compact_decode_max_batch_size=1,
            context_engineering_decode_attention_token_budget=15,
            context_engineering_compact_attention_cost_ratio=1.0,
        )
        with patch("sglang.srt.managers.scheduler.get_schedule", return_value=config):
            scheduler._apply_pair_decode_retract_limits(batch)
            self.assertEqual(batch.reqs, [main])
            self.assertEqual(scheduler.paused_compact_queue, [compact])
            self.assertTrue(compact.is_context_engineering_paused)
            batch.release_req.assert_not_called()

            config.context_engineering_decode_attention_token_budget = 100
            scheduler._resume_paused_compacts(batch)

        self.assertEqual(batch.reqs, [main, compact])
        self.assertEqual(scheduler.paused_compact_queue, [])
        self.assertFalse(compact.is_context_engineering_paused)
        scheduler._build_gpu_resident_decode_batch.assert_called_once_with([compact])

    def test_kv_pressure_retracts_paused_compact_from_active_pair(self):
        active_main = make_req("active-main", "main", 10)
        unrelated = make_req("unrelated", "compact", 20)
        unrelated.context_engineering_pair_key = "other"
        paired = make_req("paired", "compact", 20)
        paired.is_context_engineering_paused = True
        unrelated.is_context_engineering_paused = True

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.paused_compact_queue = [unrelated, paired]
        scheduler.req_to_token_pool = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.tree_cache = object()
        scheduler.hisparse_coordinator = None
        batch = SimpleNamespace(
            reqs=[active_main],
            _get_decode_retraction_order=lambda reqs: [0],
        )

        with patch("sglang.srt.managers.scheduler.release_req", return_value=True) as release:
            retracted, aborted, pair_key = (
                scheduler._retract_paused_compact_for_kv_pressure(batch)
            )

        self.assertEqual(retracted, [paired])
        self.assertEqual(aborted, [])
        self.assertEqual(pair_key, "pair")
        self.assertEqual(scheduler.paused_compact_queue, [unrelated])
        self.assertFalse(paired.is_context_engineering_paused)
        self.assertIs(release.call_args.kwargs["req"], paired)
