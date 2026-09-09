from types import SimpleNamespace
from unittest.mock import MagicMock

import sglang.srt.managers.schedule_batch as schedule_batch
import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode


class _FakeReq:
    def __init__(self):
        self.reset_count = 0

    def finished(self):
        return False

    def reset_for_retract(self):
        self.reset_count += 1


def test_release_req_can_transfer_pressure_victim_to_radix(monkeypatch):
    req = _FakeReq()
    cache_calls = []
    evict_calls = []

    monkeypatch.setattr(
        schedule_batch,
        "release_kv_cache",
        lambda actual_req, tree_cache, is_insert: cache_calls.append(
            (actual_req, tree_cache, is_insert)
        ),
    )
    monkeypatch.setattr(
        schedule_batch,
        "evict_from_tree_cache",
        lambda tree_cache, num_tokens: evict_calls.append((tree_cache, num_tokens)),
    )
    monkeypatch.setattr(
        schedule_batch.envs.SGLANG_RETRACT_DECODE_STEPS, "get", lambda: 2
    )

    tree_cache = object()
    schedule_batch.release_req(
        req=req,
        remaing_req_count=3,
        server_args=SimpleNamespace(disaggregation_mode="null"),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        tree_cache=tree_cache,
        hisparse_coordinator=None,
        offload_kv=False,
        insert_into_radix_cache=True,
    )

    assert cache_calls == [(req, tree_cache, True)]
    assert evict_calls == [(tree_cache, 6)]
    assert req.reset_count == 1


def test_radix_evictable_pause_detaches_pd_req_without_cpu_offload(monkeypatch):
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.tree_cache = object()

    req = MagicMock()
    req.output_ids = [11, 22]
    req.effective_kv_committed_len.return_value = 17

    release_calls = []
    monkeypatch.setattr(
        scheduler_module,
        "release_kv_cache",
        lambda actual_req, tree_cache, is_insert: release_calls.append(
            (actual_req, tree_cache, is_insert)
        ),
    )

    scheduler._park_compact_in_radix_cache(req)

    assert req.output_ids == [11]
    assert req.pd_rebootstrap_forced_output_id == 22
    assert req.pd_rebootstrap_in_progress is True
    assert release_calls == [(req, scheduler.tree_cache, True)]
    req.detach_gpu_state_for_cache_pause.assert_called_once_with(17)
    assert req.is_context_engineering_paused is True
    assert req.is_context_engineering_cache_paused is True
