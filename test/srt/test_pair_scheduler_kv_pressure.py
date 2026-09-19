from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import sglang.srt.disaggregation.decode as decode_module
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


def test_radix_evictable_pause_pins_mamba_checkpoint_on_host(monkeypatch):
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=16)
    host_pool = SimpleNamespace(available_size=lambda: 100)
    mamba_host = SimpleNamespace(available_size=lambda: 1)
    host_lock = object()
    tree_cache = MagicMock()
    tree_cache.supports_mamba.return_value = True
    tree_cache.cache_controller.mem_pool_host = host_pool
    tree_cache.components[scheduler_module.ComponentType.MAMBA]._mamba_pool_host = (
        mamba_host
    )
    tree_cache.protect_cache_paused_node.return_value = host_lock
    scheduler.tree_cache = tree_cache

    req = MagicMock()
    req.origin_input_ids = list(range(31))
    req.output_ids = [11, 22]
    req.effective_kv_committed_len.return_value = 32
    req.last_node = 7
    monkeypatch.setattr(
        scheduler_module,
        "get_exec",
        lambda: SimpleNamespace(mamba=SimpleNamespace(enable_linear_replayssm=False)),
    )
    release_calls = []

    def release(actual_req, actual_cache, is_insert):
        release_calls.append((actual_req, actual_cache, is_insert))
        assert actual_req.is_context_engineering_cache_parking is True

    monkeypatch.setattr(scheduler_module, "release_kv_cache", release)

    assert scheduler._park_compact_in_radix_cache(req) is True
    assert release_calls == [(req, tree_cache, True)]
    tree_cache.protect_cache_paused_node.assert_called_once_with(7)
    req.detach_gpu_state_for_cache_pause.assert_called_once_with(32)
    assert req.output_ids == [11, 22]
    assert req.context_engineering_pause_node == 7
    assert req.context_engineering_pause_host_lock is host_lock
    assert req.context_engineering_pause_pending_token_id == 22
    assert req.is_context_engineering_cache_parking is False


def test_radix_evictable_pause_rejects_missing_host_backup(monkeypatch):
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=16)
    tree_cache = MagicMock()
    tree_cache.supports_mamba.return_value = True
    tree_cache.cache_controller.mem_pool_host.available_size.return_value = 100
    tree_cache.components[
        scheduler_module.ComponentType.MAMBA
    ]._mamba_pool_host.available_size.return_value = 1
    tree_cache.protect_cache_paused_node.return_value = None
    scheduler.tree_cache = tree_cache

    req = MagicMock()
    req.rid = "strict-host-backup"
    req.origin_input_ids = list(range(31))
    req.output_ids = [11, 22]
    req.effective_kv_committed_len.return_value = 32
    req.last_node = 7
    monkeypatch.setattr(
        scheduler_module,
        "get_exec",
        lambda: SimpleNamespace(mamba=SimpleNamespace(enable_linear_replayssm=False)),
    )
    monkeypatch.setattr(
        scheduler_module, "release_kv_cache", lambda *args, **kwargs: None
    )

    with pytest.raises(RuntimeError, match="protected host backup"):
        scheduler._park_compact_in_radix_cache(req)

    req.detach_gpu_state_for_cache_pause.assert_not_called()
    assert req.is_context_engineering_cache_parking is False


def test_local_cache_resume_restores_pending_token_and_radix_ownership(monkeypatch):
    node_id = 7
    mamba_match = SimpleNamespace(best_match_node=node_id)
    full_match = SimpleNamespace(
        device_indices=torch.arange(2, dtype=torch.int64),
        host_hit_length=0,
        last_device_node=node_id,
    )
    matches = iter((mamba_match, full_match))
    monkeypatch.setattr(
        decode_module, "match_prefix_for_req", lambda *args, **kwargs: next(matches)
    )

    lock = MagicMock()
    lock.swa_uuid_for_lock = None
    lock.skip_lock_node_ids = None
    host_lock = object()
    tree_core = MagicMock()
    tree_core.get_component_device_value.return_value = torch.tensor(
        [3], dtype=torch.int64
    )
    tree_core.component_has_host_value_only.return_value = False
    tree_cache = MagicMock()
    tree_cache.tree_core = tree_core
    tree_cache.inc_lock_ref.return_value = lock
    tree_cache.full_evictable_size.return_value = 0
    tree_cache.mamba_evictable_size.return_value = 0

    mamba_pool = MagicMock()
    req_pool = SimpleNamespace(
        available_size=lambda: 1,
        enable_mamba_extra_buffer=False,
        mamba_allocator=SimpleNamespace(available_size=lambda: 1),
        mamba_ckpt_pool=None,
        mamba_pool=mamba_pool,
        translate_mamba_indices=lambda value: value,
    )
    req = SimpleNamespace(
        rid="local-resume",
        origin_input_ids=[1, 2],
        output_ids=[],
        context_engineering_pause_committed_len=2,
        context_engineering_pause_pending_token_id=9,
        context_engineering_pause_node=node_id,
        context_engineering_pause_host_lock=host_lock,
        is_context_engineering_cache_paused=True,
        is_retracted=True,
        cache_protected_len=0,
        mamba_pool_idx=None,
        mamba_needs_clear=True,
        kv_committed_len=0,
    )

    queue = SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 0),
        tree_cache=tree_cache,
        _pre_alloc_fill_len=lambda actual_req: 2,
        _required_alloc_tokens=lambda **kwargs: 0,
        _pre_alloc=lambda *args: setattr(
            req, "mamba_pool_idx", torch.tensor(4, dtype=torch.int64)
        ),
    )

    restored = decode_module.DecodePreallocQueue.restore_cache_paused_req(queue, req)

    assert restored is True
    assert req.output_ids == [9]
    assert req.cache_protected_len == 2
    assert req.kv_committed_len == 2
    assert req.is_context_engineering_cache_paused is False
    tree_cache.dec_host_lock_ref.assert_called_once_with(node_id, host_lock)
    mamba_pool.copy_from.assert_called_once()
