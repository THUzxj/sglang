from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import sglang.srt.disaggregation.decode as decode_module
import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode


def test_radix_pause_transfers_mamba_checkpoint_ownership(monkeypatch):
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(page_size=16)
    cache = MagicMock()
    cache.supports_mamba.return_value = True
    cache.cache_controller.mem_pool_host.available_size.return_value = 100
    cache.components[
        scheduler_module.ComponentType.MAMBA
    ]._mamba_pool_host.available_size.return_value = 1
    cache.protect_cache_paused_node.return_value = "host-lock"
    scheduler.tree_cache = cache
    monkeypatch.setattr(
        scheduler_module,
        "get_exec",
        lambda: SimpleNamespace(
            mamba=SimpleNamespace(enable_linear_replayssm=False)
        ),
    )

    req = MagicMock()
    req.rid = "compact"
    req.origin_input_ids = list(range(31))
    req.output_ids = [11, 22]
    req.kv.kv_committed_len = 32
    req.owned_kv_len.return_value = 32
    req.skip_radix_cache_insert = False
    req.context_engineering_pause_node = None

    def release(actual_req, actual_cache, is_insert):
        assert actual_req is req and actual_cache is cache and is_insert
        assert actual_req.is_context_engineering_cache_parking
        actual_req.context_engineering_pause_node = 7

    monkeypatch.setattr(scheduler_module, "release_kv_cache", release)

    assert scheduler._park_compact_in_radix_cache(req)
    assert req.output_ids == [11, 22]
    assert req.context_engineering_pause_pending_token_id == 22
    assert req.context_engineering_pause_node == 7
    assert req.context_engineering_pause_host_lock == "host-lock"
    req.detach_gpu_state_for_cache_pause.assert_called_once_with(32)
    cache.protect_cache_paused_node.assert_called_once_with(7)


def test_local_cache_resume_uses_restored_kv_and_releases_host_lock(monkeypatch):
    mamba_match = SimpleNamespace(best_match_node=7)
    before = SimpleNamespace(
        device_indices=torch.tensor([3]),
        host_hit_length=1,
        last_device_node=5,
    )
    after = SimpleNamespace(
        device_indices=torch.tensor([3, 4]),
        host_hit_length=0,
        last_device_node=7,
    )
    matches = iter((mamba_match, before, after))
    monkeypatch.setattr(
        decode_module, "match_prefix_for_req", lambda *args, **kwargs: next(matches)
    )

    tree = MagicMock()
    tree.component_has_host_value_only.return_value = True
    tree.get_component_device_value.return_value = torch.tensor([8])
    cache = MagicMock()
    cache.tree_core = tree
    cache.mamba_evictable_size.return_value = 0
    cache.inc_lock_ref.side_effect = lambda node: SimpleNamespace(
        to_dec_params=lambda: ("lock", node)
    )
    cache.load_back.return_value = True
    cache.ready_to_load_host_cache.return_value = -1
    pool = MagicMock()
    pool.available_size.return_value = 1
    pool.enable_mamba_extra_buffer = False
    pool.mamba_allocator.available_size.return_value = 1
    pool.mamba_ckpt_pool = None
    pool.translate_mamba_indices.side_effect = lambda value: value

    req = SimpleNamespace(
        rid="compact",
        origin_input_ids=[1, 2],
        output_ids=[9],
        context_engineering_pause_committed_len=2,
        context_engineering_pause_node=7,
        context_engineering_pause_pending_token_id=9,
        context_engineering_pause_host_lock="host-lock",
        is_context_engineering_cache_paused=True,
        is_retracted=True,
        kv=SimpleNamespace(
            req_pool_idx=None,
            mamba_pool_idx=None,
            mamba_needs_clear=True,
            cache_protected_len=0,
        ),
        lock_receipt=None,
    )

    def prealloc(actual_req, *args):
        actual_req.kv.req_pool_idx = 5
        actual_req.kv.mamba_pool_idx = torch.tensor(4)
        return torch.empty((0,), dtype=torch.int64)

    queue = SimpleNamespace(
        req_to_token_pool=pool,
        tree_cache=cache,
        _pre_alloc_fill_len=lambda actual_req: 2,
        _pre_alloc=prealloc,
    )

    assert decode_module.DecodePreallocQueue.restore_cache_paused_req(queue, req)
    cache.load_back.assert_called_once_with(7, req=req, force=True)
    cache.dec_lock_ref.assert_called_once_with(5, ("lock", 5))
    cache.dec_host_lock_ref.assert_called_once_with(7, "host-lock")
    pool.write.assert_called_once()
    pool.mamba_pool.copy_from.assert_called_once()
    assert req.lock_receipt == ("lock", 7)
    assert req.kv.cache_protected_len == 2
    assert not req.is_context_engineering_cache_paused
