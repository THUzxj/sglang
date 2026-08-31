"""Unit tests for hybrid decode-side KV offload control logic."""

from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHybridDecodeKVCacheOffloadManager(CustomTestCase):
    @staticmethod
    def _manager(req_to_token_pool):
        manager = object.__new__(DecodeKVCacheOffloadManager)
        manager.is_hybrid_linear_kv_pool = True
        manager.req_to_token_pool = req_to_token_pool
        return manager

    def test_build_mamba_backup_transfer_from_extra_buffer_checkpoint(self):
        req_to_token_pool = SimpleNamespace(
            enable_mamba_extra_buffer=True,
            get_mamba_ping_pong_keep_idx=lambda req: 1,
        )
        manager = self._manager(req_to_token_pool)
        req = SimpleNamespace(
            mamba_pool_idx=torch.tensor(3),
            mamba_ping_pong_track_buffer=torch.tensor([7, 9]),
            mamba_last_track_seqlen=16,
            mamba_lazy_is_insert=True,
        )

        transfers = manager._build_mamba_backup_transfers(req, chunk_end=16)

        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].name, PoolName.MAMBA)
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(transfers[0].device_indices.tolist(), [9])

    def test_skip_mamba_backup_transfer_when_checkpoint_not_aligned(self):
        req_to_token_pool = SimpleNamespace(
            enable_mamba_extra_buffer=True,
            get_mamba_ping_pong_keep_idx=lambda req: 0,
        )
        manager = self._manager(req_to_token_pool)
        req = SimpleNamespace(
            mamba_pool_idx=torch.tensor(3),
            mamba_ping_pong_track_buffer=torch.tensor([7, 9]),
            mamba_last_track_seqlen=8,
            mamba_lazy_is_insert=True,
        )

        self.assertIsNone(manager._build_mamba_backup_transfers(req, chunk_end=16))

    def test_build_mamba_backup_transfer_from_finished_active_slot(self):
        req_to_token_pool = SimpleNamespace(
            enable_mamba_extra_buffer=False,
            mamba_pool=SimpleNamespace(replayssm_write_pos=None),
        )
        manager = self._manager(req_to_token_pool)
        req = SimpleNamespace(
            mamba_pool_idx=torch.tensor(5),
            finished=lambda: True,
            effective_kv_committed_len=lambda: 12,
        )

        transfers = manager._build_mamba_backup_transfers(req, chunk_end=12)

        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].device_indices.tolist(), [5])

    def test_free_extra_host_indices_skips_anchor_and_frees_mamba(self):
        freed = []
        mamba_host_pool = SimpleNamespace(free=lambda indices: freed.append(indices))
        host_group = object.__new__(HostPoolGroup)
        host_group.entry_map = {
            PoolName.KV: SimpleNamespace(
                is_primary_index_anchor=True,
                host_pool=SimpleNamespace(free=lambda indices: None),
            ),
            PoolName.MAMBA: SimpleNamespace(
                is_primary_index_anchor=False,
                host_pool=mamba_host_pool,
            ),
        }
        manager = object.__new__(DecodeKVCacheOffloadManager)
        manager.decode_host_mem_pool = host_group
        mamba_indices = torch.tensor([4])

        manager._free_extra_host_indices(
            [
                PoolTransfer(name=PoolName.KV, host_indices=torch.tensor([1, 2])),
                PoolTransfer(name=PoolName.MAMBA, host_indices=mamba_indices),
            ]
        )

        self.assertEqual(freed, [mamba_indices])
