import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)


class TestDecodeHiCacheRestoreInterval(unittest.TestCase):
    def test_restore_writes_only_missing_request_interval(self):
        prefix = torch.tensor([10, 11], dtype=torch.int64)
        pm = DecodePrefixMatch(
            prefix_indices=prefix,
            l2_host_hit_length=2,
            l3_storage_hit_length=0,
            last_device_node=5,
            match_full_kv_only=True,
        )
        req = SimpleNamespace(
            rid="restore",
            origin_input_ids=[1, 2, 3, 4],
            last_node=5,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            lock_receipt="original-lock",
            swa_prefix_lock_released=False,
            kv=SimpleNamespace(req_pool_idx=9),
            prefix_indices=prefix,
        )
        dr = SimpleNamespace(
            req=req,
            prefix_match=pm,
            hicache_restored_node=None,
            hicache_restored_kv_indices=None,
            hicache_restore_lock_receipt=None,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )
        before = SimpleNamespace(
            device_indices=torch.tensor([10, 11, 12]),
            last_device_node=6,
            best_match_node=7,
            host_hit_length=1,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
        )
        after = SimpleNamespace(
            device_indices=torch.tensor([10, 11, 12, 13]),
            last_device_node=7,
        )
        cache = MagicMock()

        def init_load_back(params):
            self.assertEqual(req.last_node, 6)
            self.assertTrue(params.full_kv_only)
            self.assertTrue(params.force)
            # The cache may return a path overlapping the L1 prefix.
            return torch.tensor([11, 12, 13]), 7

        cache.init_load_back.side_effect = init_load_back
        cache.inc_lock_ref.return_value.to_dec_params.return_value = "restored-lock"
        queue = SimpleNamespace(tree_cache=cache)

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req",
            side_effect=[before, after],
        ) as match:
            self.assertTrue(
                DecodeHiCacheTransferMixin._try_hicache_queue_load_back(queue, dr)
            )
        self.assertEqual(match.call_count, 2)
        self.assertTrue(
            all(call.kwargs["update_req"] is False for call in match.call_args_list)
        )
        self.assertEqual(req.last_node, 5)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [12, 13])
        self.assertEqual(dr.hicache_restored_node, 7)

        DecodeHiCacheTransferMixin._commit_hicache_local_restore_to_req(queue, dr)
        cache.req_to_token_pool.write.assert_called_once()
        args = cache.req_to_token_pool.write.call_args.args
        self.assertEqual(args[0], (9, slice(2, 4)))
        self.assertEqual(args[1].tolist(), [12, 13])
        cache.dec_lock_ref.assert_called_once_with(
            5, "original-lock", skip_swa=False
        )
        self.assertEqual(req.last_node, 7)
        self.assertEqual(req.lock_receipt, "restored-lock")
        self.assertIsNone(dr.hicache_restored_node)


if __name__ == "__main__":
    unittest.main()
