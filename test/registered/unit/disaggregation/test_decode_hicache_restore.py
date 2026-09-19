import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.base_prefix_cache import IncLockRefResult
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecodeHiCacheRestore(unittest.TestCase):
    def _make_queue(self):
        queue = DecodeHiCacheTransferMixin()
        queue.tree_cache = MagicMock()
        return queue

    def _make_decode_req(self):
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(5, dtype=torch.int64),
            l2_host_hit_length=10,
            l3_storage_hit_length=0,
            last_device_node="old-device-node",
            match_full_kv_only=True,
        )
        req = SimpleNamespace(
            rid="restore-superset",
            origin_input_ids=array("q", range(16)),
            output_ids=array("q"),
            is_context_engineering_cache_paused=True,
            req_pool_idx=0,
            last_node="original-device-node",
            prefix_indices=torch.arange(5, dtype=torch.int64),
            swa_uuid_for_lock=11,
            skip_lock_node_ids={ComponentType.MAMBA: {17}},
            swa_prefix_lock_released=False,
        )
        return SimpleNamespace(
            req=req,
            prefix_match=prefix_match,
            hicache_restored_kv_indices=None,
            hicache_restored_node=None,
            hicache_restored_lock_params=None,
            hicache_restore_status=HiCacheRestoreResult.PENDING,
        )

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_load_back_uses_exact_rematched_interval(self, mock_match_prefix):
        queue = self._make_queue()
        decode_req = self._make_decode_req()

        before_load = SimpleNamespace(
            device_indices=torch.arange(5, dtype=torch.int64),
            best_match_node="host-node",
            host_hit_length=10,
        )
        exact_after_load = SimpleNamespace(
            device_indices=torch.arange(100, 115, dtype=torch.int64),
            last_device_node="restored-device-node",
        )
        mock_match_prefix.side_effect = [before_load, exact_after_load]

        # The radix operation materializes a path larger than this request's
        # ten-token restore hole.  This was the shape mismatch seen in the
        # parallel-compaction experiment.
        queue.tree_cache.init_load_back.return_value = (
            torch.arange(18, dtype=torch.int64),
            "host-node",
        )
        queue.tree_cache.inc_lock_ref.return_value = IncLockRefResult(
            swa_uuid_for_lock=23,
            skip_lock_node_ids={ComponentType.MAMBA: {29}},
        )

        queued = queue._try_hicache_queue_load_back(decode_req)

        self.assertTrue(queued)
        torch.testing.assert_close(
            decode_req.hicache_restored_kv_indices,
            exact_after_load.device_indices[5:15],
        )
        self.assertEqual(len(decode_req.hicache_restored_kv_indices), 10)
        self.assertEqual(decode_req.hicache_restored_node, "restored-device-node")
        self.assertEqual(decode_req.req.last_node, "original-device-node")
        self.assertEqual(decode_req.req.swa_uuid_for_lock, 11)
        self.assertEqual(
            decode_req.hicache_restored_lock_params.swa_uuid_for_lock, 23
        )
        self.assertEqual(
            decode_req.hicache_restored_lock_params.skip_lock_node_ids,
            {ComponentType.MAMBA: {29}},
        )
        self.assertEqual(mock_match_prefix.call_count, 2)
        for call in mock_match_prefix.call_args_list:
            self.assertTrue(call.kwargs["match_full_kv_only"])
            self.assertFalse(call.kwargs["update_req"])
        init_params = queue.tree_cache.init_load_back.call_args.args[0]
        self.assertTrue(init_params.full_kv_only)
        queue.tree_cache.inc_lock_ref.assert_called_once_with(
            "restored-device-node"
        )

    def test_commit_rejects_wrong_restore_length_before_mutating_cache(self):
        queue = self._make_queue()
        decode_req = self._make_decode_req()
        decode_req.hicache_restored_kv_indices = torch.arange(18)
        decode_req.hicache_restored_node = "restored-device-node"

        with self.assertRaisesRegex(
            RuntimeError, "restored=18, expected=10"
        ):
            queue._commit_hicache_local_restore_to_req(decode_req)

        queue.tree_cache.dec_lock_ref.assert_not_called()
        queue.tree_cache.req_to_token_pool.write.assert_not_called()

    def test_mamba_host_only_hit_requires_local_restore(self):
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(16, dtype=torch.int64),
            l2_host_hit_length=0,
            l3_storage_hit_length=0,
            last_device_node="device-node",
            mamba_host_hit_length=1,
        )

        self.assertTrue(prefix_match.needs_local_restore)
        self.assertEqual(prefix_match.restore_token_count, 0)
    def test_cleanup_releases_only_the_temporary_restored_lock(self):
        queue = self._make_queue()
        decode_req = self._make_decode_req()
        decode_req.prefix_match.prefetch_registered = True
        decode_req.hicache_restored_node = "restored-device-node"
        decode_req.hicache_restored_lock_params = IncLockRefResult(
            swa_uuid_for_lock=23,
            skip_lock_node_ids={ComponentType.MAMBA: {29}},
        ).to_dec_params()
        restored_lock_params = decode_req.hicache_restored_lock_params

        queue._clean_hicache_prefetch_resources(decode_req)

        queue.tree_cache.dec_lock_ref.assert_called_once_with(
            "restored-device-node", restored_lock_params
        )
        queue.tree_cache.release_aborted_request.assert_called_once_with(
            decode_req.req.rid
        )
        self.assertFalse(decode_req.prefix_match.prefetch_registered)
        self.assertEqual(decode_req.req.last_node, "original-device-node")
        self.assertIsNone(decode_req.hicache_restored_node)
        self.assertIsNone(decode_req.hicache_restored_lock_params)

        queue._clean_hicache_prefetch_resources(decode_req)
        queue.tree_cache.release_aborted_request.assert_called_once()
        queue.tree_cache.dec_lock_ref.assert_called_once()

    def test_cleanup_rejects_restored_node_without_lock_params(self):
        queue = self._make_queue()
        decode_req = self._make_decode_req()
        decode_req.hicache_restored_node = "restored-device-node"

        with self.assertRaisesRegex(RuntimeError, "missing its lock parameters"):
            queue._clean_hicache_prefetch_resources(decode_req)

        queue.tree_cache.dec_lock_ref.assert_not_called()

    def test_commit_transfers_restored_lock_ownership_to_req(self):
        queue = self._make_queue()
        decode_req = self._make_decode_req()
        decode_req.hicache_restored_kv_indices = torch.arange(100, 110)
        decode_req.hicache_restored_node = "restored-device-node"
        restored_lock_params = IncLockRefResult(
            swa_uuid_for_lock=23,
            skip_lock_node_ids={ComponentType.MAMBA: {29}},
        ).to_dec_params()
        decode_req.hicache_restored_lock_params = restored_lock_params

        queue._commit_hicache_local_restore_to_req(decode_req)

        original_release = queue.tree_cache.dec_lock_ref.call_args
        self.assertEqual(original_release.args[0], "original-device-node")
        self.assertEqual(original_release.args[1].swa_uuid_for_lock, 11)
        self.assertEqual(
            original_release.args[1].skip_lock_node_ids,
            {ComponentType.MAMBA: {17}},
        )
        self.assertEqual(decode_req.req.last_node, "restored-device-node")
        self.assertEqual(decode_req.req.swa_uuid_for_lock, 23)
        self.assertEqual(
            decode_req.req.skip_lock_node_ids, {ComponentType.MAMBA: {29}}
        )
        self.assertIsNone(decode_req.hicache_restored_node)
        self.assertIsNone(decode_req.hicache_restored_lock_params)

        queue._clean_hicache_prefetch_resources(decode_req)
        queue.tree_cache.dec_lock_ref.assert_called_once()


if __name__ == "__main__":
    unittest.main()
