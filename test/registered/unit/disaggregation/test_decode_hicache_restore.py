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
        )
        req = SimpleNamespace(
            rid="restore-superset",
            origin_input_ids=array("q", range(16)),
            output_ids=array("q"),
            is_context_engineering_cache_paused=True,
            req_pool_idx=0,
        )
        return SimpleNamespace(
            req=req,
            prefix_match=prefix_match,
            hicache_restored_kv_indices=None,
            hicache_restored_node=None,
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

        queued = queue._try_hicache_queue_load_back(decode_req)

        self.assertTrue(queued)
        torch.testing.assert_close(
            decode_req.hicache_restored_kv_indices,
            exact_after_load.device_indices[5:15],
        )
        self.assertEqual(len(decode_req.hicache_restored_kv_indices), 10)
        self.assertEqual(decode_req.hicache_restored_node, "restored-device-node")
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


if __name__ == "__main__":
    unittest.main()
