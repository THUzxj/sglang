import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiCacheLoadBackEventSync(unittest.TestCase):
    def _make_cache(self, local_done, globally_done):
        finish_event = MagicMock()
        finish_event.query.return_value = local_done
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                layer_done_counter=SimpleNamespace(
                    events=[SimpleNamespace(finish_event=finish_event)]
                )
            ),
            loading_check=MagicMock(),
        )

        def all_reduce(event_done, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            event_done.fill_(int(globally_done))

        cache._all_reduce = MagicMock(side_effect=all_reduce)
        return cache

    def test_local_completion_waits_for_all_ranks(self):
        for cache_cls in (HiRadixCache, UnifiedRadixCache):
            with self.subTest(cache_cls=cache_cls.__name__):
                cache = self._make_cache(local_done=True, globally_done=False)

                done = cache_cls.is_load_back_event_done(cache, 0)

                self.assertFalse(done)
                cache._all_reduce.assert_called_once()
                cache.loading_check.assert_not_called()

    def test_all_ranks_complete_enter_loading_check(self):
        for cache_cls in (HiRadixCache, UnifiedRadixCache):
            with self.subTest(cache_cls=cache_cls.__name__):
                cache = self._make_cache(local_done=True, globally_done=True)

                done = cache_cls.is_load_back_event_done(cache, 0)

                self.assertTrue(done)
                cache._all_reduce.assert_called_once()
                cache.loading_check.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
