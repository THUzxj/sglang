import unittest
import threading

import torch
import numpy as np

from sglang.srt.mem_cache.virtual_l2 import (
    StagePool,
    StagePoolExhausted,
    VirtualHostPool,
    VirtualL2Config,
    VirtualPoolGeometry,
    VirtualPageAllocator,
    split_chunks,
)


class VirtualL2Test(unittest.TestCase):
    def test_hybrid_capacity_split_uses_unbounded_integer_arithmetic(self):
        from types import SimpleNamespace
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _split_virtual_l2_capacity,
        )

        # Device allocations can exceed signed 64-bit byte counts in hybrid
        # models. NumPy scalar multiplication used to overflow while splitting
        # a valid GiB budget and leave the KV pool with zero capacity.
        pools = (
            SimpleNamespace(get_kv_size_bytes=lambda: np.uint64(31 * 1024**3)),
            SimpleNamespace(get_kv_size_bytes=lambda: np.uint64(3 * 1024**3)),
        )
        kv_bytes, mamba_bytes = _split_virtual_l2_capacity(64, pools)
        self.assertEqual(kv_bytes + mamba_bytes, 64 * 1024**3)
        self.assertGreater(kv_bytes, 0)
        self.assertGreater(mamba_bytes, 0)
        self.assertAlmostEqual(kv_bytes / mamba_bytes, 31 / 3, places=5)

    def test_allocator_generation_and_double_free(self):
        allocator = VirtualPageAllocator(3)
        allocation = allocator.alloc(2)
        self.assertEqual(allocation.slots, (0, 1))
        self.assertEqual(allocator.available, 1)
        self.assertEqual(allocator.free([0]), 1)
        with self.assertRaises(RuntimeError):
            allocator.free([0])

    def test_legacy_chunks_split_large_multi_layer_payload(self):
        from sglang.srt.mem_cache.virtual_l2.staging import StagePool

        class DevicePool:
            layer_num = 3
            kv_buffer = [torch.zeros((8, 6), dtype=torch.uint8) for _ in range(3)]

        pool = VirtualHostPool(VirtualPoolGeometry("kv", 18, 1, 8),
                               layer_num=3, device_pool=DevicePool())
        pool.virtual_l2_d2h_stage = StagePool(12, 2, direction="D2H")
        chunks = pool._legacy_chunks([0], [2], "D2H")
        self.assertEqual([c.target_layers for c in chunks], [(0, 1), (2,)])
        self.assertEqual([c.byte_count for c in chunks], [12, 6])

    def test_pool_has_logical_capacity_without_payload(self):
        pool = VirtualHostPool(VirtualPoolGeometry("full", 128, 1, 4))
        indices = pool.alloc(3)
        pool.mark_ready(indices)
        self.assertEqual(pool.allocated_bytes, 384)
        self.assertEqual(pool.state(0), "READY_L2")

    def test_stage_ring_is_bounded(self):
        stage = StagePool(16, 1, direction="D2H")
        lease = stage.acquire()
        with self.assertRaises(StagePoolExhausted):
            stage.acquire()
        lease.release()
        self.assertEqual(stage.free_slots, 1)

    def test_chunk_byte_conservation(self):
        chunks = split_chunks([10, 11, 12, 13, 14], [1, 2, 3, 4, 5],
                              pool_name="mha", bytes_per_unit=32, max_units=2)
        self.assertEqual([c.byte_count for c in chunks], [64, 64, 32])
        self.assertEqual(sum(c.byte_count for c in chunks), 5 * 32)

    def test_config_guards(self):
        with self.assertRaises(ValueError):
            VirtualL2Config(enabled=True, capacity_gb=1, staging_gb=1,
                            disable_real_l3=False).validate()
        # Paced DMA is intentionally unimplemented until pacing is validated.
        with self.assertRaises(ValueError):
            VirtualL2Config(enabled=True, capacity_gb=1, staging_gb=1,
                            dma_mode="paced", bandwidth_gbps=100).validate()

    def test_h2d_uses_bounded_read_only_source(self):
        stage = StagePool(64, 1, direction="H2D")
        chunks = split_chunks(range(8), range(8), pool_name="kv",
                              bytes_per_unit=16, max_units=4)

        class Adapter:
            def __init__(self):
                self.calls = []

            def bytes_for_chunk(self, chunk):
                return chunk.byte_count

            def enqueue_d2h(self, chunk, stream):
                raise AssertionError("unexpected D2H")

            def enqueue_h2d_layer(self, chunk, layer_id, stream):
                self.calls.append((layer_id, tuple(chunk.physical_stage_ids)))
                return None

        adapter = Adapter()
        engine = __import__(
            "sglang.srt.mem_cache.virtual_l2.transfer", fromlist=["VirtualL2TransferEngine"]
        ).VirtualL2TransferEngine(None, stage)
        engine.submit_host_to_device(chunks, adapter, [0, 1])
        self.assertEqual(len(adapter.calls), 4)
        self.assertEqual(stage.free_slots, 1)

    def test_submission_failure_rolls_back_stage_leases(self):
        class FailingAdapter:
            def bytes_for_chunk(self, chunk):
                return chunk.byte_count

            def enqueue_d2h(self, chunk, stream):
                raise RuntimeError("copy failed")

            def enqueue_h2d_layer(self, chunk, layer_id, stream):
                return None

        d2h = StagePool(16, 2, direction="D2H")
        h2d = StagePool(16, 1, direction="H2D")
        engine = __import__(
            "sglang.srt.mem_cache.virtual_l2.transfer", fromlist=["VirtualL2TransferEngine"]
        ).VirtualL2TransferEngine(d2h, h2d)
        chunks = split_chunks([0, 1], [0, 1], pool_name="kv", bytes_per_unit=8, max_units=1)
        with self.assertRaises(RuntimeError):
            engine.submit_device_to_host(chunks, FailingAdapter(), None)
        self.assertEqual(d2h.free_slots, 2)
        # H2D is only touched if submission reaches the adapter; verify that a
        # completed synchronous H2D submission recycles its source slot.
        engine.submit_host_to_device(chunks, FailingAdapter(), [0], None)
        self.assertEqual(h2d.free_slots, 1)

    def test_mamba_pool_backs_up_and_restores_device_state(self):
        class Cache:
            def __init__(self):
                self.temporal = [torch.arange(6, dtype=torch.float32).reshape(2, 3)]
                self.conv = [torch.zeros(2, 2)]

        class DevicePool:
            def __init__(self):
                self.mamba_cache = Cache()

        device = DevicePool()
        pool = VirtualHostPool(
            VirtualPoolGeometry("mamba", 4, 1, 8), layer_num=1, device_pool=device
        )
        from sglang.srt.mem_cache.virtual_l2.staging import StagePool

        d2h_stage = StagePool(32, 1, direction="D2H")
        pool.virtual_l2_d2h_stage = d2h_stage
        chunk = split_chunks([0, 1], [0, 1], pool_name="mamba", bytes_per_unit=4,
                             max_units=2, target_layers=(0,))[0]
        chunk = type(chunk)(
            chunk.logical_host_ids, [0], chunk.device_indices,
            chunk.pool_name, chunk.target_layers, chunk.byte_count,
        )
        pool.backup_stage_chunk(chunk, stream=None)
        # Corrupt the device payload, then restore the synthetic staged bytes.
        device.mamba_cache.temporal[0].fill_(-1)
        h2d_stage = StagePool(32, 1, direction="H2D")
        pool.virtual_l2_h2d_stage = h2d_stage
        pool.load_stage_chunk(chunk, 0, stream=None)
        self.assertTrue(torch.equal(device.mamba_cache.temporal[0], torch.zeros(2, 3)))


    def test_legacy_host_pool_backup_and_layer_load(self):
        from types import SimpleNamespace
        from sglang.srt.mem_cache.virtual_l2.factory import build_virtual_host_pool

        class DevicePool:
            layer_num = 2
            start_layer = 0
            def __init__(self):
                self.kv_buffer = [torch.arange(8, dtype=torch.uint8).reshape(4, 2) + i * 8
                                  for i in range(2)]

        args = SimpleNamespace(
            virtual_l2_enable=True, virtual_l2_capacity_gb=1.0,
            virtual_l2_staging_gb=0.000001, virtual_l2_stage_ring_entries=2,
            virtual_l2_data_pattern="zeros", virtual_l2_dma_mode="native",
            virtual_l2_disable_real_l3=True, virtual_l2_allow_pd=False,
            virtual_l2_bandwidth_gbps=0.0, virtual_l2_fixed_latency_us=0.0,
            disaggregation_mode="null", disaggregation_enable=False,
        )
        device = DevicePool()
        pool = build_virtual_host_pool(device, page_size=1, server_args=args)
        host = pool.alloc(2)
        pool.backup_from_device_all_layer(device, host, torch.tensor([0, 1]), "direct")
        pool._drain_dma()
        self.assertEqual(pool.state(0), "READY_L2")
        self.assertEqual(pool.virtual_l2_metrics.d2h_bytes, 8)
        device.kv_buffer[0][0:2].zero_()
        pool.load_to_device_per_layer(device, host, torch.tensor([0, 1]), 0, "direct")
        pool._drain_dma()
        self.assertTrue(torch.equal(device.kv_buffer[0][0:2], torch.zeros((2, 2), dtype=torch.uint8)))
        self.assertEqual(pool.virtual_l2_metrics.h2d_bytes, 4)
        self.assertEqual(pool.virtual_l2_h2d_stage.free_slots, 2)
        pool.free(host)
        self.assertEqual(pool.available_size(), pool.size)

    def test_factory_shares_worker_bounded_stage_rings(self):
        from types import SimpleNamespace
        from sglang.srt.mem_cache.virtual_l2.factory import build_virtual_host_pool

        class DevicePool:
            layer_num = 1
            size = 4

            def __init__(self, width):
                self.kv_buffer = [torch.zeros((4, width), dtype=torch.uint8)]

        args = SimpleNamespace(
            virtual_l2_enable=True, virtual_l2_capacity_gb=1.0,
            virtual_l2_staging_gb=0.001, virtual_l2_stage_ring_entries=2,
            virtual_l2_data_pattern="zeros", virtual_l2_dma_mode="native",
            virtual_l2_disable_real_l3=True, virtual_l2_allow_pd=False,
            virtual_l2_bandwidth_gbps=0.0, virtual_l2_fixed_latency_us=0.0,
        )
        first = build_virtual_host_pool(DevicePool(2), page_size=1, pool_name="full", server_args=args)
        second = build_virtual_host_pool(DevicePool(4), page_size=1, pool_name="swa", server_args=args)
        self.assertIs(first.virtual_l2_d2h_stage, second.virtual_l2_d2h_stage)
        self.assertIs(first.virtual_l2_h2d_stage, second.virtual_l2_h2d_stage)
        d2h, h2d = first.virtual_l2_d2h_stage, first.virtual_l2_h2d_stage
        physical_bytes = d2h.physical_bytes + h2d.physical_bytes
        self.assertLessEqual(physical_bytes, int(args.virtual_l2_staging_gb * 1024**3))
        self.assertIn(first, d2h.owners)
        self.assertIn(second, d2h.owners)
        self.assertIn(first, h2d.owners)
        self.assertIn(second, h2d.owners)
        self.assertFalse(hasattr(args, "_virtual_l2_stage_pools"))

    def test_mamba_factory_rounds_sub_slot_capacity_up_to_one_slot(self):
        from types import SimpleNamespace
        from sglang.srt.mem_cache.virtual_l2.factory import build_virtual_host_pool

        class DevicePool:
            num_mamba_layers = 1
            size = 32
            mamba_layer_ids = [0]

            def _iter_transfer_state_tensors(self):
                return iter((
                    ("temporal", torch.zeros((1, 32, 4), dtype=torch.float32), 1),
                ))

        args = SimpleNamespace(
            virtual_l2_enable=True, virtual_l2_capacity_gb=1.0,
            virtual_l2_staging_gb=0.001, virtual_l2_stage_ring_entries=2,
            virtual_l2_data_pattern="zeros", virtual_l2_dma_mode="native",
            virtual_l2_disable_real_l3=True, virtual_l2_allow_pd=False,
            virtual_l2_bandwidth_gbps=0.0, virtual_l2_fixed_latency_us=0.0,
        )
        mamba = DevicePool()
        # The hybrid splitter may assign a tiny nonzero share. The Mamba
        # builder rounds it to the payload size of one logical state slot.
        from sglang.srt.mem_cache.virtual_l2.factory import _device_bytes_per_token

        slot_bytes = _device_bytes_per_token(mamba, page_size=1)
        pool = build_virtual_host_pool(
            mamba, page_size=1, pool_name="mamba", server_args=args,
            capacity_bytes=max(1, slot_bytes),
        )
        self.assertEqual(pool.geometry.codec, "MAMBA")
        self.assertEqual(pool.geometry.capacity_bytes, slot_bytes)

    def test_pool_retains_stage_lease_until_completion_event(self):
        from sglang.srt.mem_cache.virtual_l2.staging import StagePool

        pool = VirtualHostPool(VirtualPoolGeometry("kv", 4, 1, 4))
        indices = pool.alloc(1)
        pool.mark_pending(indices, "PENDING_D2H")
        stage = StagePool(8, 1, direction="D2H")
        lease = stage.acquire()

        class Event:
            def __init__(self):
                self.done = threading.Event()

            def synchronize(self):
                self.done.wait()

        event = Event()
        pool._schedule_dma_completion(event, [lease], indices, "READY_L2")
        self.assertEqual(stage.free_slots, 0)
        self.assertEqual(pool.state(0), "PENDING_D2H")
        event.done.set()
        pool._drain_dma()
        self.assertEqual(stage.free_slots, 1)
        self.assertEqual(pool.state(0), "READY_L2")


if __name__ == "__main__":
    unittest.main()
