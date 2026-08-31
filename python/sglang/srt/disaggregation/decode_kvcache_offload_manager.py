from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_kv_host_pool,
    build_pool_entry,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, MambaPoolHost
from sglang.srt.mem_cache.pool_host.common import get_allocator_type
from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def _pool_size_bytes(pool) -> int:
    size_bytes = pool.get_kv_size_bytes()
    return sum(size_bytes) if isinstance(size_bytes, tuple) else size_bytes


def _split_hicache_size(hicache_size: int, pools: tuple) -> tuple[float, ...]:
    pool_sizes = [_pool_size_bytes(pool) for pool in pools]
    total_size = sum(pool_sizes)
    if total_size <= 0:
        raise ValueError("Cannot split HiCache host size across empty device pools.")
    return tuple(hicache_size * size / total_size for size in pool_sizes)


class DecodeKVCacheOffloadManager:
    """Manage decode-side KV cache offloading lifecycle and operations."""

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = server_args.page_size
        self.server_args = server_args
        self.request_counter = 0
        self.tree_cache = tree_cache
        env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
        if env_stride is None or env_stride <= 0:
            self.offload_stride = self.page_size
        else:
            self.offload_stride = max(
                self.page_size, (env_stride // self.page_size) * self.page_size
            )
        kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
        allocator_type = get_allocator_type(server_args)
        self.is_hybrid_linear_kv_pool = isinstance(kv_cache, HybridLinearKVPool)
        self.mamba_pool_host = None
        transfer_layer_num = None

        logger.info(f"kv cache dtype: {kv_cache}")

        if self.is_hybrid_linear_kv_pool:
            self.decode_host_mem_pool, transfer_layer_num = (
                self._init_hybrid_host_mem_pool(kv_cache, allocator_type)
            )
        elif isinstance(kv_cache, MHATokenToKVPool):
            self.decode_host_mem_pool = get_mha_host_pool_cls(kv_cache)(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=allocator_type,
            )
        elif isinstance(kv_cache, MLATokenToKVPool):
            self.decode_host_mem_pool = MLATokenToKVPoolHost(
                kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                allocator_type=allocator_type,
            )
        else:
            raise ValueError("Unsupported KV cache type for decode offload")

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        hicache_storage_backend_extra_config = {}
        if server_args.hicache_storage_backend_extra_config:
            try:
                hicache_storage_backend_extra_config = json.loads(
                    server_args.hicache_storage_backend_extra_config
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid hicache storage backend extra config JSON: {e}"
                )

        controller_cls = (
            HybridCacheController
            if self.is_hybrid_linear_kv_pool
            else HiCacheController
        )
        controller_kwargs = {}
        if self.is_hybrid_linear_kv_pool:
            controller_kwargs["transfer_layer_num"] = transfer_layer_num
        self.cache_controller = controller_cls(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            mem_pool_host=self.decode_host_mem_pool,
            page_size=self.page_size,
            tp_group=tp_group,
            io_backend=server_args.hicache_io_backend,
            load_cache_event=threading.Event(),
            storage_backend=server_args.hicache_storage_backend,
            model_name=server_args.served_model_name,
            storage_backend_extra_config=hicache_storage_backend_extra_config,
            **controller_kwargs,
        )
        if self.is_hybrid_linear_kv_pool:
            kv_cache.register_layer_transfer_counter(
                self.cache_controller.layer_done_counter
            )
            if hasattr(self.req_to_token_pool, "register_layer_transfer_counter"):
                self.req_to_token_pool.register_layer_transfer_counter(
                    self.cache_controller.layer_done_counter
                )

        self.ongoing_offload = {}
        self.ongoing_backup = {}
        self.offloaded_state = {}
        self.offload_inflight = {}
        logger.info("Enable offload kv cache for decode side")

    def _init_hybrid_host_mem_pool(
        self, kv_cache: HybridLinearKVPool, allocator_type: str
    ) -> tuple[HostPoolGroup, int]:
        full_kv_pool = kv_cache.full_kv_pool
        mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        mamba_map = getattr(self.req_to_token_pool, "mamba_map", None)
        if mamba_pool is None or mamba_allocator is None or mamba_map is None:
            raise ValueError(
                "HybridLinearKVPool decode offload requires HybridReqToTokenPool "
                "with mamba_pool, mamba_allocator, and mamba_map."
            )

        full_host_size = None
        mamba_host_size = 0
        if self.server_args.hicache_size > 0:
            full_host_size, mamba_host_size = _split_hicache_size(
                self.server_args.hicache_size,
                (full_kv_pool, mamba_pool),
            )

        full_host_pool = build_kv_host_pool(
            kv_pool=full_kv_pool,
            page_size=self.page_size,
            server_args=self.server_args,
            use_mla=kv_cache.use_mla,
            host_size=full_host_size,
        )
        self.mamba_pool_host = MambaPoolHost(
            mamba_pool,
            self.server_args.hicache_ratio,
            mamba_host_size,
            allocator_type=allocator_type,
            layout=self.server_args.hicache_mem_layout,
        )

        full_layer_mapping = dict(kv_cache.full_attention_layer_id_mapping)
        mamba_layer_mapping = dict(mamba_map)
        transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)
        host_pool_group = HostPoolGroup(
            [
                build_pool_entry(
                    name=PoolName.KV,
                    host_pool=full_host_pool,
                    device_pool=full_kv_pool,
                    layer_mapping=full_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                    is_anchor=True,
                ),
                build_pool_entry(
                    name=PoolName.MAMBA,
                    host_pool=self.mamba_pool_host,
                    device_pool=mamba_pool,
                    layer_mapping=mamba_layer_mapping,
                    transfer_layer_num=transfer_layer_num,
                    device_alloc_fn=mamba_allocator.alloc,
                    device_free_fn=mamba_allocator.free,
                ),
            ]
        )
        return host_pool_group, transfer_layer_num

    def release_host_resources(self) -> None:
        self.decode_host_mem_pool.destroy()

    def _mark_offload_started(self, rid):
        self.offload_inflight[rid] = self.offload_inflight.get(rid, 0) + 1

    def _mark_offload_finished(self, rid):
        count = self.offload_inflight.get(rid, 0)
        if count <= 1:
            self.offload_inflight.pop(rid, None)
        else:
            self.offload_inflight[rid] = count - 1

    def _has_inflight_offload(self, rid):
        return self.offload_inflight.get(rid, 0) > 0

    def offload_kv_cache(self, req) -> bool:
        """Offload incremental KV cache for decode side."""

        if self.cache_controller is None or self.decode_host_mem_pool is None:
            return False

        if req.req_pool_idx == -1 or len(req.output_ids) == 0:
            return False

        token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        if token_indices.dim() == 0 or token_indices.numel() == 0:
            return False

        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        prefill_offloaded_len = (
            len(req.origin_input_ids) // self.page_size * self.page_size
        )
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_hashes = self._compute_prefix_hash(
                req.origin_input_ids[:prefill_offloaded_len]
            )
            last_prefill_hash = (
                prefill_hashes[-1] if prefill_offloaded_len > 0 else None
            )
            state = OffloadedState(
                prefill_len=prefill_offloaded_len,
                inc_len=0,
                last_hash=last_prefill_hash,
            )
            self.offloaded_state[req.rid] = state
        incremental_total = len(all_tokens) - state.prefill_len
        incremental_new = incremental_total - state.inc_len
        incremental_aligned_len = (
            incremental_new // self.offload_stride * self.offload_stride
        )

        if incremental_aligned_len == 0:
            return False

        # Extract incremental tokens and indices for the newly available chunk
        start = state.prefill_len + state.inc_len
        end = start + incremental_aligned_len
        incremental_tokens = all_tokens[start:end]
        incremental_indices = token_indices[start:end]

        # Prefill-aligned GPU slots are freed at request finish in
        # _release_finished_req, NOT here. The decoding request
        # continues to attend to those slots via req_to_token; freeing
        # them mid-decode races with concurrent admission, which can
        # reuse the slots and produce cross-pollinated KV reads.

        # Asynchronously offload incremental KV cache from device to host
        self.request_counter += 1
        ack_id = self.request_counter
        extra_pools = self._build_mamba_backup_transfers(req, end)
        write_kwargs = (
            {"extra_pools": extra_pools} if self.is_hybrid_linear_kv_pool else {}
        )
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
            **write_kwargs,
        )
        if host_indices is None:
            logger.error(f"Not enough host memory for request {req.rid}")
            return False

        self._mark_offload_started(req.rid)
        self.ongoing_offload[ack_id] = (
            req,
            host_indices,
            incremental_tokens,
            time.time(),
            start,
            end,
            extra_pools,
        )
        state.inc_len += incremental_aligned_len
        return True

    def _build_mamba_backup_transfers(
        self, req: Req, chunk_end: int
    ) -> Optional[list[PoolTransfer]]:
        if not self.is_hybrid_linear_kv_pool:
            return None
        if req.mamba_pool_idx is None:
            return None

        mamba_device_index = None
        enable_extra_buffer = getattr(
            self.req_to_token_pool, "enable_mamba_extra_buffer", False
        )
        if enable_extra_buffer:
            if not getattr(req, "mamba_lazy_is_insert", True):
                return None
            if req.mamba_ping_pong_track_buffer is None:
                return None
            if req.mamba_last_track_seqlen != chunk_end:
                return None
            keep_idx = self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
            mamba_device_index = req.mamba_ping_pong_track_buffer[keep_idx]
            if int(mamba_device_index.item()) < 0:
                return None
        elif req.finished() and chunk_end == req.effective_kv_committed_len():
            write_pos_buf = getattr(
                self.req_to_token_pool.mamba_pool, "replayssm_write_pos", None
            )
            if (
                write_pos_buf is not None
                and int(write_pos_buf[req.mamba_pool_idx].item()) != 0
            ):
                return None
            mamba_device_index = req.mamba_pool_idx

        if mamba_device_index is None:
            return None

        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                device_indices=mamba_device_index.reshape(1),
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def check_offload_progress(self):
        """Check the progress of offload from device to host and backup from host to storage."""
        cc = self.cache_controller

        qsizes = torch.tensor(
            [
                len(cc.ack_write_queue),
                cc.ack_backup_queue.qsize(),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )

        n_write, n_backup = map(int, qsizes.tolist())
        self._check_offload_progress(n_write)
        self._check_backup_progress(n_backup)

    def _check_offload_progress(self, finish_count):
        """Check the progress of offload from device to host."""
        while finish_count > 0:
            ack = self.cache_controller.ack_write_queue.pop(0)
            ack.finish_event.synchronize()
            for ack_id in ack.node_ids:
                (
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    start,
                    end,
                    extra_pools,
                ) = self.ongoing_offload.pop(ack_id)

                self._mark_offload_finished(req.rid)
                prior_hash = (
                    self.offloaded_state[req.rid].last_hash
                    if req.rid in self.offloaded_state
                    else None
                )
                last_hash = self._trigger_backup(
                    req,
                    host_indices,
                    incremental_tokens,
                    start_time,
                    prior_hash,
                    extra_pools,
                )
                if req.rid in self.offloaded_state:
                    self.offloaded_state[req.rid].last_hash = last_hash

                if req.finished() and not self._has_inflight_offload(req.rid):
                    state = self.offloaded_state.get(req.rid)
                    start_offset = state.prefill_len if state is not None else start
                    self._release_finished_req(req, start_offset)
            finish_count -= 1

    def _release_finished_req(self, req: Req, start_offset: int):
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (e.g. tree_cache.protected_size_
        # double-decrement, host pool double-free).
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return

        kv_committed_len = req.effective_kv_committed_len()

        # Free the prefill-aligned slots. Previously this was done
        # eagerly in offload_kv_cache (mid-decode), which raced with
        # concurrent admission. Now consolidated here at request
        # finish, where the request is guaranteed to no longer attend
        # to those slots.
        state = self.offloaded_state.get(req.rid)
        if state is not None and state.prefill_len > 0:
            prefill_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : state.prefill_len
            ]
            self.token_to_kv_pool_allocator.free(prefill_indices)
        start = start_offset
        end = kv_committed_len
        # Free the incremental part of the request (DSA-aware)
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, start:end]
        self.token_to_kv_pool_allocator.free(kv_indices)

        # Free over-allocated KV cache slots (e.g. from speculative decoding v2).
        # Without spec v2, start_p == end_p so this is a no-op.
        start_p, end_p = kv_committed_len, req.kv.kv_allocated_len
        if self.page_size > 1:
            start_p = ceil_align(start_p, self.page_size)
        if start_p < end_p:
            overalloc_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start_p:end_p
            ]
            self.token_to_kv_pool_allocator.free(overalloc_indices)

        if (
            self.is_hybrid_linear_kv_pool
            and hasattr(self.req_to_token_pool, "free_mamba_cache")
            and req.mamba_pool_idx is not None
        ):
            self.req_to_token_pool.free_mamba_cache(req)
        self.req_to_token_pool.free(req)
        req.kv = None
        self.tree_cache.protected_size_ -= len(req.prefix_indices)
        if req.rid in self.offloaded_state:
            del self.offloaded_state[req.rid]

    def _check_backup_progress(self, finish_count):
        """Check the progress of backup from host to storage."""
        for _ in range(finish_count):
            storage_operation = self.cache_controller.ack_backup_queue.get()
            ack_id = storage_operation.id
            req_id, host_indices, start_time, extra_pools = self.ongoing_backup.pop(
                ack_id
            )

            # Release host memory
            self.decode_host_mem_pool.free(host_indices)
            self._free_extra_host_indices(extra_pools)

            logger.debug(
                f"Finished backup request {req_id}, free host memory, len:{len(host_indices)}, cost time:{time.time() - start_time:.2f} seconds."
            )

    def _free_extra_host_indices(
        self, extra_pools: Optional[list[PoolTransfer]]
    ) -> None:
        if not extra_pools or not isinstance(
            self.decode_host_mem_pool, HostPoolGroup
        ):
            return
        for transfer in extra_pools:
            if (
                transfer.host_indices is None
                or transfer.indices_from_pool is not None
            ):
                continue
            entry = self.decode_host_mem_pool.entry_map.get(transfer.name)
            if entry is not None and not entry.is_primary_index_anchor:
                entry.host_pool.free(transfer.host_indices)
                transfer.host_indices = None

    def _trigger_backup(
        self,
        req,
        host_indices,
        incremental_tokens,
        start_time,
        prior_hash,
        extra_pools=None,
    ):
        """Trigger async backup from host to storage."""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        if extra_pools and page_hashes:
            for transfer in extra_pools:
                if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    transfer.keys = [page_hashes[-1]]
        write_storage_kwargs = (
            {"extra_pools": extra_pools} if self.is_hybrid_linear_kv_pool else {}
        )
        ack_id = self.cache_controller.write_storage(
            host_indices,
            incremental_tokens,
            hash_value=page_hashes,
            **write_storage_kwargs,
        )
        self.ongoing_backup[ack_id] = (
            req.rid,
            host_indices,
            start_time,
            extra_pools,
        )
        return page_hashes[-1] if len(page_hashes) > 0 else prior_hash

    def _compute_prefix_hash(self, tokens, prior_hash=""):
        page_hashes = []
        last_hash = prior_hash
        for offset in range(0, len(tokens), self.page_size):
            page_tokens = tokens[offset : offset + self.page_size]
            last_hash = self.cache_controller.get_hash_str(page_tokens, last_hash)
            page_hashes.append(last_hash)
        return page_hashes

    def finalize_release_on_finish(self, req: Req):
        """Free any remaining tail KV that was not offloaded due to non-aligned length."""
        # ReqToTokenPool.free sets req_pool_idx to None on release, so
        # guard against both sentinels here.
        if req.req_pool_idx is None or req.req_pool_idx == -1:
            return
        state = self.offloaded_state.get(req.rid)
        if state is None:
            prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
            inc_len = 0
        else:
            prefill_len = state.prefill_len
            inc_len = state.inc_len
        # Prefill-aligned slots are freed by _release_finished_req. Make
        # sure state exists so it can find prefill_len.
        if state is None:
            self.offloaded_state[req.rid] = OffloadedState(
                prefill_len=prefill_len, inc_len=0, last_hash=None
            )
        if self._has_inflight_offload(req.rid):
            return
        start_offset = prefill_len
        self._release_finished_req(req, start_offset)
