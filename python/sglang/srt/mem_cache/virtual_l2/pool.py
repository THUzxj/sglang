"""HostPool-compatible logical pool backed only by metadata."""

from __future__ import annotations

import threading
import logging

try:
    import torch
except ImportError:  # pragma: no cover - lightweight allocator tests
    torch = None

from .allocator import VirtualPageAllocator
from .geometry import VirtualPoolGeometry
from .metrics import VirtualL2Metrics

logger = logging.getLogger(__name__)


class VirtualHostPool:
    def __init__(self, geometry: VirtualPoolGeometry, *, layer_num: int = 0,
                 size_per_token: int | None = None, device_pool=None):
        self.layout = "page_first_direct"
        self.device = "cpu"
        self.dtype = getattr(device_pool, "dtype", torch.uint8 if torch else None)
        self.allocator = None
        self.start_layer = 0
        self.end_layer = layer_num
        self._pending_dma = []
        self.geometry = geometry
        self.size = geometry.capacity_units
        self.page_size = geometry.logical_page_size
        self.layer_num = layer_num
        self.size_per_token = size_per_token or geometry.bytes_per_logical_unit
        self.kv_buffer = None
        self.device_pool = device_pool
        self.logical_size = self.size
        self._allocator = VirtualPageAllocator(self.size)
        self._states: dict[int, str] = {}
        self.synthetic_cache_data = True
        self.is_virtual_l2_pool = True
        self.virtual_l2_state = {}
        self.virtual_l2_adapter = None
        self.virtual_l2_d2h_stage = None
        self.virtual_l2_h2d_stage = None
        self.virtual_l2_metrics = VirtualL2Metrics()
        self._state_lock = threading.RLock()

    def alloc(self, req_size: int):
        with self._state_lock:
            allocation = self._allocator.alloc(int(req_size))
            if allocation is None:
                return None
            if hasattr(allocation.slots, "start") and hasattr(allocation.slots, "stop"):
                for slot in range(allocation.slots.start, allocation.slots.stop):
                    self._states[slot] = "RESERVED"
            else:
                for slot in allocation.slots:
                    self._states[slot] = "RESERVED"
            if torch is not None:
                # Metadata-only pool: the returned tensor is still bounded by
                # the caller's request size, never by total L2 capacity.
                return torch.arange(
                    allocation.slots.start,
                    allocation.slots.stop,
                    dtype=torch.int64,
                )
            return list(allocation.slots)

    def free(self, indices) -> int:
        self._drain_dma()
        slots = [int(x) for x in indices.tolist()] if hasattr(indices, "tolist") else list(indices)
        with self._state_lock:
            pending = [slot for slot in slots if self._states.get(slot) in {"PENDING_D2H", "PENDING_H2D"}]
            if pending:
                raise RuntimeError(f"cannot free Virtual L2 slots while DMA is pending: {pending[:4]}")
            count = self._allocator.free(slots)
            for slot in slots:
                self._states.pop(slot, None)
            return count

    def available_size(self) -> int:
        return self._allocator.available

    def clear(self) -> None:
        self._drain_dma()
        with self._state_lock:
            if any(state in {"PENDING_D2H", "PENDING_H2D"} for state in self._states.values()):
                raise RuntimeError("cannot clear Virtual L2 pool while DMA is pending")
            self._allocator.clear()
            self._states.clear()

    @property
    def allocated_size(self) -> int:
        return self._allocator.allocated

    @property
    def allocated_bytes(self) -> int:
        return self.allocated_size * self.geometry.bytes_per_logical_unit

    def get_size_per_token(self) -> int:
        return self.size_per_token

    def mark_ready(self, indices, state: str = "READY_L2") -> None:
        with self._state_lock:
            for index in indices.tolist() if hasattr(indices, "tolist") else indices:
                index = int(index)
                if index not in self._states:
                    raise RuntimeError("cannot mark an unallocated virtual slot ready")
                # D2H and H2D can overlap for the same logical page: a page
                # may be restored while a newer backup completion is still
                # pending. Either DMA completion is a valid terminal state.
                if self._states[index] not in {
                    "RESERVED", "PENDING_D2H", "PENDING_H2D", "READY_L1", "READY_L2"
                }:
                    raise RuntimeError(
                        f"virtual slot {index} cannot become ready from {self._states[index]}"
                    )
                self._states[index] = state

    def mark_pending(self, indices, state: str) -> None:
        if state not in {"PENDING_D2H", "PENDING_H2D", "READY_L1", "INVALID"}:
            raise ValueError(f"invalid Virtual L2 state: {state}")
        with self._state_lock:
            values = [int(index) for index in (indices.tolist() if hasattr(indices, "tolist") else indices)]
            missing = [index for index in values if index not in self._states]
            if missing:
                raise RuntimeError("cannot transition an unallocated virtual slot")
            for index in values:
                index = int(index)
                self._states[index] = state

    def state(self, index: int) -> str:
        return self._states.get(int(index), "FREE")

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        # Logical IDs are metadata only. The transfer adapter performs the
        # actual remapping into a bounded StagePool.
        return host_indices, device_indices

    def _record_event(self, stream):
        if stream is None or torch is None or not torch.cuda.is_available():
            return None
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def _device_layers(self, layer_id):
        if self.device_pool is None:
            raise RuntimeError("VirtualHostPool is not attached to a device pool")
        iterator = getattr(self.device_pool, "_iter_transfer_state_tensors", None)
        if iterator is not None:
            return tuple(t[layer_id] for _, t, _ in iterator())
        # Public getters wait for the layer event this transfer produces.
        if hasattr(self.device_pool, "kv_buffer"):
            return (self.device_pool.kv_buffer[layer_id],)
        if hasattr(self.device_pool, "k_buffer"):
            buffers = [self.device_pool.k_buffer[layer_id]]
            if hasattr(self.device_pool, "v_buffer"):
                buffers.append(self.device_pool.v_buffer[layer_id])
            for field in ("k_scale_buffer", "v_scale_buffer"):
                values = getattr(self.device_pool, field, None)
                if values is not None:
                    buffers.append(values[layer_id])
            return tuple(buffers)
        getter = getattr(self.device_pool, "get_kv_buffer", None)
        if getter is None:
            cache = getattr(self.device_pool, "mamba_cache", None)
            if cache is None:
                raise NotImplementedError("device pool has no KV buffer accessor")
            # Mamba state has one temporal tensor and zero or more conv state
            # tensors per layer; expose each component to the same byte packer.
            return tuple(
                [cache.temporal[layer_id],
                 *(conv[layer_id] for conv in getattr(cache, "conv", ()))]
            )
        buffers = getter(layer_id + getattr(self.device_pool, "start_layer", 0))
        return buffers if isinstance(buffers, (tuple, list)) else (buffers,)

    def backup_stage_chunk(self, chunk, stream=None):
        if torch is None:
            raise RuntimeError("Virtual L2 DMA requires torch")
        stage = self.virtual_l2_d2h_stage.buffer[chunk.physical_stage_ids[0]]
        dst = stage.view(torch.uint8)
        offset = 0
        for layer_id in chunk.target_layers:
            for source in self._device_layers(layer_id):
                rows = source.index_select(0, torch.as_tensor(chunk.device_indices, device=source.device))
                raw = rows.contiguous().view(torch.uint8).view(-1)
                end = offset + raw.numel()
                if end > dst.numel():
                    raise ValueError("Virtual L2 stage slot is smaller than the DMA payload")
                dst[offset:end].copy_(raw, non_blocking=stream is not None)
                offset = end
        return self._record_event(stream)

    def load_stage_chunk(self, chunk, layer_id, stream=None):
        if torch is None:
            raise RuntimeError("Virtual L2 DMA requires torch")
        source = self.virtual_l2_h2d_stage.buffer[chunk.physical_stage_ids[0]].view(torch.uint8)
        offset = 0
        indices = torch.as_tensor(chunk.device_indices)
        for target in self._device_layers(layer_id):
            target_indices = indices.to(target.device)
            rows = target.index_select(0, target_indices)
            count = rows.numel() * rows.element_size()
            raw = source[offset : offset + count].view(rows.dtype).view_as(rows)
            if raw.device != target.device:
                raw = raw.to(target.device, non_blocking=stream is not None)
            target.index_copy_(0, target_indices, raw)
            offset += count
        return self._record_event(stream)

    def backup_from_device_all_layer_physical(self, *args, **kwargs):
        raise RuntimeError(
            "VirtualHostPool requires L2TransferEngine.submit_virtual_device_to_host; "
            "logical host IDs must be remapped into staging slots first"
        )

    def load_to_device_per_layer_physical(self, *args, **kwargs):
        raise RuntimeError(
            "VirtualHostPool requires L2TransferEngine.submit_virtual_host_to_device; "
            "logical host IDs must be remapped into staging slots first"
        )

    def get_ksize_per_token(self):
        return self.size_per_token

    def is_stride_page_aligned(self, page_size_bytes=4096):
        return False

    def destroy(self):
        self._drain_dma()
        self.clear()

    def _drain_dma(self):
        with self._state_lock:
            pending = list(self._pending_dma)
        for record in pending:
            self._finish_dma(record)

    def _finish_dma(self, record):
        while True:
            with self._state_lock:
                if not any(item is record for item in self._pending_dma):
                    return
                if record["finishing"]:
                    done = record["done"]
                else:
                    record["finishing"] = True
                    done = None
            if done is None:
                break
            done.wait()
        try:
            if record["event"] is not None:
                record["event"].synchronize()
            for lease in record["leases"]:
                if not lease.released:
                    lease.release()
            self.mark_ready(record["indices"], record["state"])
        except Exception:
            with self._state_lock:
                record["finishing"] = False
                record["done"].set()
            raise
        with self._state_lock:
            self._pending_dma[:] = [item for item in self._pending_dma if item is not record]
            record["done"].set()

    def _schedule_dma_completion(self, event, leases, indices, state):
        record = {
            "event": event,
            "leases": leases,
            "indices": indices,
            "state": state,
            "finishing": False,
            "done": threading.Event(),
        }
        with self._state_lock:
            self._pending_dma.append(record)
        if event is None:
            self._finish_dma(record)
            return

        def complete():
            try:
                self._finish_dma(record)
            except Exception:
                logger.exception("Virtual L2 transfer completion failed")

        threading.Thread(target=complete, name="virtual-l2-completion", daemon=True).start()

    def _legacy_chunks(self, host_indices, device_indices, direction, layer_id=None):
        from .transfer import split_chunks

        stage = (self.virtual_l2_d2h_stage if direction == "D2H"
                 else self.virtual_l2_h2d_stage)
        layers = tuple(range(self.layer_num)) if layer_id is None else (layer_id,)
        per_layer_bytes = {}
        for layer in layers:
            tensors = self._device_layers(layer)
            if hasattr(self.device_pool, "_iter_transfer_state_tensors"):
                per_layer_bytes[layer] = sum(
                    t.select(1, 0).numel() * t.element_size() for t in tensors
                )
            else:
                per_layer_bytes[layer] = sum(
                    t[0].numel() * t.element_size() for t in tensors
                )
        # A full KV logical unit can exceed one staging slot on large models.
        # Transfer layer groups independently; each group still contains the
        # same token IDs, so the resulting metadata allocation is completed
        # only after every group has been submitted. This keeps the staging
        # ring bounded without requiring a model-specific tensor packer.
        groups = []
        if layer_id is not None:
            groups = [layers]
        else:
            current = []
            current_bytes = 0
            for layer in layers:
                layer_bytes = per_layer_bytes[layer]
                if layer_bytes > stage.bytes_per_slot:
                    raise ValueError(
                        "Virtual L2 stage slot cannot hold one complete layer"
                    )
                if current and current_bytes + layer_bytes > stage.bytes_per_slot:
                    groups.append(tuple(current))
                    current, current_bytes = [], 0
                current.append(layer)
                current_bytes += layer_bytes
            if current:
                groups.append(tuple(current))

        chunks = []
        for group in groups:
            unit_bytes = sum(per_layer_bytes[layer] for layer in group)
            units = stage.bytes_per_slot // unit_bytes
            if units < 1:
                raise ValueError("Virtual L2 stage slot cannot hold one complete logical unit")
            chunks.extend(split_chunks(
                host_indices, device_indices, pool_name=self.geometry.pool_name,
                bytes_per_unit=unit_bytes, max_units=units,
                target_layers=group,
            ))
        return chunks

    def _legacy_adapter(self):
        from .adapters.mamba import MambaStageAdapter
        from .adapters.mha import MHAStageAdapter

        if hasattr(self.device_pool, "_iter_transfer_state_tensors"):
            return MambaStageAdapter(self)
        return MHAStageAdapter(self)

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        from .transfer import VirtualL2TransferEngine

        stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        chunks = self._legacy_chunks(host_indices, device_indices, "D2H")
        previous = self._snapshot_states(host_indices)
        self.mark_pending(host_indices, "PENDING_D2H")
        engine = VirtualL2TransferEngine(self.virtual_l2_d2h_stage, None, metrics=self.virtual_l2_metrics)
        try:
            leases = engine.submit_device_to_host(chunks, self._legacy_adapter(), stream)
        except Exception:
            self._restore_states(previous)
            raise
        self._schedule_dma_completion(
            self._record_event(stream), leases, host_indices, "READY_L2"
        )

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        from .transfer import VirtualL2TransferEngine

        stream = torch.cuda.current_stream() if torch.cuda.is_available() else None
        chunks = self._legacy_chunks(host_indices, device_indices, "H2D", layer_id)
        previous = self._snapshot_states(host_indices)
        self.mark_pending(host_indices, "PENDING_H2D")
        engine = VirtualL2TransferEngine(None, self.virtual_l2_h2d_stage, metrics=self.virtual_l2_metrics)
        try:
            leases = engine.submit_host_to_device(
                chunks, self._legacy_adapter(), (layer_id,), stream
            )
        except Exception:
            self._restore_states(previous)
            raise
        self._schedule_dma_completion(
            self._record_event(stream), leases, host_indices, "READY_L1"
        )

    def _snapshot_states(self, indices):
        values = indices.tolist() if hasattr(indices, "tolist") else indices
        with self._state_lock:
            return {int(index): self._states[int(index)] for index in values}

    def _restore_states(self, states):
        with self._state_lock:
            for index, state in states.items():
                if index in self._states:
                    self._states[index] = state
