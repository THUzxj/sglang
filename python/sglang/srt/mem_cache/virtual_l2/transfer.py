"""Chunk planning and transfer orchestration for Virtual L2.

The adapter methods are intentionally small: model-specific pools can reuse
their existing CUDA copy kernels while this module owns leases, accounting,
and completion ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .metrics import VirtualL2Metrics
from .staging import StageLease, StagePool


@dataclass(frozen=True)
class StageChunk:
    logical_host_ids: Any
    physical_stage_ids: Any
    device_indices: Any
    pool_name: str
    target_layers: tuple[int, ...]
    byte_count: int


class StageTransferAdapter(Protocol):
    def bytes_for_chunk(self, chunk: StageChunk) -> int: ...

    def enqueue_d2h(self, chunk: StageChunk, stream) -> Any: ...

    def enqueue_h2d_layer(self, chunk: StageChunk, layer_id: int, stream) -> Any: ...


def split_chunks(logical_host_ids, device_indices, *, pool_name: str, bytes_per_unit: int,
                 max_units: int, target_layers=()) -> list[StageChunk]:
    """Split logical IDs without ever confusing them with stage IDs."""
    if bytes_per_unit <= 0 or max_units <= 0:
        raise ValueError("bytes_per_unit and max_units must be positive")
    host = logical_host_ids.tolist() if hasattr(logical_host_ids, "tolist") else list(logical_host_ids)
    device = device_indices.tolist() if hasattr(device_indices, "tolist") else list(device_indices)
    if len(host) != len(device):
        raise ValueError("host and device index lengths differ")
    chunks = []
    for start in range(0, len(host), max_units):
        end = min(start + max_units, len(host))
        chunks.append(StageChunk(
            logical_host_ids=host[start:end],
            physical_stage_ids=list(range(end - start)),
            device_indices=device[start:end],
            pool_name=pool_name,
            target_layers=tuple(target_layers),
            byte_count=(end - start) * bytes_per_unit,
        ))
    return chunks


class VirtualL2TransferEngine:
    def __init__(self, d2h_stage: StagePool | None, h2d_stage: StagePool | None, *, metrics=None):
        if d2h_stage is not None and d2h_stage.direction != "D2H":
            raise ValueError("d2h stage pool must use D2H direction")
        if h2d_stage is not None and h2d_stage.direction != "H2D":
            raise ValueError("h2d stage pool must use H2D direction")
        self.d2h_stage = d2h_stage
        self.h2d_stage = h2d_stage
        self.metrics = metrics or VirtualL2Metrics()

    @staticmethod
    def _wait_and_release(leases):
        if not leases:
            return
        lease = leases.pop(0)
        event = lease.last_cuda_event
        if event is not None and hasattr(event, "synchronize"):
            event.synchronize()
        if not lease.released:
            lease.release()

    def submit_device_to_host(self, chunks: list[StageChunk], adapter: StageTransferAdapter, stream=None):
        if self.d2h_stage is None:
            raise RuntimeError("D2H stage pool is not configured")
        leases = []
        try:
            for chunk in chunks:
                while True:
                    try:
                        lease = self.d2h_stage.acquire(1)
                        break
                    except Exception as exc:
                        if not leases or exc.__class__.__name__ != "StagePoolExhausted":
                            raise
                        self._wait_and_release(leases)
                leases.append(lease)
                physical = list(lease.slot_ids)
                mapped = StageChunk(chunk.logical_host_ids, physical, chunk.device_indices,
                                    chunk.pool_name, chunk.target_layers, chunk.byte_count)
                event = adapter.enqueue_d2h(mapped, stream)
                lease.last_cuda_event = event
                self.metrics.record("D2H", adapter.bytes_for_chunk(mapped))
                # A synchronous adapter may return no event. CUDA adapters keep
                # the lease alive by returning their completion event.
                if event is None:
                    lease.release()
            return leases
        except Exception:
            for lease in leases:
                if lease.released:
                    continue
                event = lease.last_cuda_event
                if event is not None and hasattr(event, "synchronize"):
                    event.synchronize()
                lease.release()
            raise

    def submit_host_to_device(self, chunks: list[StageChunk], adapter: StageTransferAdapter,
                              layer_ids, stream=None, on_layer_done=None):
        if self.h2d_stage is None:
            raise RuntimeError("H2D stage pool is not configured")
        leases = []
        layer_ids = tuple(layer_ids)
        # H2D sources are immutable templates, so one pinned slot can be read
        # by every chunk and layer. This keeps submission bounded even when a
        # request spans millions of logical pages.
        lease = self.h2d_stage.acquire(1)
        leases.append(lease)
        try:
            for layer_id in layer_ids:
                for chunk in chunks:
                    mapped = StageChunk(chunk.logical_host_ids, list(lease.slot_ids), chunk.device_indices,
                                        chunk.pool_name, (layer_id,), chunk.byte_count)
                    event = adapter.enqueue_h2d_layer(mapped, layer_id, stream)
                    if event is not None:
                        lease.last_cuda_event = event
                    self.metrics.record("H2D", adapter.bytes_for_chunk(mapped))
                if on_layer_done is not None:
                    on_layer_done(layer_id)
            # A synchronous/mock adapter returns no CUDA event, so the source
            # lease is safe to recycle as soon as submission has completed.
            if lease.last_cuda_event is None:
                lease.release()
            return leases
        except Exception:
            if not lease.released:
                event = lease.last_cuda_event
                if event is not None and hasattr(event, "synchronize"):
                    event.synchronize()
                lease.release()
            raise
