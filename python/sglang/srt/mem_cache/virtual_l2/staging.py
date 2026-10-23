"""Physically bounded pinned staging buffers and event-held leases."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - lightweight allocator tests
    torch = None


class StagePoolExhausted(RuntimeError):
    pass


@dataclass
class StageLease:
    stage_pool: "StagePool"
    slot_ids: tuple[int, ...]
    direction: str
    allocation_generation: int
    last_cuda_event: Any = None
    released: bool = False

    def release(self) -> None:
        if self.released:
            raise RuntimeError("stage lease released twice")
        self.stage_pool.release(self)
        self.released = True


class StagePool:
    """A fixed ring of tensors; physical allocation never scales with L2 size."""

    def __init__(self, bytes_per_slot: int, ring_entries: int, *, direction: str = "D2H", dtype=None,
                 data_pattern: str = "zeros"):
        if bytes_per_slot <= 0 or ring_entries <= 0:
            raise ValueError("stage slot size and ring entries must be positive")
        if direction not in ("D2H", "H2D"):
            raise ValueError("direction must be D2H or H2D")
        if data_pattern not in ("zeros", "ones"):
            raise ValueError("data_pattern must be zeros or ones")
        self.bytes_per_slot = int(bytes_per_slot)
        self.ring_entries = int(ring_entries)
        self.direction = direction
        self._lock = threading.Lock()
        self._free = set(range(ring_entries))
        self._generation = 0
        self._active_generations = {}
        if torch is None:
            self.buffer = bytearray(ring_entries * bytes_per_slot)
        else:
            dtype = torch.uint8 if dtype is None else dtype
            pin = bool(torch.cuda.is_available())
            try:
                self.buffer = torch.empty((ring_entries, bytes_per_slot), dtype=dtype, pin_memory=pin)
            except RuntimeError:
                if pin:
                    raise RuntimeError("Virtual L2 requires a pinned host allocator")
                self.buffer = torch.empty((ring_entries, bytes_per_slot), dtype=dtype)
        if direction == "H2D":
            if hasattr(self.buffer, "zero_"):
                self.buffer.fill_(1 if data_pattern == "ones" else 0)
            elif data_pattern == "ones":
                self.buffer[:] = b"\x01" * len(self.buffer)

    @property
    def physical_bytes(self) -> int:
        if torch is None:
            return len(self.buffer)
        return self.buffer.numel() * self.buffer.element_size()

    @property
    def free_slots(self) -> int:
        with self._lock:
            return len(self._free)

    def acquire(self, count: int = 1) -> StageLease:
        if count < 1 or count > self.ring_entries:
            raise ValueError("invalid stage lease size")
        with self._lock:
            if len(self._free) < count:
                raise StagePoolExhausted(f"no {self.direction} stage lease for {count} slots")
            slots = tuple(sorted(self._free)[:count])
            self._free.difference_update(slots)
            self._generation += 1
            generation = self._generation
            for slot in slots:
                self._active_generations[slot] = generation
        return StageLease(self, slots, self.direction, generation)

    def release(self, lease: StageLease) -> None:
        if lease.stage_pool is not self or lease.direction != self.direction:
            raise ValueError("lease belongs to another stage pool")
        event = lease.last_cuda_event
        if event is not None and hasattr(event, "query") and not event.query():
            raise RuntimeError("cannot release a stage lease before its CUDA event completes")
        with self._lock:
            overlap = set(lease.slot_ids) & self._free
            if overlap:
                raise RuntimeError("stage slot was released or reused already")
            for slot in lease.slot_ids:
                if self._active_generations.get(slot) != lease.allocation_generation:
                    raise RuntimeError("stale Virtual L2 stage lease generation")
            self._free.update(lease.slot_ids)
            for slot in lease.slot_ids:
                self._active_generations.pop(slot, None)


def create_stage_pools(config, *, bytes_per_slot: int | None = None):
    """Create independent D2H/H2D rings within the configured RAM budget."""
    config.validate()
    total_budget = config.staging_bytes
    entries = config.stage_ring_entries
    if bytes_per_slot is None:
        # Split the budget between directions. The explicit floor also makes
        # the bound obvious to callers inspecting the pool metrics.
        bytes_per_slot = total_budget // (2 * entries)
    if bytes_per_slot <= 0:
        raise ValueError("virtual L2 staging budget is smaller than one ring slot")
    required = 2 * entries * bytes_per_slot
    if required > total_budget:
        raise ValueError("requested stage ring exceeds virtual L2 staging budget")
    return (
        StagePool(bytes_per_slot, entries, direction="D2H"),
        StagePool(bytes_per_slot, entries, direction="H2D", dtype=(
            torch.uint8 if torch is not None else None
        ), data_pattern=config.data_pattern),
    )
