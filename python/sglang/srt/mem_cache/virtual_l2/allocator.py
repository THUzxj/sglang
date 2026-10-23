"""Bounded page allocator for logical Virtual L2 slots."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence


class SlotRange(Sequence):
    """Compact immutable sequence for a contiguous allocation.

    It deliberately does not retain one Python integer per slot.  Iteration
    remains available for legacy callers, while consumers that need a tensor
    can choose when (and where) to materialize the IDs.
    """

    __slots__ = ("start", "stop")

    def __init__(self, start: int, stop: int):
        self.start = int(start)
        self.stop = int(stop)

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return tuple(self)[index]
            return SlotRange(self.start + start, self.start + stop)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.start + index

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __repr__(self):
        return f"SlotRange({self.start}, {self.stop})"

    def __eq__(self, other):
        if isinstance(other, SlotRange):
            return (self.start, self.stop) == (other.start, other.stop)
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(other)
        return NotImplemented


@dataclass(frozen=True)
class Allocation:
    slots: SlotRange
    generation: int


class VirtualPageAllocator:
    """Range/bitmap allocator that stores metadata only, never cache payload."""

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        # Keep interval metadata rather than one entry per logical slot.  A
        # 1 TiB logical cache may contain billions of slots.
        self._free_ranges = [(0, capacity)]
        self._allocated_ranges = []  # (start, end, generation)
        self._next_generation = 0
        self._allocated = 0

    @property
    def allocated(self) -> int:
        return self._allocated

    @property
    def available(self) -> int:
        return self.capacity - self._allocated

    def alloc(self, count: int) -> Allocation | None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return Allocation(SlotRange(0, 0), 0)
        if count > self.available:
            return None
        start = end = None
        for range_start, range_end in self._free_ranges:
            if range_end - range_start >= count:
                start, end = range_start, range_start + count
                break
        if start is None:
            return None
        new_free = []
        for range_start, range_end in self._free_ranges:
            if (range_start, range_end) == (start, start + count):
                continue
            if range_start <= start < range_end:
                if range_start < start:
                    new_free.append((range_start, start))
                if end < range_end:
                    new_free.append((end, range_end))
            else:
                new_free.append((range_start, range_end))
        self._free_ranges = sorted(new_free)
        self._next_generation += 1
        generation = self._next_generation
        self._allocated_ranges.append((start, end, generation))
        self._allocated += count
        return Allocation(SlotRange(start, end), generation)

    def free(self, slots, generation: int | None = None) -> int:
        if isinstance(slots, SlotRange):
            return self._free_range(slots.start, slots.stop, generation)
        requested = sorted(int(slot) for slot in slots)
        if len(set(requested)) != len(requested):
            raise RuntimeError("duplicate virtual slot in free request")
        if not requested:
            return 0
        for slot in requested:
            if slot < 0 or slot >= self.capacity:
                raise IndexError(f"slot {slot} is outside allocator capacity")
        matches = []
        for slot in requested:
            match = next((item for item in self._allocated_ranges if item[0] <= slot < item[1]), None)
            if match is None:
                raise RuntimeError(f"virtual slot {slot} is already free")
            if generation is not None and match[2] != generation:
                raise RuntimeError("stale Virtual L2 allocation generation")
            matches.append(match)
        # Remove requested points from allocated intervals, preserving any
        # still-live portions. Validation above makes this operation atomic.
        requested_set = set(requested)
        new_allocated = []
        for start, end, item_generation in self._allocated_ranges:
            cuts = [p for p in requested if start <= p < end]
            cursor = start
            for point in cuts + [end]:
                if cursor < point:
                    new_allocated.append((cursor, point, item_generation))
                cursor = point + 1
        self._allocated_ranges = new_allocated
        self._free_ranges.extend((slot, slot + 1) for slot in requested)
        self._free_ranges.sort()
        merged = []
        for start, end in self._free_ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._free_ranges = merged
        freed = len(requested)
        self._allocated -= freed
        return freed

    def _free_range(self, start: int, end: int, generation: int | None) -> int:
        if start < 0 or end > self.capacity or start > end:
            raise IndexError("slot range is outside allocator capacity")
        if start == end:
            return 0
        covering = next(
            (item for item in self._allocated_ranges
             if item[0] <= start and end <= item[1]), None
        )
        if covering is None:
            raise RuntimeError("virtual slot range is not fully allocated")
        if generation is not None and covering[2] != generation:
            raise RuntimeError("stale Virtual L2 allocation generation")
        new_allocated = []
        for item_start, item_end, item_generation in self._allocated_ranges:
            if item_start == covering[0] and item_end == covering[1]:
                if item_start < start:
                    new_allocated.append((item_start, start, item_generation))
                if end < item_end:
                    new_allocated.append((end, item_end, item_generation))
            else:
                new_allocated.append((item_start, item_end, item_generation))
        self._allocated_ranges = new_allocated
        self._free_ranges.append((start, end))
        self._free_ranges.sort()
        merged = []
        for item_start, item_end in self._free_ranges:
            if merged and item_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], item_end))
            else:
                merged.append((item_start, item_end))
        self._free_ranges = merged
        self._allocated -= end - start
        return end - start

    def generation(self, slot: int) -> int:
        value = int(slot)
        for start, end, generation in self._allocated_ranges:
            if start <= value < end:
                return generation
        return 0

    def clear(self) -> None:
        self._free_ranges = [(0, self.capacity)]
        self._allocated_ranges.clear()
        self._allocated = 0
