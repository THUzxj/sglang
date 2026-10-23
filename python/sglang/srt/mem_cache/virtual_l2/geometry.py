"""Logical Virtual L2 pool geometry."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VirtualPoolGeometry:
    pool_name: str
    bytes_per_logical_unit: int
    logical_page_size: int
    logical_capacity_pages: int
    layer_map: Any = None
    codec: str = "MHA"

    def __post_init__(self):
        if self.bytes_per_logical_unit <= 0:
            raise ValueError("bytes_per_logical_unit must be positive")
        if self.logical_page_size <= 0:
            raise ValueError("logical_page_size must be positive")
        if self.logical_capacity_pages < 0:
            raise ValueError("logical_capacity_pages cannot be negative")

    @property
    def capacity_units(self) -> int:
        return self.logical_capacity_pages * self.logical_page_size

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_units * self.bytes_per_logical_unit
