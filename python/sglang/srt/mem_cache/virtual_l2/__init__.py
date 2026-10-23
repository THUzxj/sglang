"""Virtual L2 performance-simulation primitives."""

from .allocator import Allocation, VirtualPageAllocator
from .config import VirtualL2Config, virtual_l2_config_from_args
from .geometry import VirtualPoolGeometry
from .metrics import VirtualL2Metrics
from .pool import VirtualHostPool
from .provenance import SyntheticCacheProvenance
from .staging import StageLease, StagePool, StagePoolExhausted, create_stage_pools
from .transfer import StageChunk, StageTransferAdapter, VirtualL2TransferEngine, split_chunks
from .factory import build_virtual_host_pool
from . import adapters

__all__ = [
    "Allocation", "VirtualPageAllocator", "VirtualL2Config",
    "virtual_l2_config_from_args", "VirtualPoolGeometry", "VirtualL2Metrics",
    "VirtualHostPool", "SyntheticCacheProvenance", "StageLease", "StagePool",
    "StagePoolExhausted",
    "create_stage_pools",
    "StageChunk", "StageTransferAdapter", "VirtualL2TransferEngine", "split_chunks",
    "build_virtual_host_pool",
    "adapters",
]
