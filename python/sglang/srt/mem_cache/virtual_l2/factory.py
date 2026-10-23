"""Construction helpers for integrating VirtualHostPool with HiCache."""

from __future__ import annotations

from .config import virtual_l2_config_from_args
from .geometry import VirtualPoolGeometry
from .pool import VirtualHostPool
from .staging import create_stage_pools
from .metrics import VirtualL2Metrics
import threading
import logging

_STAGE_POOLS_LOCK = threading.RLock()
_STAGE_POOLS_BY_CONFIG = {}
logger = logging.getLogger(__name__)


def _device_bytes_per_token(device_pool, page_size: int) -> int:
    size = int(getattr(device_pool, "size", 0))
    # get_kv_size_bytes() describes the whole allocated device pool, while a
    # virtual slot's payload is one row across the pool's layers.
    layer_num = int(getattr(device_pool, "layer_num", getattr(device_pool, "num_mamba_layers", 0)))
    probe = VirtualHostPool(VirtualPoolGeometry("probe", 1, 1, 1),
                            layer_num=layer_num, device_pool=device_pool)
    if layer_num:
        return sum(
            t[0].numel() * t.element_size()
            for layer in range(layer_num)
            for t in probe._device_layers(layer)
        )
    if size > 0:
        # Some pools report split K/V sizes but expose no total layer count;
        # each reported component is the full allocated backing tensor.
        size_bytes = device_pool.get_kv_size_bytes()
        if isinstance(size_bytes, tuple):
            size_bytes = sum(size_bytes)
        if size_bytes > 0:
            return int(size_bytes // size)
    raise ValueError(
        f"cannot determine bytes per logical unit for {type(device_pool).__name__}; "
        "Virtual L2 refuses to guess payload size"
    )


def build_virtual_host_pool(device_pool, *, page_size: int, pool_name: str = "kv",
                            server_args=None, capacity_units: int | None = None,
                            capacity_bytes: int | None = None):
    """Build logical pool and bounded rings for a device pool.

    The returned pool intentionally has no payload tensor. Callers pass the
    rings and adapter to ``VirtualL2TransferEngine`` for actual DMA.
    """
    config = virtual_l2_config_from_args(server_args)
    if getattr(device_pool, "use_hnd", False) or getattr(
        device_pool, "kv_cache_layout", None
    ) == "vectorized_5d":
        raise NotImplementedError(
            "Virtual L2 supports slot-major KV layouts only; HND and vectorized_5d "
            "need layout-specific index adapters"
        )
    if "mla" in type(device_pool).__name__.lower():
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().dcp_enabled:
            raise NotImplementedError(
                "Virtual L2 MLA with DCP is not supported by the logical-index adapter"
            )
    bytes_per_token = _device_bytes_per_token(device_pool, page_size)
    if capacity_units is None:
        budget = config.capacity_bytes if capacity_bytes is None else int(capacity_bytes)
        capacity_units = (budget // bytes_per_token // page_size) * page_size
    else:
        budget = int(capacity_units) * bytes_per_token
    if capacity_units < page_size:
        raise ValueError("virtual L2 capacity is smaller than one logical page")
    geometry = VirtualPoolGeometry(
        pool_name=pool_name,
        bytes_per_logical_unit=bytes_per_token,
        logical_page_size=page_size,
        logical_capacity_pages=capacity_units // page_size,
        layer_map=getattr(device_pool, "layer_map", None),
        codec="MLA" if "mla" in type(device_pool).__name__.lower() else "MHA",
    )
    layer_num = int(getattr(device_pool, "layer_num", getattr(device_pool, "num_mamba_layers", 0)))
    pool = VirtualHostPool(
        geometry,
        layer_num=layer_num,
        size_per_token=bytes_per_token,
        device_pool=device_pool,
    )
    if pool_name == "mamba":
        pool.geometry = VirtualPoolGeometry(
            pool_name=pool_name,
            bytes_per_logical_unit=bytes_per_token,
            logical_page_size=page_size,
            logical_capacity_pages=capacity_units // page_size,
            layer_map=getattr(device_pool, "mamba_layer_ids", None),
            codec="MAMBA",
        )
    # All pools lease from worker-wide rings. Each lease identifies a distinct
    # physical slot, so simultaneous transfers cannot overwrite one another.
    # Chunk sizing is pool-specific and rejects a payload larger than a slot.
    # Keep synchronization objects and staging pools out of ServerArgs. That
    # object is sent through scheduler IPC and must remain pickleable.
    stage_config = (
        config.staging_bytes,
        config.stage_ring_entries,
    )
    with _STAGE_POOLS_LOCK:
        shared_stages = _STAGE_POOLS_BY_CONFIG.get(stage_config)
        if shared_stages is None:
            shared_stages = create_stage_pools(config)
            _STAGE_POOLS_BY_CONFIG[stage_config] = shared_stages
        d2h_stage, h2d_stage = shared_stages
    if not hasattr(d2h_stage, "owners"):
        d2h_stage.owners = []
    if not hasattr(h2d_stage, "owners"):
        h2d_stage.owners = []
    d2h_stage.owners.append(pool)
    h2d_stage.owners.append(pool)
    pool.virtual_l2_config = config
    pool.virtual_l2_d2h_stage = d2h_stage
    pool.virtual_l2_h2d_stage = h2d_stage
    pool.virtual_l2_metrics = VirtualL2Metrics()
    logger.info(
        "Virtual L2 pool configured: name=%s budget_bytes=%d capacity_bytes=%d "
        "bytes_per_unit=%d logical_units=%d page_size=%d "
        "worker_shared_staging_budget_bytes=%d "
        "worker_shared_staging_physical_bytes=%d",
        pool_name,
        budget,
        geometry.capacity_bytes,
        bytes_per_token,
        geometry.capacity_units,
        page_size,
        config.staging_bytes,
        d2h_stage.physical_bytes + h2d_stage.physical_bytes,
    )
    return pool
