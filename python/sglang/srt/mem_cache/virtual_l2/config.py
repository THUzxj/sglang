"""Configuration and validation for the benchmark-only Virtual L2 cache.

Virtual L2 deliberately lives behind its own configuration namespace.  It is
not a replacement for a real HiCache host pool or a fake PD transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DataPattern = Literal["zeros", "ones"]
DMAMode = Literal["native", "paced"]


@dataclass(frozen=True)
class VirtualL2Config:
    enabled: bool = False
    capacity_gb: float = 0.0
    staging_gb: float = 0.0
    stage_ring_entries: int = 4
    data_pattern: DataPattern = "zeros"
    dma_mode: DMAMode = "native"
    disable_real_l3: bool = True
    allow_pd: bool = False
    bandwidth_gbps: float = 0.0
    fixed_latency_us: float = 0.0

    @property
    def capacity_bytes(self) -> int:
        return int(self.capacity_gb * (1024**3))

    @property
    def staging_bytes(self) -> int:
        return int(self.staging_gb * (1024**3))

    def validate(self) -> "VirtualL2Config":
        if not self.enabled:
            return self
        if self.capacity_gb <= 0:
            raise ValueError("virtual L2 capacity must be greater than zero")
        if self.staging_gb <= 0:
            raise ValueError("virtual L2 staging size must be greater than zero")
        if self.stage_ring_entries < 1:
            raise ValueError("virtual L2 stage ring entries must be positive")
        if self.data_pattern not in ("zeros", "ones"):
            raise ValueError(f"unsupported virtual L2 data pattern: {self.data_pattern}")
        if self.dma_mode not in ("native", "paced"):
            raise ValueError(f"unsupported virtual L2 DMA mode: {self.dma_mode}")
        if self.dma_mode == "paced":
            raise ValueError(
                "paced Virtual L2 DMA is not implemented; use --virtual-l2-dma-mode native"
            )
        if not self.disable_real_l3:
            raise ValueError(
                "Virtual L2 cannot write production L3; use a separate shadow namespace"
            )
        if self.bandwidth_gbps < 0 or self.fixed_latency_us < 0:
            raise ValueError("Virtual L2 pacing values cannot be negative")
        return self


def virtual_l2_config_from_args(args) -> VirtualL2Config:
    """Build and validate a config from ServerArgs-like objects."""
    config = VirtualL2Config(
        enabled=bool(getattr(args, "virtual_l2_enable", False)),
        capacity_gb=float(getattr(args, "virtual_l2_capacity_gb", 0.0)),
        staging_gb=float(getattr(args, "virtual_l2_staging_gb", 0.0)),
        stage_ring_entries=int(getattr(args, "virtual_l2_stage_ring_entries", 4)),
        data_pattern=getattr(args, "virtual_l2_data_pattern", "zeros"),
        dma_mode=getattr(args, "virtual_l2_dma_mode", "native"),
        disable_real_l3=bool(getattr(args, "virtual_l2_disable_real_l3", True)),
        allow_pd=bool(getattr(args, "virtual_l2_allow_pd", False)),
        bandwidth_gbps=float(getattr(args, "virtual_l2_bandwidth_gbps", 0.0)),
        fixed_latency_us=float(getattr(args, "virtual_l2_fixed_latency_us", 0.0)),
    )
    return config.validate()


def validate_pd_virtual_l2(args, config: VirtualL2Config | None = None) -> None:
    """Reject PD unless the run explicitly opts into synthetic local caches."""
    config = config or virtual_l2_config_from_args(args)
    pd_enabled = bool(getattr(args, "disaggregation_mode", None)) or bool(
        getattr(args, "disaggregation_enable", False)
    )
    if config.enabled and pd_enabled and not config.allow_pd:
        raise ValueError(
            "Virtual L2 with PD requires --virtual-l2-allow-pd; "
            "P/D local caches remain worker-private and synthetic"
        )
