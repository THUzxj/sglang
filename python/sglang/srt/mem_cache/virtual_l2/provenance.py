from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticCacheProvenance:
    synthetic_cache_data: bool = True
    source: str = "virtual_l2"
    data_pattern: str = "zeros"

    def as_dict(self) -> dict[str, object]:
        return {
            "synthetic_cache_data": self.synthetic_cache_data,
            "source": self.source,
            "data_pattern": self.data_pattern,
        }
