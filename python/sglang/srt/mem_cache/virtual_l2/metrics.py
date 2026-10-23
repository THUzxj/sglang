from dataclasses import dataclass


@dataclass
class VirtualL2Metrics:
    d2h_bytes: int = 0
    h2d_bytes: int = 0
    d2h_submissions: int = 0
    h2d_submissions: int = 0
    premature_acks: int = 0
    lease_waits: int = 0

    def record(self, direction: str, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("byte count cannot be negative")
        if direction == "D2H":
            self.d2h_bytes += byte_count
            self.d2h_submissions += 1
        elif direction == "H2D":
            self.h2d_bytes += byte_count
            self.h2d_submissions += 1
        else:
            raise ValueError(f"unknown transfer direction: {direction}")
