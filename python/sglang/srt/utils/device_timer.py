from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import torch


def device_timer_ctx(timer: Optional["DeviceTimer"], category: str):
    """Timing context for one forward segment; no-op when the timer is absent.

    A segment that skips this stays out of the fwd_occupancy numerator while
    still counting in its wall-clock denominator, i.e. reads as GPU idle.
    """
    if timer is None:
        return nullcontext()
    return timer.wrap(metadata={"category": category})


class DeviceTimer:
    def __init__(self, reporter: Callable):
        self._intervals: Deque[_TimingInterval] = deque()
        self._reporters: List[Callable] = [reporter]
        self._in_wrap = False

    def add_reporter(self, reporter: Callable):
        self._reporters.append(reporter)

    @contextmanager
    def wrap(self, metadata: Dict):
        # Not re-entrant: a nested wrap would end the wrong interval and leave
        # an un-ended one at the head of the queue for _report() to trip over.
        assert not self._in_wrap, "DeviceTimer.wrap is not re-entrant"
        interval = _TimingInterval.create()
        self._intervals.append(interval)
        self._in_wrap = True
        try:
            yield
        finally:
            self._in_wrap = False
            interval.end(metadata=metadata)
            self._report()

    def _report(self):
        while len(self._intervals) > 0:
            interval = self._intervals[0]
            if not interval.end_event.query():
                break

            self._intervals.popleft()
            elapsed = interval.elapsed_time() / 1000.0
            for reporter in self._reporters:
                reporter(t=elapsed, **interval.metadata)


class GapTimer(DeviceTimer):
    """Measures GPU idle gaps between consecutive uses of a stream.

    Where DeviceTimer.wrap() measures the duration *inside* a block,
    GapTimer.wrap() measures the time *between* consecutive blocks
    (gap = next_block_start - last_block_end).
    """

    def __init__(self, reporter: Callable):
        super().__init__(reporter)
        self._pending: Optional[_TimingInterval] = None

    @contextmanager
    def wrap(self, metadata: Dict):
        if self._pending is not None:
            self._pending.end(metadata=metadata)
            self._intervals.append(self._pending)
            self._pending = None
            self._report()
        try:
            yield
        finally:
            self._pending = _TimingInterval.create()

    def cancel(self):
        """Discard a pending gap (e.g. server went idle)."""
        self._pending = None


class GraphReplayTimer:
    """Async full-graph replay timing, independent of torch.profiler.

    The gap includes any intervening stream work, not just GPU idle. Samples
    are reported when ready, so a log window need not match its current batch.
    Event pairs are bounded; a slow consumer drops measurements, never waits.
    """

    def __init__(self, max_pending: int = 256):
        self._pending = deque()
        self._previous_end = None
        self._previous_stream = None
        self._max_pending = max_pending
        self._sequence = 0
        self._clear_window()

    def _clear_window(self):
        self._count = self._gap_count = self._dropped = 0
        self._total_ms = self._gap_total_ms = 0.0
        self._max_ms = self._gap_max_ms = 0.0
        self._last_id = self._last_bs = None

    def break_sequence(self):
        """Do not count server idle / eager / prefill time as a replay gap."""
        self._previous_end = self._previous_stream = None

    @contextmanager
    def wrap(self, batch_size: int):
        self._drain()
        self._sequence += 1
        if len(self._pending) >= self._max_pending:
            self._dropped += 1
            self.break_sequence()
            yield
            return
        stream = torch.cuda.current_stream()
        stream_key = (stream.device, stream.cuda_stream)
        previous = (
            self._previous_end if stream_key == self._previous_stream else None
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        try:
            yield
        except BaseException:
            self.break_sequence()
            raise
        else:
            end.record(stream)
            self._pending.append(
                (start, end, previous, self._sequence, batch_size)
            )
            self._previous_end = end
            self._previous_stream = stream_key

    def _drain(self):
        while self._pending:
            start, end, previous, sample_id, batch_size = self._pending[0]
            if not end.query():
                break
            self._pending.popleft()
            elapsed = start.elapsed_time(end)
            self._count += 1
            self._total_ms += elapsed
            self._max_ms = max(self._max_ms, elapsed)
            self._last_id, self._last_bs = sample_id, batch_size
            if previous is not None:
                gap = previous.elapsed_time(start)
                self._gap_count += 1
                self._gap_total_ms += gap
                self._gap_max_ms = max(self._gap_max_ms, gap)

    def take_log_suffix(self) -> str:
        """Consume completed samples since the last log, without synchronizing."""
        self._drain()
        mean = self._total_ms / self._count if self._count else float("nan")
        gap_mean = (
            self._gap_total_ms / self._gap_count
            if self._gap_count else float("nan")
        )
        maximum = self._max_ms if self._count else float("nan")
        gap_maximum = self._gap_max_ms if self._gap_count else float("nan")
        suffix = (
            f", cuda graph samples: {self._count}"
            f", cuda graph ms avg: {mean:.3f}"
            f", cuda graph ms max: {maximum:.3f}"
            f", cuda graph gap samples: {self._gap_count}"
            f", cuda graph gap ms avg: {gap_mean:.3f}"
            f", cuda graph gap ms max: {gap_maximum:.3f}"
            f", cuda graph last sample: {self._last_id}"
            f", cuda graph last bs: {self._last_bs}"
            f", cuda graph pending: {len(self._pending)}"
            f", cuda graph dropped: {self._dropped}"
        )
        self._clear_window()
        return suffix


@dataclass
class _TimingInterval:
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None
    metadata: Optional[Dict] = None

    @staticmethod
    def create():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return _TimingInterval(start_event=start_event)

    def end(self, metadata: Dict):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        assert self.end_event is None
        self.end_event = end_event
        self.metadata = metadata

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)
