"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

import torch


logger = logging.getLogger(__name__)
_DEBUG_MEMORY_POOL = os.environ.get("SGLANG_DEBUG_MEMORY_POOL", "0").lower() in (
    "1",
    "true",
    "yes",
)


class MambaSlotAllocator:
    """Manages the free-list of Mamba pool slot indices.

    Unlike ``BaseTokenToKVPoolAllocator`` which is designed for per-token KV
    pages, Mamba slots are request-level (typically 1 slot per request).
    We keep the interface minimal and do NOT inherit the KV base class.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        self._event_kind = "unclassified"
        self._event_request_id = None
        self._event_req_pool_idx = None
        # Active preallocated batch for `alloc_group_begin` / `alloc_group_end`.
        # When non-None, `alloc(1)` consumes the next slot from this iterator
        # instead of calling `_do_alloc(1)` per request. Reset to None outside
        # a group window so `alloc` falls through to the per-call path.
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def _log_event(self, event: str, requested: int = 0, actual: int = 0):
        if not _DEBUG_MEMORY_POOL or not logger.isEnabledFor(logging.INFO):
            return
        available = self.available_size()
        logger.info(
            "[MambaAllocator] event=%s kind=%s request_id=%s req_pool_idx=%s "
            "requested=%d actual=%d used=%d capacity=%d available=%d",
            event,
            self._event_kind,
            self._event_request_id,
            self._event_req_pool_idx,
            requested,
            actual,
            self.size - available,
            self.size,
            available,
        )

    @contextmanager
    def event_context(
        self, kind: str, request_id=None, req_pool_idx=None
    ):
        previous = (
            self._event_kind,
            self._event_request_id,
            self._event_req_pool_idx,
        )
        self._event_kind = kind
        self._event_request_id = request_id
        self._event_req_pool_idx = req_pool_idx
        try:
            yield
        finally:
            (
                self._event_kind,
                self._event_request_id,
                self._event_req_pool_idx,
            ) = previous

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Identity to ``available_size`` for the
        static pool (slot-count and byte-coordinated views coincide); the shared
        ``UnifiedMambaSlotAllocator`` overrides it with the byte-coordinated view.
        Lets ``alloc_req_slots`` call it uniformly without a getattr fallback."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))
            else:
                self._log_event("alloc_group_failed", requested=num_reqs)

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
                self._log_event("alloc_group_release", actual=len(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                self._log_event("alloc_from_group", requested=need_size, actual=1)
                return slot
        result = self._do_alloc(need_size)
        if result is None:
            self._log_event("alloc_failed", requested=need_size)
        return result

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        self._log_event("alloc", requested=need_size, actual=need_size)
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        self.free_slots = torch.cat((self.free_slots, free_index))
        self._log_event("free", actual=free_index.numel())

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self._log_event("clear")
