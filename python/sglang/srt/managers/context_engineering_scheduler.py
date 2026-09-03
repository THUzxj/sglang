from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Iterable, List, Optional, Tuple


def is_context_engineering_request(req: Any) -> bool:
    method = getattr(req, "is_context_engineering_request", None)
    if callable(method):
        return bool(method())
    return context_engineering_kind(req) in {"main", "compact"}


def is_context_engineering_main(req: Any) -> bool:
    method = getattr(req, "is_context_engineering_main", None)
    if callable(method):
        return bool(method())
    return context_engineering_kind(req) == "main"


def is_context_engineering_compact(req: Any) -> bool:
    method = getattr(req, "is_context_engineering_compact", None)
    if callable(method):
        return bool(method())
    return context_engineering_kind(req) == "compact"


def context_engineering_kind(req: Any) -> str:
    return str(getattr(req, "context_engineering_kind", "") or "")


def context_engineering_pair_key(req: Any) -> Optional[str]:
    key = getattr(req, "context_engineering_pair_key", None)
    return str(key) if key is not None and str(key) else None


def attention_tokens(req: Any) -> int:
    return max(
        int(getattr(req, "kv_committed_len", 0) or 0),
        int(getattr(req, "seqlen", 0) or 0),
    )


def budget_attention_tokens(
    req: Any, *, compact_attention_cost_ratio: float = 1.0
) -> int:
    tokens = attention_tokens(req)
    if not is_context_engineering_compact(req):
        return tokens
    ratio = max(float(compact_attention_cost_ratio), 0.0)
    if tokens <= 0 or ratio <= 0:
        return 0
    return max(1, int(math.ceil(tokens * ratio)))


def order_prefill_waiting_queue(waiting_queue: Iterable[Any]) -> List[Any]:
    """Return a stable queue order with foreground/main requests first.

    The input is assumed to have already been sorted by the normal SGLang policy
    such as priority + FCFS. Prefill does not form main/compact pairs: all
    foreground/main requests are tried before compact requests.
    """

    queue = list(waiting_queue)
    return [req for req in queue if not is_context_engineering_compact(req)] + [
        req for req in queue if is_context_engineering_compact(req)
    ]


def should_try_prefill_request(
    req: Any,
    *,
    can_run_reqs: Iterable[Any],
    running_reqs: Iterable[Any],
    waiting_queue: Iterable[Any],
    max_batch_size: int,
    attention_budget: Optional[int],
    compact_attention_cost_ratio: float = 1.0,
) -> bool:
    """Return whether the scheduler should attempt to prefill this request.

    Main/foreground requests use normal admission. Compact requests are tried
    after them in queue order and may form a compact-only prefill batch; no pair
    with an active main is required. Prefill uses ordinary request resource cost
    rather than the decode compact cost ratio.
    """

    if not is_context_engineering_compact(req):
        return True

    active_reqs = list(running_reqs) + list(can_run_reqs)
    if max_batch_size and len(active_reqs) >= max_batch_size:
        return False

    if attention_budget is not None:
        used_attention_tokens = sum(attention_tokens(active_req) for active_req in active_reqs)
        if used_attention_tokens + attention_tokens(req) > attention_budget:
            return False

    return True


def select_decode_keep_indices(
    batch_reqs: List[Any],
    waiting_queue: Iterable[Any],
    *,
    max_batch_size: int,
    attention_budget: Optional[int],
    compact_attention_cost_ratio: float = 1.0,
) -> List[int]:
    """Select decode requests under compact-aware budget constraints.

    All main/foreground requests are kept even if they exceed the compact-aware
    budget. Paired compact requests are admitted only if the remaining batch-size
    and attention-token budgets can cover them. A decode batch containing only
    compact requests keeps none of them, regardless of remaining budget, so the
    caller retracts/requeues them until a main request is active again.
    """

    if not batch_reqs:
        return []

    if all(is_context_engineering_compact(req) for req in batch_reqs):
        return []

    keep_indices: List[int] = []
    keep_ids: set[int] = set()
    used_attention_tokens = 0

    def can_fit(req: Any) -> bool:
        if max_batch_size and len(keep_indices) >= max_batch_size:
            return False
        if attention_budget is not None:
            return (
                used_attention_tokens
                + budget_attention_tokens(
                    req,
                    compact_attention_cost_ratio=compact_attention_cost_ratio,
                )
                <= attention_budget
            )
        return True

    def add_req(idx: int, *, force: bool = False) -> bool:
        nonlocal used_attention_tokens
        req = batch_reqs[idx]
        if id(req) in keep_ids:
            return True
        if not force and not can_fit(req):
            return False
        keep_indices.append(idx)
        keep_ids.add(id(req))
        used_attention_tokens += budget_attention_tokens(
            req,
            compact_attention_cost_ratio=compact_attention_cost_ratio,
        )
        return True

    compact_by_pair_key: dict[str, Deque[Tuple[int, Any]]] = {}
    for idx, req in enumerate(batch_reqs):
        if not is_context_engineering_compact(req):
            continue
        pair_key = context_engineering_pair_key(req)
        if pair_key:
            compact_by_pair_key.setdefault(pair_key, deque()).append((idx, req))

    main_pair_keys_in_order: List[str] = []
    for idx, req in enumerate(batch_reqs):
        if is_context_engineering_compact(req):
            continue
        add_req(idx, force=True)
        if is_context_engineering_main(req):
            pair_key = context_engineering_pair_key(req)
            if pair_key:
                main_pair_keys_in_order.append(pair_key)

    for pair_key in main_pair_keys_in_order:
        paired = compact_by_pair_key.get(pair_key)
        if paired:
            compact_idx, _ = paired.popleft()
            add_req(compact_idx)

    return sorted(keep_indices)


def select_compact_indices_after_paired_main_finished(batch_reqs: List[Any]) -> List[int]:
    """Return compact indices whose paired main request finished in this batch."""

    finished_main_pair_keys = {
        context_engineering_pair_key(req)
        for req in batch_reqs
        if is_context_engineering_main(req)
        and context_engineering_pair_key(req)
        and _is_finished(req)
    }
    if not finished_main_pair_keys:
        return []

    return [
        idx
        for idx, req in enumerate(batch_reqs)
        if is_context_engineering_compact(req)
        and not _is_finished(req)
        and not bool(getattr(req, "is_retracted", False))
        and context_engineering_pair_key(req) in finished_main_pair_keys
    ]


def _is_finished(req: Any) -> bool:
    method = getattr(req, "finished", None)
    if callable(method):
        return bool(method())
    return bool(getattr(req, "finished_reason", None))


def batch_context_engineering_stats(reqs: Iterable[Any]) -> dict[str, int]:
    req_list = list(reqs)
    main_pair_keys = {
        context_engineering_pair_key(req)
        for req in req_list
        if is_context_engineering_main(req) and context_engineering_pair_key(req)
    }
    compact_pair_keys = [
        context_engineering_pair_key(req)
        for req in req_list
        if is_context_engineering_compact(req)
    ]
    paired_compacts = sum(
        1 for pair_key in compact_pair_keys if pair_key and pair_key in main_pair_keys
    )
    return {
        "main": sum(1 for req in req_list if is_context_engineering_main(req)),
        "compact": sum(1 for req in req_list if is_context_engineering_compact(req)),
        "paired_compact": paired_compacts,
        "foreground": sum(
            1
            for req in req_list
            if not is_context_engineering_main(req)
            and not is_context_engineering_compact(req)
        ),
    }


def batch_context_engineering_observation(reqs: Iterable[Any]) -> dict[str, Any]:
    req_list = list(reqs)
    main_pair_keys = {
        context_engineering_pair_key(req)
        for req in req_list
        if is_context_engineering_main(req) and context_engineering_pair_key(req)
    }
    observations = []
    for req in req_list:
        pair_key = context_engineering_pair_key(req)
        kind = context_engineering_kind(req)
        observations.append(
            {
                "rid": str(getattr(req, "rid", "")),
                "kind": kind,
                "pair_key": pair_key,
                "paired": bool(
                    kind == "compact" and pair_key and pair_key in main_pair_keys
                ),
                "seqlen": int(getattr(req, "seqlen", 0) or 0),
                "kv_committed_len": int(getattr(req, "kv_committed_len", 0) or 0),
            }
        )
    return {
        **batch_context_engineering_stats(req_list),
        "requests": observations,
        "pair_keys": sorted(pair_key for pair_key in main_pair_keys if pair_key),
    }
