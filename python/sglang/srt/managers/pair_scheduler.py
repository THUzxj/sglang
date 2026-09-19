from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Iterable, List, NamedTuple, Optional, Tuple


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


def allow_compact_drain(req: Any) -> bool:
    value = getattr(req, "allow_compact_drain", False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


class CompactResumeContext(NamedTuple):
    """Running-main state shared by compact resume checks in one scheduler tick."""

    main_pair_keys: frozenset[str]
    has_running_main: bool


def build_compact_resume_context(
    running_reqs: Iterable[Any],
) -> CompactResumeContext:
    """Index running main requests once for repeated compact resume checks."""

    main_pair_keys = set()
    has_running_main = False
    for running_req in running_reqs:
        if not is_context_engineering_main(running_req):
            continue
        has_running_main = True
        pair_key = context_engineering_pair_key(running_req)
        if pair_key:
            main_pair_keys.add(pair_key)
    return CompactResumeContext(frozenset(main_pair_keys), has_running_main)


def can_resume_compact_req(
    req: Any,
    running_reqs: Iterable[Any] = (),
    *,
    resume_context: Optional[CompactResumeContext] = None,
) -> bool:
    """Return whether a compact may leave either paused/retracted state."""
    if not is_context_engineering_compact(req):
        return True

    if resume_context is None:
        resume_context = build_compact_resume_context(running_reqs)

    pair_key = context_engineering_pair_key(req)
    if pair_key and pair_key in resume_context.main_pair_keys:
        return True
    if allow_compact_drain(req):
        return not resume_context.has_running_main
    return False


# Compatibility for out-of-tree users of the prototype helper.
can_resume_retracted_decode_req = can_resume_compact_req


def should_cache_paused_compact_for_kv_pressure(
    *, enable_hierarchical_cache: bool, disaggregation_mode: str
) -> bool:
    """Whether pressure retraction can hand the compact KV to HiCache.

    Unified scheduling can re-admit the request through radix matching, which
    transparently reloads an L2-only prefix.  PD decode's retracted queue has a
    different contract: it expects ``Req.kv_cache_cpu`` and restores that copy
    directly, so it must keep using its existing private host-copy path.
    """

    return enable_hierarchical_cache and disaggregation_mode == "null"


def attention_tokens(req: Any) -> int:
    return max(
        int(getattr(req, "kv_committed_len", 0) or 0),
        int(getattr(req, "context_engineering_pause_committed_len", 0) or 0),
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


def order_prefill_waiting_queue(
    waiting_queue: Iterable[Any],
    *,
    compact_starvation_threshold_seconds: float = 60.0,
    enable_priority_scheduling: bool = False,
    schedule_low_priority_values_first: bool = False,
    now: Optional[float] = None,
) -> List[Any]:
    """Order prefill requests, allowing sufficiently older compacts ahead of mains.

    The input is assumed to have already been sorted by the normal SGLang policy
    such as priority + FCFS. Each class keeps that order. A compact may pass a
    main only when its priority is higher and it has waited at this scheduler
    for at least the configured number of seconds. Other non-compact requests
    keep their place in the original order.
    """

    queue = list(waiting_queue)
    non_compacts = [req for req in queue if not is_context_engineering_compact(req)]
    compacts = [req for req in queue if is_context_engineering_compact(req)]
    if not enable_priority_scheduling:
        return non_compacts + compacts

    if now is None:
        now = time.perf_counter()
    ordered = []
    compact_index = 0
    for req in non_compacts:
        if is_context_engineering_main(req):
            while compact_index < len(compacts):
                compact = compacts[compact_index]
                compact_time = getattr(compact.time_stats, "scheduler_recv_time", 0.0)
                if (
                    compact_time <= 0
                    or now - compact_time < compact_starvation_threshold_seconds
                    or compact.priority is None
                    or req.priority is None
                    or (
                        compact.priority >= req.priority
                        if schedule_low_priority_values_first
                        else compact.priority <= req.priority
                    )
                ):
                    break
                ordered.append(compact)
                compact_index += 1
        ordered.append(req)
    ordered.extend(compacts[compact_index:])
    return ordered


def should_try_prefill_request(
    req: Any,
    *,
    can_run_reqs: Iterable[Any],
    running_reqs: Iterable[Any],
    waiting_queue: Iterable[Any],
    attention_budget: Optional[int],
    compact_attention_cost_ratio: float = 1.0,
    main_turn_decode_max_batch_size: Optional[int] = None,
) -> bool:
    """Return whether the scheduler should attempt to prefill this request.

    Main/foreground requests use normal admission. Compact requests are tried
    after them in queue order and may form a compact-only prefill batch; no pair
    with an active main is required. Prefill uses ordinary request resource cost
    rather than the decode compact cost ratio.
    """

    can_run_req_list = list(can_run_reqs)
    running_req_list = list(running_reqs)

    if is_context_engineering_main(req):
        if main_turn_decode_max_batch_size is None:
            return True
        active_main_turns = sum(
            1
            for active_req in running_req_list + can_run_req_list
            if is_context_engineering_main(active_req)
        )
        return active_main_turns < main_turn_decode_max_batch_size

    if not is_context_engineering_compact(req):
        return True

    # A compact that was genuinely retracted for KV pressure must wait for its
    # paired main (or explicit compact drain).  Fresh compact requests may still
    # perform compact-only prefill as required by the pair scheduler design.
    if getattr(req, "is_retracted", False) and not can_resume_compact_req(
        req, running_req_list + can_run_req_list
    ):
        return False

    active_reqs = running_req_list + can_run_req_list
    if attention_budget is not None:
        used_attention_tokens = sum(
            attention_tokens(active_req) for active_req in active_reqs
        )
        if used_attention_tokens + attention_tokens(req) > attention_budget:
            return False

    return True


def select_main_turn_decode_keep_indices(
    batch_reqs: List[Any], *, max_batch_size: Optional[int]
) -> List[int]:
    """Keep at most ``max_batch_size`` CE main turns, preserving batch order.

    Foreground and compact requests are left untouched here. Compact pairing and
    its separate budgets are applied by ``select_decode_keep_indices`` after
    excess main turns have been retracted.
    """

    if max_batch_size is None:
        return list(range(len(batch_reqs)))

    keep_indices: List[int] = []
    kept_main_turns = 0
    for idx, req in enumerate(batch_reqs):
        if is_context_engineering_main(req):
            if kept_main_turns >= max_batch_size:
                continue
            kept_main_turns += 1
        keep_indices.append(idx)
    return keep_indices


def select_decode_keep_indices(
    batch_reqs: List[Any],
    waiting_queue: Iterable[Any],
    *,
    max_compact_batch_size: int,
    attention_budget: Optional[int],
    compact_attention_cost_ratio: float = 1.0,
) -> List[int]:
    """Select decode requests under compact-aware budget constraints.

    All main/foreground requests are kept. Paired compact requests are admitted
    only if the compact-count and attention-token budgets can cover them. A
    decode batch containing only compact requests keeps none of them unless
    explicit compact drain is enabled.
    """

    if not batch_reqs:
        return []

    if all(is_context_engineering_compact(req) for req in batch_reqs):
        if not any(allow_compact_drain(req) for req in batch_reqs):
            return []
        keep_indices: List[int] = []
        used_attention_tokens = 0
        for idx, req in enumerate(batch_reqs):
            if not allow_compact_drain(req):
                continue
            if (
                max_compact_batch_size
                and len(keep_indices) >= max_compact_batch_size
            ):
                break
            tokens = budget_attention_tokens(
                req,
                compact_attention_cost_ratio=compact_attention_cost_ratio,
            )
            if (
                attention_budget is not None
                and used_attention_tokens + tokens > attention_budget
            ):
                break
            keep_indices.append(idx)
            used_attention_tokens += tokens
        return keep_indices

    keep_indices: List[int] = []
    keep_ids: set[int] = set()
    kept_compact_reqs = 0
    used_attention_tokens = 0

    def can_fit(req: Any) -> bool:
        if max_compact_batch_size and kept_compact_reqs >= max_compact_batch_size:
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
        nonlocal kept_compact_reqs, used_attention_tokens
        req = batch_reqs[idx]
        if id(req) in keep_ids:
            return True
        if not force and not can_fit(req):
            return False
        keep_indices.append(idx)
        keep_ids.add(id(req))
        if is_context_engineering_compact(req):
            kept_compact_reqs += 1
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
