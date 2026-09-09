import importlib.util
from dataclasses import dataclass
from pathlib import Path


def _load_pair_scheduler():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "managers"
        / "pair_scheduler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pair_scheduler", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_scheduler = _load_pair_scheduler()
order_prefill_waiting_queue = _scheduler.order_prefill_waiting_queue
should_try_prefill_request = _scheduler.should_try_prefill_request
select_decode_keep_indices = _scheduler.select_decode_keep_indices
can_resume_retracted_decode_req = _scheduler.can_resume_retracted_decode_req
can_resume_compact_req = _scheduler.can_resume_compact_req
should_cache_paused_compact_for_kv_pressure = (
    _scheduler.should_cache_paused_compact_for_kv_pressure
)
select_compact_indices_after_paired_main_finished = (
    _scheduler.select_compact_indices_after_paired_main_finished
)
batch_context_engineering_stats = _scheduler.batch_context_engineering_stats
batch_context_engineering_observation = (
    _scheduler.batch_context_engineering_observation
)


@dataclass
class FakeReq:
    name: str
    kind: str = ""
    pair_key: str | None = None
    seqlen: int = 1
    kv_committed_len: int = 0
    finished_reason: object | None = None
    is_retracted: bool = False
    allow_compact_drain: bool = False

    @property
    def rid(self):
        return self.name

    @property
    def context_engineering_kind(self):
        return self.kind

    @property
    def context_engineering_pair_key(self):
        return self.pair_key

    def is_context_engineering_request(self):
        return self.kind in {"main", "compact"}

    def is_context_engineering_main(self):
        return self.kind == "main"

    def is_context_engineering_compact(self):
        return self.kind == "compact"

    def finished(self):
        return self.finished_reason is not None


def test_order_prefill_waiting_queue_prioritizes_main_without_pairing():
    foreground = FakeReq("foreground")
    compact_a = FakeReq("compact-a", "compact", "a")
    compact_b = FakeReq("compact-b", "compact", "b")
    main_a = FakeReq("main-a", "main", "a")
    main_b = FakeReq("main-b", "main", "b")

    ordered = order_prefill_waiting_queue(
        [foreground, compact_a, compact_b, main_a, main_b]
    )

    assert [req.name for req in ordered] == [
        "foreground",
        "main-a",
        "main-b",
        "compact-a",
        "compact-b",
    ]


def test_select_decode_keep_indices_uses_remaining_budget_for_paired_compact():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    main_b = FakeReq("main-b", "main", "b", seqlen=100)
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=1000)

    keep = select_decode_keep_indices(
        [main_a, compact_a, main_b, compact_b],
        [],
        max_batch_size=3,
        attention_budget=300,
    )

    assert keep == [0, 1, 2]


def test_select_decode_keep_indices_does_not_keep_unpaired_compact_with_main():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=10)

    keep = select_decode_keep_indices(
        [main_a, compact_b],
        [],
        max_batch_size=2,
        attention_budget=200,
    )

    assert keep == [0]


def test_select_decode_keep_indices_keeps_paired_compact_with_main_without_budget():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=1000)

    keep = select_decode_keep_indices(
        [main_a, compact_a],
        [],
        max_batch_size=256,
        attention_budget=None,
    )

    assert keep == [0, 1]


def test_select_decode_keep_indices_does_not_let_compact_displace_later_main():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    main_b = FakeReq("main-b", "main", "b", seqlen=100)

    keep = select_decode_keep_indices(
        [main_a, compact_a, main_b],
        [],
        max_batch_size=2,
        attention_budget=300,
    )

    assert keep == [0, 2]


def test_should_try_prefill_request_allows_paired_compact_with_remaining_budget():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a],
        max_batch_size=2,
        attention_budget=200,
    )

    assert should_try


def test_should_try_prefill_request_uses_normal_slot_admission():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    main_b = FakeReq("main-b", "main", "b", seqlen=100)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a, main_b],
        max_batch_size=2,
        attention_budget=300,
    )

    assert should_try


def test_should_try_prefill_request_admits_paired_compact_after_reserving_main_budget():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    main_b = FakeReq("main-b", "main", "b", seqlen=100)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a, main_b],
        max_batch_size=3,
        attention_budget=260,
    )

    assert should_try


def test_should_try_prefill_request_allows_compact_only_prefill():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=10)
    main_b = FakeReq("main-b", "main", "b", seqlen=100)
    main_c = FakeReq("main-c", "main", "c", seqlen=100)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a, main_b, main_c],
        max_batch_size=3,
        attention_budget=1_000,
    )

    assert should_try


def test_should_try_prefill_request_holds_pressure_retracted_compact_until_pair():
    compact_a = FakeReq("compact-a", "compact", "a", is_retracted=True)
    main_a = FakeReq("main-a", "main", "a")

    assert not should_try_prefill_request(
        compact_a,
        can_run_reqs=[],
        running_reqs=[],
        waiting_queue=[compact_a],
        max_batch_size=2,
        attention_budget=None,
    )
    assert should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a],
        max_batch_size=2,
        attention_budget=None,
    )


def test_should_try_prefill_request_does_not_reserve_waiting_main_attention():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    main_b = FakeReq("main-b", "main", "b", seqlen=100)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a, main_b],
        max_batch_size=3,
        attention_budget=220,
    )

    assert should_try


def test_should_try_prefill_request_rejects_paired_compact_over_budget():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=150)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a],
        max_batch_size=2,
        attention_budget=200,
    )

    assert not should_try


def test_should_try_prefill_request_ignores_compact_attention_cost_ratio():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=300)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[main_a],
        running_reqs=[],
        waiting_queue=[main_a, compact_a],
        max_batch_size=2,
        attention_budget=250,
        compact_attention_cost_ratio=0.5,
    )

    assert not should_try


def test_should_try_prefill_request_allows_compact_only_when_main_waits():
    waiting_main = FakeReq("main-a", "main", "a", seqlen=100)
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=50)

    should_try = should_try_prefill_request(
        compact_b,
        can_run_reqs=[],
        running_reqs=[],
        waiting_queue=[compact_b, waiting_main],
        max_batch_size=2,
        attention_budget=200,
    )

    assert should_try


def test_should_try_prefill_request_allows_compact_only_without_main():
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)

    should_try = should_try_prefill_request(
        compact_a,
        can_run_reqs=[],
        running_reqs=[],
        waiting_queue=[compact_a],
        max_batch_size=256,
        attention_budget=None,
    )

    assert should_try


def test_select_decode_keep_indices_retracts_compact_only_batch_for_waiting_main():
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=50)
    waiting_main = FakeReq("main-a", "main", "a", seqlen=100)

    keep = select_decode_keep_indices(
        [compact_a, compact_b],
        [waiting_main],
        max_batch_size=256,
        attention_budget=None,
    )

    assert keep == []


def test_select_compact_indices_after_paired_main_finished_retracts_matching_compact():
    main_a = FakeReq("main-a", "main", "a", finished_reason=object())
    compact_a = FakeReq("compact-a", "compact", "a")
    compact_b = FakeReq("compact-b", "compact", "b")

    retract = select_compact_indices_after_paired_main_finished(
        [main_a, compact_a, compact_b]
    )

    assert retract == [1]


def test_select_compact_indices_after_paired_main_finished_ignores_unfinished_main():
    main_a = FakeReq("main-a", "main", "a")
    compact_a = FakeReq("compact-a", "compact", "a")

    retract = select_compact_indices_after_paired_main_finished([main_a, compact_a])

    assert retract == []


def test_select_compact_indices_after_paired_main_finished_ignores_finished_compact():
    main_a = FakeReq("main-a", "main", "a", finished_reason=object())
    compact_a = FakeReq("compact-a", "compact", "a", finished_reason=object())

    retract = select_compact_indices_after_paired_main_finished([main_a, compact_a])

    assert retract == []


def test_select_decode_keep_indices_uses_compact_attention_cost_ratio():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=300)

    keep = select_decode_keep_indices(
        [main_a, compact_a],
        [],
        max_batch_size=2,
        attention_budget=250,
        compact_attention_cost_ratio=0.5,
    )

    assert keep == [0, 1]


def test_select_decode_keep_indices_retracts_compact_only_without_main():
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=1000)
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=1000)

    keep = select_decode_keep_indices(
        [compact_a, compact_b],
        [],
        max_batch_size=0,
        attention_budget=1,
    )

    assert keep == []


def test_select_decode_keep_indices_allows_explicit_compact_drain():
    compact_a = FakeReq(
        "compact-a",
        "compact",
        "a",
        seqlen=100,
        allow_compact_drain=True,
    )
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=100)

    keep = select_decode_keep_indices(
        [compact_a, compact_b],
        [],
        max_batch_size=256,
        attention_budget=None,
    )

    assert keep == [0]


def test_select_decode_keep_indices_applies_budget_to_compact_drain():
    compact_a = FakeReq(
        "compact-a",
        "compact",
        "a",
        seqlen=100,
        allow_compact_drain=True,
    )

    keep = select_decode_keep_indices(
        [compact_a],
        [],
        max_batch_size=256,
        attention_budget=50,
    )

    assert keep == []


def test_can_resume_retracted_decode_req_allows_non_compact():
    foreground = FakeReq("foreground")

    assert can_resume_retracted_decode_req(foreground, [])


def test_can_resume_compact_req_is_shared_by_paused_and_retracted_paths():
    compact_a = FakeReq("compact-a", "compact", "a")
    main_a = FakeReq("main-a", "main", "a")

    assert can_resume_compact_req(compact_a, [main_a])
    assert not can_resume_compact_req(compact_a, [])


def test_paused_compact_pressure_uses_hicache_only_in_unified_mode():
    assert should_cache_paused_compact_for_kv_pressure(
        enable_hierarchical_cache=True, disaggregation_mode="null"
    )
    assert not should_cache_paused_compact_for_kv_pressure(
        enable_hierarchical_cache=False, disaggregation_mode="null"
    )
    assert not should_cache_paused_compact_for_kv_pressure(
        enable_hierarchical_cache=True, disaggregation_mode="decode"
    )


def test_can_resume_retracted_decode_req_requires_running_pair_main_for_compact():
    main_a = FakeReq("main-a", "main", "a")
    compact_a = FakeReq("compact-a", "compact", "a")
    compact_b = FakeReq("compact-b", "compact", "b")

    assert can_resume_retracted_decode_req(compact_a, [main_a])
    assert not can_resume_retracted_decode_req(compact_b, [main_a])


def test_can_resume_retracted_decode_req_allows_compact_drain_without_running_main():
    compact_a = FakeReq(
        "compact-a",
        "compact",
        "a",
        allow_compact_drain=True,
    )

    assert can_resume_retracted_decode_req(compact_a, [])


def test_can_resume_retracted_decode_req_holds_compact_drain_behind_running_main():
    main_b = FakeReq("main-b", "main", "b")
    compact_a = FakeReq(
        "compact-a",
        "compact",
        "a",
        allow_compact_drain=True,
    )

    assert not can_resume_retracted_decode_req(compact_a, [main_b])


def test_batch_context_engineering_stats_counts_pairs():
    foreground = FakeReq("foreground")
    main_a = FakeReq("main-a", "main", "a")
    compact_a = FakeReq("compact-a", "compact", "a")
    compact_b = FakeReq("compact-b", "compact", "b")

    stats = batch_context_engineering_stats(
        [foreground, main_a, compact_a, compact_b]
    )

    assert stats == {
        "main": 1,
        "compact": 2,
        "paired_compact": 1,
        "foreground": 1,
    }


def test_batch_context_engineering_observation_is_structured():
    main_a = FakeReq("main-a", "main", "a", seqlen=100)
    compact_a = FakeReq("compact-a", "compact", "a", seqlen=50)
    compact_b = FakeReq("compact-b", "compact", "b", seqlen=20)

    observation = batch_context_engineering_observation(
        [main_a, compact_a, compact_b]
    )

    assert observation["main"] == 1
    assert observation["compact"] == 2
    assert observation["paired_compact"] == 1
    assert observation["pair_keys"] == ["a"]
    assert observation["requests"] == [
        {
            "rid": "main-a",
            "kind": "main",
            "pair_key": "a",
            "paired": False,
            "seqlen": 100,
            "kv_committed_len": 0,
        },
        {
            "rid": "compact-a",
            "kind": "compact",
            "pair_key": "a",
            "paired": True,
            "seqlen": 50,
            "kv_committed_len": 0,
        },
        {
            "rid": "compact-b",
            "kind": "compact",
            "pair_key": "b",
            "paired": False,
            "seqlen": 20,
            "kv_committed_len": 0,
        },
    ]
