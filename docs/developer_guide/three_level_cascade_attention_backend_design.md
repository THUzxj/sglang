# Three-Level FlashInfer Cascade Attention Backend Design

## Goal

This design note proposes a new SGLang attention backend that uses FlashInfer
`MultiLevelCascadeAttentionWrapper` with three cascade levels. The backend will
support externally supplied request groups and shared-prefix metadata, instead
of relying only on automatic shared-prefix detection from `req_to_token`.

The design is intentionally limited to planning. It does not implement the new
backend, CLI flags, scheduler changes, or public APIs yet.

## Motivation

The existing `flashinfer-cascade` backend supports a two-level decode plan:

1. A single shared prefix across the whole running batch.
2. A per-request unique tail.

This works when every request in the batch shares the same leading physical KV
slots. It is less expressive when requests form multiple groups, or when there
is a nested prefix structure such as:

```text
all requests share prefix A
group 0 shares prefix B0 after A
group 1 shares prefix B1 after A
each request has its own tail after its group prefix
```

The proposed backend should support that structure by building a three-level
cascade plan:

1. Global shared prefix level.
2. Group shared prefix level.
3. Per-request unique tail level.

## Non-Goals

- No prefill/extend optimization in the first version. Prefill should continue
  to fall back to the parent FlashInfer backend.
- No support for MLA in the first version.
- No cross-attention support in the first version.
- No scheduler policy change is required for the backend itself.
- No token-content prefix matching inside the backend. The backend should plan
  from physical KV slot ids or trusted external metadata.

## Proposed Backend

Register a new backend name, for example:

```text
flashinfer-cascade3
```

The backend should subclass `FlashInferAttnBackend`, similar to the current
`FlashInferCascadeAttnBackend`.

Expected registration constraints:

- `runner.model_config.is_encoder_decoder` must be false.
- `runner.use_mla_backend` must be false.
- FlashInfer must be available.
- KV layout remains `NHD`.
- `page_size` remains `1`, matching the current FlashInfer paged KV slot usage
  in SGLang.

## External Group Metadata

The backend needs a way for code outside the attention backend to provide a
structured cascade plan. The recommended internal representation is a
`msgspec.Struct` stored on `ForwardBatch`, with a conservative name such as
`cascade_plan_metadata`.

Proposed shape:

```python
class CascadeRequestGroup(msgspec.Struct):
    request_indices: list[int]
    shared_prefix_len: int


class CascadePlanMetadata(msgspec.Struct):
    global_shared_prefix_len: int
    groups: list[CascadeRequestGroup]
    version: int = 1
```

Semantics:

- `request_indices` are positions in the current running batch, not request pool
  row ids. The backend maps them to `req_pool_indices`.
- `global_shared_prefix_len` is the number of leading tokens shared by all
  requests in the plan.
- `group.shared_prefix_len` is the total prefix length shared by that group,
  including the global prefix.
- The per-request tail begins at `group.shared_prefix_len`.

Example:

```text
batch positions: [0, 1, 2, 3, 4, 5]
global_shared_prefix_len = 128
groups:
  - request_indices = [0, 1, 2], shared_prefix_len = 256
  - request_indices = [3, 4, 5], shared_prefix_len = 192
```

This means:

- Level 0 reads slots `[0, 128)` once for all 6 requests.
- Level 1 has two group segments:
  - group 0 reads slots `[128, 256)` once for requests 0, 1, 2.
  - group 1 reads slots `[128, 192)` once for requests 3, 4, 5.
- Level 2 reads each request's unique tail:
  - group 0 requests read from `256` to their sequence length.
  - group 1 requests read from `192` to their sequence length.

## Metadata Validation

The backend must validate external metadata before using it.

Validation rules:

- Every `request_indices` entry must be in `[0, batch_size)`.
- A request may appear in at most one group.
- Requests omitted from all groups should either:
  - be placed in singleton groups with `shared_prefix_len =
    global_shared_prefix_len`, or
  - force fallback to the parent FlashInfer decode path.
- `global_shared_prefix_len >= 0`.
- `group.shared_prefix_len >= global_shared_prefix_len`.
- `group.shared_prefix_len < min(seq_len for requests in group)`, so every
  request keeps at least one tail token.
- `global_shared_prefix_len < min(seq_len for all planned requests)`.
- Physical KV slots must agree with the metadata:
  - all planned requests must share identical `req_to_token` slots in
    `[0, global_shared_prefix_len)`;
  - requests in each group must share identical `req_to_token` slots in
    `[global_shared_prefix_len, group.shared_prefix_len)`.

The last check is important. External metadata may describe token-level sharing,
but FlashInfer cascade is correct only when the relevant KV slots are physically
shared or identical in `req_to_token`.

## Fallback Policy

The backend should prefer correctness over partial use of invalid metadata.

Recommended fallback behavior:

- If no external metadata is present, use automatic detection as a compatibility
  mode.
- If external metadata is present and valid, use the three-level cascade plan.
- If external metadata is present but invalid, log a debug warning and fall back
  to the parent FlashInfer decode path.
- Optionally add a strict debug mode that raises on invalid metadata.

The compatibility automatic path can initially build only a two-level plan, or
it can synthesize a three-level plan with no group level:

```text
level 0: auto-detected global shared prefix
level 1: empty
level 2: per-request tails
```

## FlashInfer Three-Level Plan Mapping

Use:

```python
MultiLevelCascadeAttentionWrapper(num_levels=3, ...)
```

The three levels map to FlashInfer arrays as follows.

### Level 0: Global Shared Prefix

One query group containing all requests in the plan:

```text
qo_indptr_l0 = [0, planned_batch_size]
kv_indptr_l0 = [0, global_shared_prefix_len]
kv_indices_l0 = req_to_token[first_request, 0:global_shared_prefix_len]
last_page_len_l0 = [1] if global_shared_prefix_len > 0 else [0]
```

### Level 1: Group Shared Prefixes

One query group per external group:

```text
qo_indptr_l1 = cumulative group sizes
kv_indptr_l1 = cumulative group-shared suffix lengths
kv_indices_l1 = concat per-group slots:
  req_to_token[group_first_request, global_shared_prefix_len:group_shared_prefix_len]
last_page_len_l1 = ones(num_groups)
```

Level 1 lengths are suffix lengths relative to the global prefix:

```text
group_level_len = group.shared_prefix_len - global_shared_prefix_len
```

### Level 2: Per-Request Unique Tails

One query group per request:

```text
qo_indptr_l2 = [0, 1, 2, ..., planned_batch_size]
kv_indptr_l2 = cumulative per-request tail lengths
kv_indices_l2 = concat per-request slots:
  req_to_token[request, group_shared_prefix_len:seq_len]
last_page_len_l2 = ones(planned_batch_size)
```

## Request Ordering

FlashInfer cascade levels need consistent query ordering across all levels.

The backend should define a planned request order:

```text
groups in metadata order, requests inside each group in metadata order
```

If this order differs from the actual `forward_batch` order, the backend has
two choices:

1. Require metadata order to match `forward_batch` order.
2. Reorder `q` before `wrapper.run(...)` and scatter the output back.

The recommended first implementation is option 1. It avoids extra indexing in
the decode hot path and keeps CUDA graph capture simpler. A later version can
support arbitrary ordering if needed.

## Eager Decode Flow

For eager decode:

1. Call `super().init_forward_metadata(forward_batch)` first.
2. Read `forward_batch.cascade_plan_metadata`.
3. Validate metadata against `req_pool_indices`, `seq_lens_cpu`, and
   `req_to_token`.
4. Build three-level FlashInfer plan tensors.
5. Call `wrapper.plan(...)`.
6. Store a small `_Cascade3PlanState` for `forward_decode`.
7. In `forward_decode`, write current K/V into KV cache, then call
   `wrapper.run(q_3d, kv_for_run)`.
8. If no valid plan exists, call `super().forward_decode(...)`.

## CUDA Graph Flow

CUDA graph support is more constrained because captured graphs depend on fixed
buffer addresses and fixed Python dispatch decisions.

Recommended first implementation:

- Support CUDA graph only when external metadata is present for every replay of
  a captured batch size.
- Allocate one wrapper and fixed buffers per captured batch size.
- In capture:
  - allocate max-sized buffers for level 0, level 1, and level 2;
  - prime the wrapper with a synthetic valid three-level plan;
  - arm the cascade path so `wrapper.run(...)` is captured.
- In replay:
  - validate the current external metadata;
  - refill the pre-allocated buffers in place;
  - call `wrapper.plan(...)` before graph replay;
  - set `_cg_cascade_plan` so the captured path remains active.

If replay metadata is invalid after the graph captured cascade, falling back to
the parent decode path is not safe because the graph already recorded
`wrapper.run(...)`. The first implementation should handle this by either:

- disabling CUDA graph for this backend by default, or
- requiring valid external metadata for all captured/replayed steps and raising
  in strict mode if it is missing or invalid.

The most conservative initial rollout is:

```text
flashinfer-cascade3 eager mode first
CUDA graph support behind an explicit opt-in flag
```

## Buffer Sizing

For each captured batch size `bs`, pre-allocate:

```text
level 0:
  qo_indptr: [2]
  kv_indptr: [2]
  kv_indices: [max_global_shared_pages]
  last_page_len: [1]

level 1:
  qo_indptr: [max_groups + 1]
  kv_indptr: [max_groups + 1]
  kv_indices: [max_group_shared_pages_total]
  last_page_len: [max_groups]

level 2:
  qo_indptr: [bs + 1]
  kv_indptr: [bs + 1]
  kv_indices: [bs * max_context_len]
  last_page_len: [bs]
```

For the first version, use safe upper bounds:

- `max_global_shared_pages = max_context_len`
- `max_groups = bs`
- `max_group_shared_pages_total = bs * max_context_len`
- `level2 kv_indices = bs * max_context_len`

This is memory-heavy but simple and safe. After correctness is established,
buffer sizing can be tightened using server args such as:

```text
--cascade3-max-groups
--cascade3-max-global-prefix-tokens
--cascade3-max-group-prefix-tokens
```

## Public API Options

There are two plausible ways to let external code provide metadata.

### Option A: Internal Scheduler API

Add an internal field to request/batch objects and let custom scheduling logic
populate it before decode.

Pros:

- Minimal HTTP API surface.
- Best for experiments with custom schedulers or trace replay.
- Avoids exposing low-level KV-slot assumptions to users.

Cons:

- Not directly usable by remote clients.
- Requires integration with SGLang scheduler internals.

### Option B: Request-Level HTTP Metadata

Expose an optional request parameter that carries a group id and prefix id, then
let the scheduler build `CascadePlanMetadata`.

Example conceptual request fields:

```json
{
  "cascade_group_id": "group-a",
  "cascade_global_prefix_id": "global-x",
  "cascade_group_prefix_len": 256
}
```

Pros:

- Useful for external systems that already know request grouping.
- Can support production traffic shaping.

Cons:

- Harder to validate.
- Prefix lengths are tokenizer/model dependent.
- Public API must avoid promising correctness from token-level metadata alone.

Recommendation: implement Option A first. Add Option B only after the internal
metadata path is stable.

## Server Arguments

Potential flags:

```text
--attention-backend flashinfer-cascade3
--cascade3-enable-auto-detect
--cascade3-strict-metadata
--cascade3-disable-cuda-graph
--cascade3-max-groups
```

Initial defaults:

```text
cascade3_enable_auto_detect = true
cascade3_strict_metadata = false
cascade3_disable_cuda_graph = true
cascade3_max_groups = 0  # 0 means infer from batch size
```

## Debugging and Metrics

Expose counters similar to the existing cascade backend:

```text
total_decode_steps
cascade3_fired
cascade3_fallback_no_metadata
cascade3_fallback_invalid_metadata
cascade3_fallback_order_mismatch
cascade3_fallback_flashinfer_plan_failed
cascade3_cg_replay_invalid_metadata
```

Debug logs should include:

- batch size;
- number of groups;
- global shared prefix length;
- per-group shared prefix lengths;
- fallback reason.

## Testing Plan

Unit tests:

- Validate external metadata with valid two-group and three-group plans.
- Reject overlapping request indices.
- Reject out-of-range request indices.
- Reject group prefix shorter than global prefix.
- Reject prefix lengths that leave no per-request tail.
- Reject metadata whose `req_to_token` physical slots do not match.
- Verify planned request ordering constraints.

Plan-construction tests:

- Build a known `req_to_token` matrix.
- Construct expected level 0, level 1, and level 2 `kv_indices` and `indptr`
  arrays.
- Compare generated arrays to expected arrays.

Integration tests:

- Compare `flashinfer-cascade3` output with stock `flashinfer` under greedy
  decoding.
- Test two groups with different group prefix lengths.
- Test singleton groups.
- Test omitted requests fallback.
- Test invalid metadata fallback.
- Run an eager-mode server smoke test first.
- Add CUDA graph tests only after eager correctness is stable.

## Open Questions

- Should metadata describe token offsets, physical slot offsets, or both?
- Should arbitrary request ordering be supported in v1?
- Should invalid metadata fall back silently, log at warning level, or raise?
- Should CUDA graph be disabled by default for `flashinfer-cascade3` until the
  metadata source can guarantee validity on every replay?
- How should the scheduler batch requests to maximize useful three-level
  structure without delaying latency-sensitive traffic?

## Recommended Implementation Sequence

1. Add internal metadata structs and validation helpers.
2. Add unit tests for metadata validation and three-level plan construction.
3. Add `FlashInferCascade3AttnBackend` in eager decode only.
4. Register `flashinfer-cascade3`.
5. Add debug counters and fallback logging.
6. Add manual integration tests comparing against `flashinfer`.
7. Add optional CUDA graph support after eager mode is correct.
8. Consider public request-level metadata only after the internal API is stable.
