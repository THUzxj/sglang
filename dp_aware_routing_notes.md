# SGLang DP-Aware Routing Notes

This note summarizes how SGLang and sgl-model-gateway handle requests that
target a specific data-parallel rank.

## Field Names

Inside SGLang, the active internal field is `routed_dp_rank`.

The older field name `data_parallel_rank` is still accepted in some entrypoints,
but it is treated as a compatibility alias and migrated to `routed_dp_rank` in
the OpenAI protocol layer.

Relevant code:

- `python/sglang/srt/managers/io_struct.py`: `GenerateReqInput.routed_dp_rank`
- `python/sglang/srt/entrypoints/openai/protocol.py`:
  `_migrate_deprecated_dp_rank`
- `python/sglang/srt/entrypoints/openai/serving_base.py`:
  `X-Data-Parallel-Rank` header override

For direct native `/generate` requests, use `routed_dp_rank`:

```json
{
  "text": ["a", "b", "c"],
  "sampling_params": {"max_new_tokens": 16},
  "routed_dp_rank": 1
}
```

For OpenAI-compatible requests, `data_parallel_rank` may still be passed through
`extra_body`, but `routed_dp_rank` is the newer name.

## Batched `/generate` Requests

`GenerateReqInput.normalize_batch_and_arguments()` determines whether the input
is a single request or a batch. For a batch, `obj[i]` creates one sub-request per
item.

The important behavior is that `routed_dp_rank` is copied from the parent
request into every sub-request:

```python
routed_dp_rank=self.routed_dp_rank
```

Therefore, one batched request with a specified rank sends all sub-requests to
the same DP rank. The current native `GenerateReqInput.routed_dp_rank` is not a
per-item list, so a single batch cannot assign different DP ranks to different
items. To target different ranks, send separate requests.

After tokenization, the tokenizer manager either sends requests one by one or
wraps them in `BatchTokenizedGenerateReqInput`. In both cases, each tokenized
request retains `routed_dp_rank`.

The data-parallel controller handles `BatchTokenizedGenerateReqInput` by
iterating over each item. For each item, if `routed_dp_rank` is set, it directly
sends the request to that DP worker and bypasses normal load-balancing policies
such as round-robin, total requests, or total tokens.

Relevant code:

- `python/sglang/srt/managers/io_struct.py`: batch normalization and `__getitem__`
- `python/sglang/srt/managers/tokenizer_manager.py`: tokenization and dispatch
- `python/sglang/srt/managers/data_parallel_controller.py`:
  `dispatch_batch_generate` and `maybe_external_dp_rank_routing`
- `python/sglang/srt/managers/scheduler.py`: scheduler receives
  `routed_dp_rank` on `Req`

## sgl-model-gateway DP-Aware Workers

`DPAwareWorkerBuilder` creates virtual worker URLs in this form:

```text
{base_url}@{dp_rank}
```

Examples:

```text
http://worker1:8080@2
grpc://cluster.local@1
```

The actual request endpoint uses the base URL without the `@rank` suffix:

```text
{base_url}{route}
```

`DPAwareWorker.endpoint_url(route)` does not enforce an API-path whitelist by
itself. It simply concatenates `base_url` and `route`. In practice, the gateway
router limits the routes by exposing fixed handlers.

Relevant code:

- `sgl-model-gateway/src/core/worker_builder.rs`: `DPAwareWorkerBuilder::build`
- `sgl-model-gateway/src/core/worker.rs`: `DPAwareWorker`
- `sgl-model-gateway/src/routers/http/router.rs`: DP rank extraction from
  `{base_url}@{rank}`

## Regular HTTP Router URLs

For the regular HTTP router, typed POST requests are routed to:

```text
{base_url}/generate
{base_url}/v1/chat/completions
{base_url}/v1/completions
{base_url}/v1/responses
{base_url}/v1/embeddings
{base_url}/v1/classify
{base_url}/v1/rerank
```

When `dp_aware` is enabled, these requests inject or overwrite:

```text
data_parallel_rank = parsed dp_rank from worker URL
```

The regular HTTP router performs this injection directly in
`send_typed_request`; it does not call `worker.prepare_request()`.

Simple GET/POST proxy paths do not inject body fields. They remove the virtual
`@rank` suffix and forward to:

```text
GET  {base_url}/health_generate
GET  {base_url}/server_info
GET  {base_url}/v1/models
GET  {base_url}/model_info
GET  {base_url}/v1/responses/{response_id}
POST {base_url}/v1/responses/{response_id}/cancel
```

Relevant code:

- `sgl-model-gateway/src/server.rs`: public route registration
- `sgl-model-gateway/src/routers/http/router.rs`: route implementations

## PD HTTP Router URLs And Field Injection

In PD mode, dual dispatch supports:

```text
{prefill_base_url}/generate
{decode_base_url}/generate

{prefill_base_url}/v1/chat/completions
{decode_base_url}/v1/chat/completions

{prefill_base_url}/v1/completions
{decode_base_url}/v1/completions

{prefill_base_url}/v1/rerank
{decode_base_url}/v1/rerank
```

PD mode explicitly rejects:

```text
/v1/embeddings
/v1/classify
```

PD query-style GET requests are forwarded to the first prefill worker:

```text
{prefill_base_url}/health_generate
{prefill_base_url}/server_info
{prefill_base_url}/v1/models
{prefill_base_url}/model_info
```

Before dual dispatch, PD mode injects or overwrites these bootstrap fields based
on the selected prefill worker:

```text
bootstrap_host
bootstrap_port
bootstrap_room
```

For a single request, these are scalars. For a batch request, these are arrays.
If `bootstrap_port` is unavailable, it is written as `null`.

For DP-aware workers, `worker.prepare_request()` injects or overwrites:

```text
data_parallel_rank = worker.dp_rank
```

For the decode request only, PD mode also injects:

```text
disagg_prefill_dp_rank = prefill_worker.dp_rank
```

Relevant code:

- `sgl-model-gateway/src/routers/http/pd_router.rs`:
  `inject_bootstrap_into_value`
- `sgl-model-gateway/src/routers/http/pd_router.rs`:
  `inject_prefill_dp_rank_for_decode`
- `sgl-model-gateway/src/routers/http/pd_router.rs`:
  `prepare_worker_request`
- `sgl-model-gateway/src/core/worker.rs`:
  `DPAwareWorker.prepare_request`

## Summary

- SGLang internal routing uses `routed_dp_rank`.
- `data_parallel_rank` is a compatibility name used by OpenAI protocol handling
  and currently injected by sgl-model-gateway.
- A batched native `/generate` request can specify only one DP rank for the
  whole batch.
- In SGLang's DP controller, an explicit routed rank bypasses normal load
  balancing.
- `DPAwareWorker` itself only injects `data_parallel_rank`.
- Regular HTTP routing injects only `data_parallel_rank` for typed POST routes.
- PD HTTP routing injects `bootstrap_host`, `bootstrap_port`, `bootstrap_room`,
  `data_parallel_rank`, and decode-only `disagg_prefill_dp_rank`.
