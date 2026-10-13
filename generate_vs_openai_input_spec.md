# Native `/generate` vs OpenAI-Compatible API Input Spec

This note compares SGLang's native `/generate` input shape with the
OpenAI-compatible APIs, mainly `/v1/completions` and `/v1/chat/completions`.

The native API is represented by `GenerateReqInput`. The OpenAI-compatible APIs
are first parsed as OpenAI-style request objects and then converted into
`GenerateReqInput` before they reach the tokenizer manager and scheduler.

Primary code references:

- Native `/generate`: `python/sglang/srt/managers/io_struct.py::GenerateReqInput`
- Native `/generate` endpoint: `python/sglang/srt/entrypoints/http_server.py::generate_request`
- OpenAI completions request:
  `python/sglang/srt/entrypoints/openai/protocol.py::CompletionRequest`
- OpenAI chat request:
  `python/sglang/srt/entrypoints/openai/protocol.py::ChatCompletionRequest`
- OpenAI completions conversion:
  `python/sglang/srt/entrypoints/openai/serving_completions.py::_convert_to_internal_request`
- OpenAI chat conversion:
  `python/sglang/srt/entrypoints/openai/serving_chat.py::_convert_to_internal_request`

## High-Level Difference

Native `/generate` is scheduler-oriented. It exposes SGLang's internal input
model directly:

```json
{
  "text": "Hello",
  "sampling_params": {
    "temperature": 0.7,
    "max_new_tokens": 64
  },
  "return_logprob": true
}
```

OpenAI-compatible APIs are client-compatibility-oriented. Sampling parameters
are top-level OpenAI fields, and the server converts them into
`sampling_params` internally:

```json
{
  "model": "default",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_completion_tokens": 64
}
```

After conversion, both paths use the same tokenizer manager and scheduler.

## Endpoint Shapes

Native `/generate` accepts one of these input carriers:

```text
text
input_ids
input_embeds
```

Exactly one should be provided. `text` can be a string or list of strings.
`input_ids` can be one token-id list or a list of token-id lists.
`input_embeds` can be one embedding sequence or a batch of embedding sequences.

OpenAI `/v1/completions` accepts:

```text
prompt: string | string[] | token_id[] | token_id[][]
model
```

OpenAI `/v1/chat/completions` accepts:

```text
messages: ChatCompletionMessageParam[]
model
```

Chat also has an SGLang extension:

```text
input_ids: token_id[]
```

When `input_ids` is provided on chat, it bypasses chat-template tokenization.
The `messages` field is still used for stop-token and tool-call handling.

## Prompt And Template Handling

Native `/generate` does not apply OpenAI chat formatting. If `text` is passed,
that exact text is tokenized. If `input_ids` is passed, tokenization is already
done.

OpenAI `/v1/completions` maps `prompt` into native `text` or `input_ids`:

- `str` or `list[str]` becomes `GenerateReqInput.text`
- `list[int]` or `list[list[int]]` becomes `GenerateReqInput.input_ids`
- a completion template may transform the prompt before conversion

OpenAI `/v1/chat/completions` processes `messages` through the chat-template
pipeline. The result becomes either:

- `GenerateReqInput.text`, for rendered text prompts
- `GenerateReqInput.input_ids`, for pre-rendered token prompts or special
  multimodal/chat encodings

This means `/v1/chat/completions` is not equivalent to calling `/generate` with
the last user message. It includes role formatting, chat template behavior,
tool-call constraints, reasoning settings, and multimodal message processing.

## Sampling Parameters

Native `/generate` puts generation controls under `sampling_params`.

Common native shape:

```json
{
  "text": "Hello",
  "sampling_params": {
    "max_new_tokens": 64,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": -1,
    "min_p": 0.0,
    "stop": ["</s>"],
    "stop_token_ids": [2],
    "ignore_eos": false
  }
}
```

OpenAI `/v1/completions` maps top-level fields to native sampling fields:

| OpenAI completions field | Native sampling field |
| --- | --- |
| `max_tokens` | `max_new_tokens` |
| `min_tokens` | `min_new_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `top_k` | `top_k` |
| `min_p` | `min_p` |
| `presence_penalty` | `presence_penalty` |
| `frequency_penalty` | `frequency_penalty` |
| `repetition_penalty` | `repetition_penalty` |
| `stop` | `stop` |
| `stop_token_ids` | `stop_token_ids` |
| `stop_regex` | `stop_regex` |
| `regex` | `regex` |
| `json_schema` | `json_schema` |
| `ebnf` | `ebnf` |
| `n` | `n` |
| `no_stop_trim` | `no_stop_trim` |
| `ignore_eos` | `ignore_eos` |
| `skip_special_tokens` | `skip_special_tokens` |
| `logit_bias` | `logit_bias` |
| `custom_params` | `custom_params` |
| `seed` | `sampling_seed` |

OpenAI `/v1/chat/completions` maps similarly:

| OpenAI chat field | Native sampling field |
| --- | --- |
| `max_completion_tokens` or `max_tokens` | `max_new_tokens` |
| `min_tokens` | `min_new_tokens` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `top_k` | `top_k` |
| `min_p` | `min_p` |
| `presence_penalty` | `presence_penalty` |
| `frequency_penalty` | `frequency_penalty` |
| `repetition_penalty` | `repetition_penalty` |
| chat-template stop strings | `stop` |
| `stop_token_ids` | `stop_token_ids` |
| `stop_regex` | `stop_regex` |
| `regex` | `regex` |
| `ebnf` | `ebnf` |
| `n` | `n` |
| `no_stop_trim` | `no_stop_trim` |
| `ignore_eos` | `ignore_eos` |
| `skip_special_tokens` | `skip_special_tokens` |
| `logit_bias` | `logit_bias` |
| `custom_params` | `custom_params` |
| `seed` | `sampling_seed` |
| `chat_template_kwargs.spaces_between_special_tokens` | `spaces_between_special_tokens` |

Chat defaults have one extra layer: for some parameters, the priority is:

```text
explicit request value > model generation_config > OpenAI/SGLang default
```

This is implemented in `ChatCompletionRequest.to_sampling_params`.

## Structured Output

Native `/generate` usually expresses structured decoding directly inside
`sampling_params`, for example:

```json
{
  "text": "Return JSON",
  "sampling_params": {
    "json_schema": "{\"type\":\"object\"}"
  }
}
```

OpenAI-compatible APIs expose both OpenAI-style and SGLang extension fields:

- `response_format: {"type": "json_object"}`
- `response_format: {"type": "json_schema", ...}`
- `response_format: {"type": "structural_tag", ...}`
- `regex`
- `ebnf`
- `json_schema`

The OpenAI layer converts `response_format` into native sampling constraints.
For chat, tool-call constraints may also generate a grammar constraint. If a
tool-call constraint conflicts with an explicit structured-output constraint,
the code warns instead of silently combining incompatible constraints.

## Logprob Controls

Native `/generate` exposes low-level logprob controls:

```text
return_logprob
logprob_start_len
top_logprobs_num
token_ids_logprob
return_text_in_logprobs
return_flat_raw_top_logprobs
return_flat_raw_top_logprobs_b64
return_sampling_mask
```

OpenAI `/v1/completions` exposes:

```text
logprobs
echo
```

Then conversion sets:

```text
return_logprob = logprobs is not None
top_logprobs_num = logprobs or 0
logprob_start_len = 0 only when echo and logprobs are both set, otherwise -1
return_text_in_logprobs = true
```

The code warns that OpenAI-style prompt logprob behavior is limited and that
native `/generate` should be used when prompt logprobs are needed directly.

OpenAI `/v1/chat/completions` exposes:

```text
logprobs
top_logprobs
```

Then conversion sets:

```text
return_logprob = logprobs
top_logprobs_num = top_logprobs or 0
logprob_start_len = -1
return_text_in_logprobs = true
```

## Batch Semantics

Native `/generate` supports batching directly:

```json
{
  "text": ["a", "b", "c"],
  "sampling_params": {"max_new_tokens": 8}
}
```

It also supports per-item lists for many fields, including `sampling_params`,
`return_logprob`, `logprob_start_len`, `top_logprobs_num`,
`return_hidden_states`, `lora_path`, `custom_logit_processor`, `extra_key`, and
PD bootstrap fields. During normalization, the parent request is split into
sub-requests with `obj[i]`.

OpenAI `/v1/completions` supports batch prompts:

```json
{
  "model": "default",
  "prompt": ["a", "b", "c"],
  "max_tokens": 8
}
```

OpenAI `/v1/chat/completions` does not expose a native list-of-chat-requests
batch shape. It accepts one conversation's `messages` array. Multiple outputs
for the same prompt are requested with `n`, not by passing multiple
conversations in one request.

For both native and OpenAI APIs, `n` maps to native parallel sampling. Native
normalization expands the effective batch when `n > 1`.

## Multimodal Inputs

Native `/generate` exposes multimodal fields directly:

```text
image_data
video_data
audio_data
mm_hashes
modalities
use_audio_in_video
max_dynamic_patch
min_dynamic_patch
image_max_dynamic_patch
video_max_dynamic_patch
images_config
```

OpenAI `/v1/chat/completions` carries multimodal content through message
content parts, then `_process_messages` extracts:

```text
image_data
video_data
audio_data
modalities
```

It also exposes SGLang multimodal extensions:

```text
max_dynamic_patch
min_dynamic_patch
use_audio_in_video
images_config
```

OpenAI `/v1/completions` is mostly text/token oriented, but this code path has
some SGLang extensions such as `images_config`.

## Session, LoRA, Reasoning, And Tools

Native `/generate` exposes:

```text
session_id
session_params
lora_path
lora_id
custom_logit_processor
require_reasoning
max_thinking_tokens
```

OpenAI APIs expose a subset and convert them:

- `session_id` / `session_params` pass through.
- `lora_path` may be supplied directly; the `model` name can also select a LoRA
  adapter through `base-model:adapter-name` syntax.
- `custom_logit_processor` passes through.
- Chat-specific reasoning controls, such as `reasoning_effort`,
  `chat_template_kwargs`, `separate_reasoning`, and `stream_reasoning`, affect
  chat-template rendering and response parsing.
- Chat tools and `tool_choice` are OpenAI-layer features. They may affect prompt
  construction, grammar constraints, finish reasons, and response formatting.

Native `/generate` does not have OpenAI `messages`, `tools`, `tool_choice`, or
OpenAI assistant message response formatting. If you need tool-call parsing or
role-aware chat formatting, use `/v1/chat/completions`.

## DP And PD Routing Fields

Native `/generate` uses:

```text
routed_dp_rank
disagg_prefill_dp_rank
bootstrap_host
bootstrap_port
bootstrap_room
bootstrap_pair_key
decode_tp_size
routing_key
```

OpenAI `/v1/completions` and `/v1/chat/completions` expose:

```text
routed_dp_rank
disagg_prefill_dp_rank
bootstrap_host
bootstrap_port
bootstrap_room
data_parallel_rank
```

`data_parallel_rank` is deprecated and migrated to `routed_dp_rank` if
`routed_dp_rank` is not already set. The HTTP header `X-Data-Parallel-Rank` has
higher priority than the body value in the OpenAI serving layer.

In sgl-model-gateway DP-aware routing, the gateway currently injects
`data_parallel_rank`; SGLang's OpenAI protocol layer then migrates it to
`routed_dp_rank`. Direct native `/generate` callers should prefer
`routed_dp_rank`.

## Request IDs And Metadata

Native `/generate` exposes low-level request and observability fields:

```text
rid
conversation_id
log_metrics
no_logs
custom_labels
return_prompt_token_ids
external_trace_header
received_time
priority
extra_key
```

OpenAI APIs expose some of these as extensions:

```text
rid
priority
extra_key
cache_salt
custom_labels
return_cached_tokens_details
return_token_ids
return_prompt_token_ids
return_meta_info
```

`cache_salt` is folded into the internal extra key path. `custom_labels` may
also come from allowed request headers.

## Streaming Behavior

Native `/generate` streams native SGLang chunks:

```text
data: {"text": "...", "output_ids": [...], "meta_info": {...}}
data: [DONE]
```

OpenAI APIs stream OpenAI-style SSE objects:

- `/v1/completions` returns `CompletionStreamResponse` chunks with `choices`.
- `/v1/chat/completions` returns `ChatCompletionStreamResponse` chunks with
  `choices[].delta`.
- Optional usage chunks are controlled by `stream_options` and server defaults.

Chat streaming rejects some SGLang extensions:

```text
return_prompt_token_ids
return_token_ids
return_meta_info
```

These must be used with `stream=false` on `/v1/chat/completions`.

## Response Shape

Native `/generate` non-streaming returns either one native object or a list of
native objects:

```json
{
  "text": "...",
  "output_ids": [1, 2, 3],
  "meta_info": {
    "id": "...",
    "finish_reason": {"type": "stop"},
    "prompt_tokens": 10,
    "completion_tokens": 3,
    "cached_tokens": 0,
    "weight_version": 0
  }
}
```

If `skip_tokenizer_init` or token-in-token-out mode is used, the native output
may omit `text` and return token IDs plus `meta_info`.

OpenAI `/v1/completions` wraps native outputs as:

```text
id
object = "text_completion"
created
model
choices[]
usage
metadata
sglext
```

OpenAI `/v1/chat/completions` wraps native outputs as:

```text
id
object = "chat.completion"
created
model
choices[]
usage
metadata
sglext
```

Chat choices contain assistant messages, optional parsed reasoning content,
optional tool calls, logprobs, hidden states, token IDs, and optionally raw
`meta_info` when `return_meta_info=true`.

## Practical Guidance

Use native `/generate` when:

- you already have raw text, token IDs, or embeddings;
- you need direct access to SGLang-specific fields;
- you need precise prompt logprob controls;
- you want native `meta_info` and `output_ids` without OpenAI response wrapping;
- you want explicit batch control over `text` or `input_ids`.

Use `/v1/chat/completions` when:

- the input is role-based chat messages;
- you need chat templates, tool calls, reasoning parsing, or OpenAI-style
  streaming deltas;
- you want OpenAI-compatible client behavior.

Use `/v1/completions` when:

- the input is OpenAI-style completion `prompt`;
- you need OpenAI-compatible completion response wrapping;
- you want batch prompts in OpenAI completion format.

For DP routing, prefer `routed_dp_rank` when calling SGLang directly. Expect
`data_parallel_rank` only as a compatibility field or gateway-injected field.
