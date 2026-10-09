---
name: model-gateway
description: Create or change provider-neutral model contracts, configurable model routing, and provider adapters. Use when adding a model provider, endpoint, capability route, normalized response, or model retry behavior.
---

# Model Gateway

Make model selection a configuration concern. Orchestration code must depend on a small normalized interface and must not branch on provider names.

## Normalized Contract

A model request should carry messages, available tool schemas, sampling controls, a timeout, and trace metadata. A normalized response should represent assistant text, zero or more tool calls, usage, provider request ID, and finish reason.

`ModelRequest.response_schema` (a JSON Schema) and `ModelRequest.tool_choice` (`auto`,
`none`, `required`, or one tool name) default to `None`, and adapters must then leave the
payload unchanged. The OpenAI-compatible adapter maps them to `response_format`
(`json_schema`, strict) and `tool_choice` (a named tool becomes
`{"type": "function", "function": {"name": ...}}`); native Ollama maps the schema to
`format` and has no `tool_choice`. The orchestrator uses them for runs with an
`output_schema` (see `orchestration/structured_output.py`) and validates results itself, because servers
enforce both loosely.

Gateway behaviour per profile (Phase 12d):
- `max_concurrency` is a semaphore on the cached client: one per profile per process.
- `Retry-After` (seconds or an HTTP date) on 429/5xx sets `RetryableModelError.retry_after`.
  `retry_delay` takes the larger of backoff and that delay, capped by
  `max_retry_after_seconds`, then applies `retry_jitter`.
- `structured_output = "prompt"` sends the schema as a final user message instead of
  `response_format` / `format`.
- New profile fields use `HASH_OMIT_DEFAULT`.
- A success after adapter retries carries `metadata["provider_attempts"]`,
  `["retried_errors"]` (error categories) and `["attempt_timeout_seconds"]`; the orchestrator
  copies them into the `model_responded` event and `model_call` telemetry.
- Ollama native mode maps `parameters.max_tokens` to `options.num_predict` (an explicit
  `num_predict` wins), so one output cap works in both API modes.

`Message.parts` (images) becomes an OpenAI content array (text, then `image_url`), or
native Ollama `images` (base64 from data URLs only).

Preserve provider-only fields inside adapter metadata. Do not leak provider SDK types into shared domain objects.

## Configuration

Support named model profiles. A profile identifies an adapter, model name, endpoint, credential environment-variable name, timeout, and optional default parameters. Resolve secrets at runtime; configuration files contain only environment-variable references.

OpenAI-compatible endpoints are the baseline adapter because they cover hosted and many local models. Keep the adapter injectable so tests use a fake transport and never access the network.

## Reliability

- Validate normalized output before returning it.
- Classify authentication, throttling, timeout, malformed-response, and provider errors.
- Tool-call arguments that are not a JSON object are not a malformed response: `_parse_tool_calls`
  keeps the raw text under `INVALID_ARGUMENTS_KEY` (and the error under `INVALID_ARGUMENTS_ERROR_KEY`),
  and the orchestrator answers that call with a validation error instead of running it
  (`_reject_invalid_arguments`). A Qwen 3.6 researcher once died on "Expecting ',' delimiter".
- Retry only explicit transient failures with bounded backoff.
- Do not silently route to a different model unless the selected routing policy permits fallback.
- Return usage when provided; mark it unknown rather than inventing counts.

## Handoff

Document how callers select a profile and how another adapter can be registered. Provide tests for text responses, tool calls, malformed responses, timeout classification, and missing credentials.

## Embeddings and token counting (Phase 14)

- `OpenAICompatibleClient.embed(texts)` posts to `{base}/embeddings` and returns vectors in
  input order (used by semantic memory).
- `count_input_tokens(request)` is advisory: it returns None on any failure.
  - **vLLM / OpenAI-compatible:** `POST /tokenize` with messages and tools → `count`.
  - **llama.cpp:** `/apply-template`, then `/tokenize`; without `/apply-template`, the joined
    texts plus 8 tokens per message.
  - **Unsupported:** Ollama, and SGLang on Modal. Setting `token_counting = "server"` there
    is a configuration error (`_counts_tokens`).
- `token_counting` is hash-stable at its default (`HASH_OMIT_DEFAULT`).

## Companion profiles (Phase 16)

`ModelGateway.resolve("llamacpp.embed")` works like any profile: `load_config` adds companions to
`config.models`. `OpenAICompatibleClient.embed()` sends `self._profile.model`, which for a companion
is the embedding model, to the companion's endpoint. List user-facing profiles with
`config.chat_models()`, never `config.models`.
