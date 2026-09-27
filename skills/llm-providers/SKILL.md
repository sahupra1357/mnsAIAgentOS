---
name: llm-providers
description: Configure, adapt, and test the three supported LLM backends — Ollama, llama.cpp (llama-server), and models hosted on Modal (vLLM/SGLang). Use when adding or changing a provider adapter, a models.toml profile, provider auth headers, health checks, cold-start handling, or provider-specific response quirks.
---

# LLM Providers: Ollama, llama.cpp, Modal

All three backends speak OpenAI-compatible `/v1/chat/completions`. Reuse
`OpenAICompatibleClient` and put each provider's differences in a thin adapter or
profile preset under `src/agent_harness/model_gateway/`. The orchestrator must never
branch on a provider name. Read `skills/model-gateway/SKILL.md` as well.

## Ollama

- Endpoint: `http://127.0.0.1:11434/v1`. No auth is needed, but some clients send a
  dummy bearer. Do not require `api_key_env`.
- `model` is the Ollama tag, e.g. `qwen2.5-coder:7b`, `llama3.1:8b`.
- Tool calling works only with tool-capable models. Some responses return
  `function.arguments` as an object rather than a JSON string, and older builds omit
  `tool_calls[].id`. Normalize both: synthesize a stable ID from response ID + index
  instead of failing as a malformed response.
- **Context length:** the OpenAI-compatible endpoint cannot set `num_ctx`, and the
  server default is small (a few thousand tokens). Longer prompts get truncated
  silently. Options: set `OLLAMA_CONTEXT_LENGTH` on the server, use a Modelfile with
  `PARAMETER num_ctx`, or add an optional native mode that calls `/api/chat` with
  `options.num_ctx`. Always record the effective context in profile config
  (`context_window`) so context management can budget for it.
- Health: `GET /api/tags` lists local models; `GET /api/version` returns liveness.
- The first call after idle loads the model, which can take several seconds. Use a
  timeout of at least 120 s for large models. `keep_alive` controls unload.

## llama.cpp (`llama-server`)

- Start with tool support: `llama-server -m model.gguf --jinja -c 16384 --port 8080`.
  Without `--jinja`, tool calls are not parsed and arrive as plain text.
- Endpoint: `http://127.0.0.1:8080/v1`. `model` is effectively ignored (one model per
  server), but config still requires a value, so use a descriptive label.
- Auth is optional (`--api-key`). If it is set, use `api_key_env` as a normal bearer token.
- Health: `GET /health` returns 503 while the model loads and 200 when it is ready. A 503
  is already classified as retryable.
- Context is fixed by `-c` at server start, and `-c` is split across `--parallel` slots.
  Set profile `context_window` to the per-slot value.
- Usage tokens are returned; cost is zero.

## Modal-hosted models

- A Modal app runs **vLLM or SGLang** behind `@modal.web_server`. The profile sets
  `server = "vllm" | "sglang"`. URL shape:
  `https://<workspace>--<app>-<function>.modal.run/v1`. Never commit a real workspace
  URL; examples use placeholders.
- **Both auth layers are used together** (the project owner's decision):
  1. Server key: vLLM `--api-key` / SGLang `--api-key` → `Authorization: Bearer`
     from `api_key_env`.
  2. Modal proxy auth (`requires_proxy_auth=True` on the web endpoint) → headers
     `Modal-Key` and `Modal-Secret`, resolved from environment variables named in
     `headers_env`. Never put the values in TOML.
  A missing variable for either layer is a `ModelConfigurationError` before any
  request. A 401/403 is an `AuthenticationError`. Error text must say which layer is
  missing without echoing values.
- Server launch reference (deploy script lives in `examples/modal/`):
  - vLLM: `vllm serve <model> --api-key $KEY --enable-auto-tool-choice
    --tool-call-parser <hermes|llama3_json|mistral|...> --max-model-len <N>`
  - SGLang: `python -m sglang.launch_server --model-path <model> --api-key $KEY
    --tool-call-parser <qwen25|llama3|mistral|...> --context-length <N>`
  Without a tool-call parser, tool calls come back as plain text. `doctor` should detect
  this and say so.
- Health: vLLM exposes `GET /health`; SGLang exposes `GET /health` (and
  `/health_generate`). Both serve `GET /v1/models`. Use `/v1/models` as the common probe.
- **Cold starts:** a scaled-to-zero container can take 30 s to several minutes on the
  first request (container start plus weight load). Handle this in config, not code
  hacks: use a longer `timeout_seconds`, a separate `connect/cold_start` allowance, and
  treat 502/503/504 and timeouts as retryable. Remember that adapter and orchestrator
  retries multiply. An optional `warmup`/health probe (`GET /health` or `/v1/models`)
  can run before the first run step.
- vLLM tool calling requires server flags `--enable-auto-tool-choice
  --tool-call-parser <hermes|llama3_json|mistral|...>` that match the model.
- Cost: Modal bills GPU-seconds. For `max_cost_usd` to work, estimate cost as
  `latency_seconds × gpu_usd_per_second` (profile config), or use per-token prices if
  configured. Mark the estimate as an estimate in event payloads.

## Prefix / KV caching on Modal (vLLM, SGLang)

- The KV cache within a request is always on. Cross-request **prefix caching** is on by
  default in vLLM (its current engine) and SGLang (RadixAttention). No request-side
  markers are needed. Do not pass `--no-enable-prefix-caching` / `--disable-radix-cache`.
- Benefit: lower time to first token and fewer GPU-seconds. There is no per-token
  discount; Modal bills GPU time.
- The cache lives in one container's GPU memory. Scale to zero empties it, and several
  replicas cause misses because Modal does not route by prefix. The deploy script
  therefore uses a long `scaledown_window`, optional `min_containers=1`,
  `@modal.concurrent(max_inputs=...)`, and a low `max_containers`.
- Capacity: memory left after weights. Tune `--gpu-memory-utilization`,
  `--max-model-len`, and optionally `--kv-cache-dtype fp8`. Eviction is least recently used.
- Visibility: start vLLM with `--enable-prompt-tokens-details` so responses include
  `usage.prompt_tokens_details.cached_tokens`. Adapters map this to
  `Usage.cached_input_tokens` (optional; `None` if absent). Verify the SGLang field
  against the deployed version before parsing it.
- `doctor` checks caching by sending an identical prompt twice and reporting cached tokens.
- The harness side (a fixed prefix) is covered in `skills/context-management/SKILL.md`.

## Profile shape (implemented; see `examples/models.toml`)

```toml
[models.ollama]
adapter = "ollama"
model = "qwen2.5-coder:7b"
endpoint = "http://127.0.0.1:11434/v1"
context_window = 8192
timeout_seconds = 180

[models.llamacpp]
adapter = "llama_cpp"
model = "local-gguf"
endpoint = "http://127.0.0.1:8080/v1"
context_window = 16384

[models.modal]
adapter = "modal"
server = "vllm"            # or "sglang"
model = "Qwen/Qwen2.5-7B-Instruct"
endpoint = "https://WORKSPACE--APP-serve.modal.run/v1"
api_key_env = "MODAL_LLM_API_KEY"
timeout_seconds = 300
context_window = 32768
[models.modal.headers_env]
Modal-Key = "MODAL_PROXY_TOKEN_ID"
Modal-Secret = "MODAL_PROXY_TOKEN_SECRET"
[models.modal.pricing]
gpu_usd_per_second = 0.0
```

New fields must be optional with safe defaults so existing profiles and tests still parse.

## LiteLLM (reaching providers that are not OpenAI-compatible)

Two routes. Prefer the first; it needs no harness code:

1. **LiteLLM Proxy**: run `litellm --config litellm.yaml` separately. It exposes an
   OpenAI-compatible endpoint, so use a normal `adapter = "openai_compatible"` profile
   pointing at it (`api_key_env` = the proxy master/virtual key). LiteLLM then handles
   Anthropic, Gemini, Bedrock, Azure, Vertex, Cohere, and others.
2. **`adapter = "litellm"`** (Phase 5): in-process `litellm.acompletion`, behind an
   optional `[litellm]` extra and imported lazily so the core stays stdlib-only. Rules:
   pass `num_retries=0` (the harness owns retries); map LiteLLM exceptions onto
   `ThrottlingError`/`ModelTimeoutError`/`AuthenticationError`/etc.; take `cost_usd`
   from LiteLLM's computed response cost when present; normalize into `ModelResponse`
   exactly like other adapters; pin the version.

Caveats: translation is lossy for provider-specific features (prompt caching,
extended thinking, citations). Tool calling still depends on the model. Neither route
must appear in the orchestrator.

## Where it lives

- `model_gateway/openai_compatible.py`: base client and hooks (`request_headers`,
  `api_base_url`, `health_url`, `models_url`, `_preprocess_response`,
  `_cached_input_tokens`, `_cold_start_timeout`).
- `model_gateway/ollama.py`, `llama_cpp.py`, `modal.py`: provider subclasses.
- `model_gateway/diagnostics.py`: `diagnose()`, used by `agent-harness doctor`.
- `examples/modal/vllm_server.py`, `examples/modal/sglang_server.py`: deploy scripts.
- `tests/test_providers.py`: fake-transport tests for all of the above.

Cached-token counts are currently exposed as `ModelResponse.metadata["cached_input_tokens"]`.
Phase 4 promotes this to `Usage.cached_input_tokens`.

## Testing

- Use a fake `HttpTransport` with recorded-shape JSON for each provider: Ollama with
  object arguments and missing IDs, llama.cpp 503-then-200, and Modal proxy headers
  present plus cold-start 504 then success.
- Assert that header values come from the environment at request time and never appear in
  events, errors, or `repr`.
- Never contact a real server in `tests/`. Put live smoke checks behind an explicit CLI
  command (e.g. `agent-harness doctor --model-profile modal`), not in the unit suite.
