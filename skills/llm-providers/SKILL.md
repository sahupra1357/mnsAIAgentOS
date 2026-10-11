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
- **One system message (Phase 20):** Qwen 3.x templates (llama.cpp `--jinja`, vLLM, SGLang)
  raise "System message must be at the beginning" when a request has two. Runs start with
  several (instruction files, system prompt, plan instructions, canary), so every client
  merges the leading system messages before sending (`model_gateway/system_messages.py`).
- **Guard model needs `--no-prefill-assistant`:** llama.cpp continues a final assistant
  message by default, and the output guard sends `[user, assistant]`. Llama Guard then never
  sees its instructions and echoes the reply. Set it for the guard model only, e.g. in a router
  preset section: `no-prefill-assistant = true` (applied when the router restarts). The harness
  reports an echoed reply as a `guard_error` that names the flag.

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
- **Modal Shared Endpoints** (profile `modal_shared`, per token): `openai_compatible` adapter, one
  bearer token `wk-<id>.ws-<secret>` (`MODAL_SHARED_TOKEN`), `affinity_header =
  "Modal-Routing-Affinity-Key"` (the adapter sends the request's `metadata["affinity_key"]`, which
  the orchestrator sets to `<run id>:<summary covers>`), `cached_input_usd_per_million_tokens`,
  `token_counting = "estimate"`, and `embedding_profile` / `guard_profile` borrowing `modal.embed` /
  `modal.guard`. Checked live 2026-10-10 with DeepSeek V4.1 Flash: structured tool calls,
  `reasoning_content`, `chat_template_kwargs` accepted, cached tokens in `prompt_tokens_details`.
- Server launch reference (deploy script lives in `deploy/modal/`):
  - vLLM: `vllm serve <model> --api-key $KEY --enable-auto-tool-choice
    --tool-call-parser <hermes|llama3_json|mistral|qwen3_coder|...> --max-model-len <N>`
    plus `--reasoning-parser <qwen3|...>` for a thinking model, so its thinking arrives as
    `reasoning_content`. The root deployment is `Qwen/Qwen3.6-35B-A3B` (BF16) on one H200
    with `qwen3_coder` and `qwen3` (vLLM recipe for Qwen 3.6). The FP8 build on an A100 is a
    commented preset only: the FP8 MoE path on Ampere (weight-only Marlin) is unconfirmed.
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

### Reasoning content (Phase 11b)
- Parsed in `OpenAICompatibleClient._normalize_raw` and `_StreamAccumulator` from
  `reasoning_content`, then `reasoning`; Ollama native maps `message.thinking`. Non-string
  or blank values are ignored, never rejected.
- `reasoning_tags = true` splits `<think>` blocks only when no separate field is present.
- `reasoning_history = "keep"` adds `reasoning_content` (Ollama native: `thinking`) to
  assistant messages in the payload; `"drop"` (default) leaves payloads byte-identical to
  before. `ContextLimits.counts_reasoning` follows it.
- Profile fields added after profile hashing existed carry `HASH_OMIT_DEFAULT` metadata,
  so they are left out of `profile_sha256` at their default. Do the same for any future
  profile field, or every existing run reports `config_drift` on resume.

## LiteLLM (reaching providers that are not OpenAI-compatible)

Two routes. Prefer the first; it needs no harness code:

1. **LiteLLM Proxy**: run `litellm --config litellm.yaml` separately. It exposes an
   OpenAI-compatible endpoint, so use a normal `adapter = "openai_compatible"` profile
   pointing at it (`api_key_env` = the proxy master/virtual key). LiteLLM then handles
   Anthropic, Gemini, Bedrock, Azure, Vertex, Cohere, and others.
2. **`adapter = "litellm"`** (implemented in Phase 5, `model_gateway/litellm_client.py`):
   in-process `litellm.acompletion`, behind an optional `[litellm]` extra and imported
   lazily so the core stays stdlib-only. It is opt-in per profile, and `endpoint` is
   optional for these profiles. `supports_http_probes = False`, so `doctor` skips the
   health and model-list checks. Rules:
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
- `deploy/modal/vllm_server.py`, `deploy/modal/sglang_server.py`: deploy scripts.
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

## Self-contained profiles (Phase 16)

One profile serves chat, embeddings, and the guard model from the same provider:

```toml
[models.llamacpp]           # router: chat + embedding model side by side (models-max >= 2)
model = "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"
embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"
# embedding_endpoint = "http://127.0.0.1:8081/v1"   # single-model servers: a 2nd llama-server --embedding
# guard_model = "..."                                # Llama Guard GGUF in the same router

[models.ollama]             # one server, several model names
embedding_model = "nomic-embed-text"
guard_model = "llama-guard3:1b"

[models.modal]              # one deployment = one model: three deployments, one profile
embedding_model = "nomic-ai/nomic-embed-text-v1.5"
embedding_endpoint = "https://WORKSPACE--agent-harness-vllm-embed-serve.modal.run/v1"
guard_model = "meta-llama/Llama-Guard-3-1B"
guard_endpoint = "https://WORKSPACE--agent-harness-vllm-guard-serve.modal.run/v1"
```

Modal deploy scripts:
- `deploy/modal/vllm_embed_server.py`: uses `--runner pooling --trust-remote-code`
  (NomicBertModel). Checked against the vLLM 0.30.0 source.
- `deploy/modal/vllm_guard_server.py`: Llama Guard 3 1B, which is gated, so the
  `agent-harness-hf` secret must hold `HF_TOKEN`.

They share the `agent-harness-llm` bearer secret and proxy auth with the chat deployment.
Changing `models.toml` never requires a redeploy; only the model a deployment serves does.

Per provider:
- **llama.cpp router:** the preset needs a section for the embedding model with
  `embedding = true` (the Llama app reads `~/.config/llama/models.user.ini`; the app's own
  `models.ini` is regenerated). The router must allow two loaded models (`--models-max 2`).
  Otherwise every embedding call unloads the chat model.
- **Ollama:** the same server answers `/v1/embeddings` for `nomic-embed-text`.
- **Modal / vLLM:** one deployment serves one model. Use `vllm_embed_server.py` (vLLM
  `--runner pooling`) and `vllm_guard_server.py`, named by `embedding_endpoint` and
  `guard_endpoint`.
- **llama.cpp preset gotcha:** each model needs its own `[org/repo:QUANT]` section header. A
  second section with an existing header silently replaces that model. One symptom is
  "Pooling type 'none' is not OAI compatible", when a chat or guard GGUF is loaded under
  the embedding model's name.

Switching providers:
- **The switch:** `AGENT_HARNESS_MODEL_PROFILE=llamacpp|ollama|modal`, normally in `.env`,
  which the CLIs read at startup; a shell variable overrides it.
- **Where the real values live:** endpoints with workspace names and every key belong in
  `.env`, never in `models.toml`. For Modal these are `AGENT_HARNESS_MODEL__MODAL__ENDPOINT`,
  `..._EMBEDDING_ENDPOINT`, `..._GUARD_ENDPOINT`, and `..._ENABLED=true`.

How it fits together:
- **Companions:** the fields become the companions `NAME.embed` / `NAME.guard`, resolved by the
  normal gateway (same credentials and headers).
- **Checking:** `doctor` checks both companions.
- **Do not add cross-provider defaults:** fallback to another provider is opt-in only.

## gpt-oss quirks (any server)

- **Leaked harmony format:** tool names come back as `functions.read_note`, or as a
  `functions` wrapper around `{"name", "arguments"}`, and the model calls a subagent by its
  name instead of `agent`. `recover_tool_names` (`model_gateway/tool_names.py`) fixes these
  against the tools offered in that request. Add any new leak patterns there, never in the
  orchestrator.
- **One tool call per message on Ollama:** parallel-call benchmarks (bfcl `parallel_*`) score
  low, and subagents run one after another. That is expected for this model and server.
