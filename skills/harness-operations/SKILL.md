---
name: harness-operations
description: Production features around the core loop — durable tool idempotency, prompt/config versioning, token streaming, structured logs/tracing/metrics, the HTTP API, queue workers with leases, artifact storage, long-term memory, and the evaluation runner. Use when working on any Phase 4 item or on api.py, workers, telemetry, or memory.
---

# Harness Operations (Phase 4)

These features wrap the orchestrator. None of them may make the orchestrator import a web
framework, telemetry SDK, or database driver. Each one enters through a protocol in
`domain.py` (or a sibling module) with a no-op or stdlib default. Optional third-party
packages live behind `pyproject` extras (`[api]`, `[otel]`) and are imported lazily.

## Status

- 4a is implemented: items 1, 2, and 10.
  - `IdempotencyStore` protocol in `domain.py`, implemented by both run stores
    (SQLite table `tool_results`).
  - `ToolRuntime(idempotency_store=...)` / `attach_idempotency_store`.
  - `versioning.py` hashes; `create_run(versions=...)`; `CONFIG_DRIFT` event.
  - `Usage.cached_input_tokens`.
- Deviation from item 1 as written: the result is stored in its own transaction
  immediately after the handler succeeds, not in the checkpoint transaction. The runtime,
  not the orchestrator, executes the handler. This covers a crash after the handler but
  before the checkpoint, which is the case that matters.

## 1. Durable tool idempotency
- Add intent-level store methods: `get_tool_result(key)` / `put_tool_result(key, fingerprint, result)`.
- SQLite: a new table keyed by the idempotency key, written in the same transaction as
  the tool-result checkpoint. `ToolRuntime` keeps its in-memory cache as a front layer.
- Test: restart (new runtime instance, same DB), then replay the same key → the handler
  is not called again. The same key with different arguments → rejected.

## 2. Prompt and config versioning
- On `create_run`, store `versions = {system_prompt_sha, tools_sha, profile_sha,
  harness_version}` in the run and the `RUN_CREATED` payload. Hash canonical JSON
  (sorted keys) and never include secrets.
- Use the stored versions on resume to warn (event) if the config changed mid-run.

## 3. Streaming
- Add an optional `StreamingModelClient.stream(request) -> AsyncIterator[ModelDelta]`.
  Adapters parse SSE (`data: ...`, `[DONE]`) and assemble tool-call argument fragments.
- The orchestrator publishes deltas to an injected `EventSink`, but checkpoints only the
  final assembled `ModelResponse`. Resume never depends on partial deltas.
- Fall back to `complete()` when the client does not stream.

## 4. Observability
- Add a `Telemetry` protocol: `span(name, attrs)` context manager, `metric(name, value, attrs)`,
  and `log(event)`. The default writes structured JSON lines to stderr, and a no-op
  version is available.
- Every model and tool step carries run_id, step, trace_id, profile, model, latency,
  tokens, cost, retry count, and error category. Arguments and outputs are sanitized or
  omitted.
- OpenTelemetry exporter behind the `[otel]` extra.
- Record `Usage.cached_input_tokens` (optional, default `None`) in model events, and
  emit a cache-hit-rate metric (`cached_input_tokens / input_tokens`) per run and per
  profile. An unknown count stays unknown; never report it as 0.

## 5. HTTP API (`api.py`)
- Endpoints: `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/events` (and SSE
  stream), `POST /runs/{id}/resume`, `POST /runs/{id}/cancel`,
  `GET /runs/{id}/approval`, `POST /runs/{id}/approval` (`allow|deny`, `call_id`),
  `GET /profiles`, `GET /healthz`.
- Validate every ID and body at the boundary. The API creates runs and enqueues them;
  workers execute them. Bearer auth token from an environment variable. Approval requires
  authentication.
- Framework behind the `[api]` extra (Starlette + uvicorn recommended). Tests use its
  in-process test client with no sockets.

## 6. Workers and queue
- Add store methods `claim_next_run(worker_id, lease_seconds)`, `renew_lease`, and
  `release`. In SQLite, the claim is a single `UPDATE ... WHERE lease_expires_at < now`
  guarded by version.
- The worker loop claims a run, runs it, and renews the lease on a heartbeat. An expired
  lease lets another worker resume the run, which the existing resume semantics already
  make safe.
- Graceful shutdown: stop claiming, finish or checkpoint, release.

## 7. Artifacts
- Add an `ArtifactStore` protocol (`put(bytes, media_type) -> ref`, `get(ref)`), with a
  default local-directory implementation.
- Tool outputs over a threshold go to artifacts. Events and messages carry the reference
  and a trimmed preview, which works with context-management.

## 8. Long-term memory
- Add a `MemoryStore` protocol (`search(query, k)`, `put(text, metadata)`). The default
  uses SQLite FTS5.
- Expose `memory_search` (read-only) and `memory_write` (side-effect, needs an
  idempotency key) as ordinary tools. The orchestrator itself does not know about memory.

## 9. Evaluation runner
- Scenario files (TOML/JSON): prompt, scripted or live profile, expected tool calls,
  expected final status, and assertions on events.
- `agent-harness eval scenarios/` → pass/fail table plus JSON report. Unit tests run
  only scripted scenarios.

## Handoff
Each item ships with network-free tests, docs in README, and no change in behavior when
the feature is not configured.
