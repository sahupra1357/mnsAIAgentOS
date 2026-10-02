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

- 4b is implemented: items 3 and 4.
  - `domain.py`: `TelemetryRecord`/`TelemetrySink`, `ModelStreamEvent`/
    `StreamListener`.
  - `telemetry.py`: sinks `Noop`, `JsonLines`, `Metrics` (cache-hit rate),
    `Multi`, and `OpenTelemetry` (lazy `opentelemetry-api`, `[otel]` extra).
  - The orchestrator emits an `event` record for every persisted event, plus
    `model_call` and `tool_call` records with durations. Emission is guarded, so it
    cannot break a run.
  - `OpenAICompatibleClient.stream()` parses SSE through `StdlibHttpTransport.post_stream`.
    It is not retried at the adapter level; the orchestrator's attempt budget applies.
  - Ollama native mode does not stream.
  - CLI: `run --stream` and `--telemetry-file`.

- 4c is implemented: items 5 and 6.
  - `RunQueue` protocol with enqueue/claim/renew/release, implemented by both stores.
    SQLite: a `run_queue` table, with claims inside `BEGIN IMMEDIATE`.
  - `worker.py`: `Worker` with a heartbeat. A lost lease cancels local execution.
    Unsettled runs are requeued; poison runs are dropped after N failures.
  - `api.py`: framework-free ASGI `HarnessAPI` with bearer auth, whose lifespan hosts
    workers, and `create_api(app)`.
  - CLI: `serve` (uvicorn, `[api]` extra) and `worker`.
- 4d is implemented: items 7, 8, and 9.
  - `artifacts.py`: content-addressed `LocalArtifactStore` and the `read_artifact`
    tool. The orchestrator offloads tool results above the threshold, keeping a
    preview in the message and `artifact_id` in the event.
  - `memory.py`: `SQLiteMemoryStore` (FTS5 with a quoted-terms query, LIKE fallback,
    namespaces, dedupe) and the `memory_search` / `memory_write` tools.
  - `evals.py`: TOML scenarios (scripted or `--live`), assertions, JSON report.
  - CLI: `eval`.
  - Config: `[artifacts]` and `[memory]` sections.

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

### Run listing (Phase 11a)
- `RunLister.list_runs(RunQuery) -> RunPage` is a separate, optional store protocol, so
  `RunStore` is unchanged. The memory, SQLite, and Postgres stores implement it, and the
  conformance suite's `test_list_runs` checks each one.
- Newest first by `(created_at, id)`, with keyset pagination. The cursor is the position
  of the page's last run, so creating runs does not shift later pages. `limit` is 1–200.
- Top-level runs only, unless `include_children` is set or `parent_run_id` asks for one
  run's subagents. Filters use the indexed `tenant`, `status`, `created_at`, and
  `parent_run_id` columns; only the returned page's snapshots are decoded.
- `RunSummary.title` is the first line of the first user message, at most 80 characters.
- `GET /runs` is scoped to the principal's tenant; only a `"*"` principal may pass
  `tenant=`. CLI: `agent-harness runs`.

### Chat API (Phase 12b)
- `streaming.StreamHub` is the orchestrator's `StreamListener` in `serve`. `GET /runs/{id}/stream`
  merges hub `delta` previews with persisted events and ends with `end` when the run is settled.
  Separate worker processes publish nothing to the hub.
- `GET /runs/{id}/messages` (`_transcript`): user and assistant turns only, plus `ask_user`
  questions and answers. Never the system prompt, tool output, or hook context.
- CORS (`[api] cors_origins`): the `send` wrapper adds headers for allowed origins. Preflight
  (OPTIONS) is answered before auth only for allowed origins.
- `RunRecord.end_user` (`create_run(end_user=)`, `RunQuery.end_user`) is audit and listing data,
  not an identity check. SQLite and Postgres filter on the snapshot JSON (no index).
- `notifications.WebhookNotifier`: workers notify settled runs. HMAC signature; best effort;
  the payload holds no message text.

## 6. Workers and queue
- Add store methods `claim_next_run(worker_id, lease_seconds)`, `renew_lease`, and
  `release`. In SQLite, the claim is a single `UPDATE ... WHERE lease_expires_at < now`
  guarded by version.
- The worker loop claims a run, runs it, and renews the lease on a heartbeat. An expired
  lease lets another worker resume the run, which the existing resume semantics already
  make safe.
- Graceful shutdown: stop claiming, finish or checkpoint, release.

### Reliability (Phase 12c)
- **Attempts.** `run_queue.attempts` (SQLite migration 6, Postgres migration 3) is
  incremented by `claim` and survives a requeue. A new queue entry starts at 0. A
  worker dead-letters a run whose attempts exceed `max_attempts` (default 5) through
  `orchestrator.dead_letter`: one checkpoint with `RUN_DEAD_LETTERED` (an annotation)
  and `RUN_FAILED`.
- **Atomic enqueue.** `create_run(enqueue=True)` sets `RunRecord.dispatch = "queue"`
  and uses `store.create_run_and_enqueue` when present (one transaction).
  `RedactingRunStore` overrides it explicitly so the event stays masked; `__getattr__`
  delegation would skip redaction.
- **Reconciler.** `store.orphaned_runs(stale_before)` returns top-level runs that are
  `dispatch = "queue"`, queued or running, and not in `run_queue`.
  `Worker.reconcile` (at start and every `reconcile_seconds`) re-enqueues them.
  Inline (CLI) runs are never touched.
- **Cancel.** The heartbeat ticks every `cancel_poll_seconds`. It checks
  `orchestrator.is_cancelled` and cancels the execution task, and renews the lease
  every `lease_seconds / 3`. A cancelled run is released without requeue.

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

## Production hardening (Phase 9)

- **Where it lives:**
  - storage migrations and the Postgres store: `persistence/`;
  - principals, roles, and quotas: `access.py`;
  - secret references: `secrets.py`;
  - redaction: `redaction.py`;
  - the egress host policy: `tool_sources.host_policy_from`.
- **Docs:** see `skills/production-hardening/SKILL.md` and `docs/DEPLOYMENT.md`.
- **Adding a store method:** add it to `tests/test_store_conformance.py` and implement it in
  all three stores (memory, SQLite, Postgres).

## Phase 13: rate limits and deadlines

- `ratelimit.RateLimiter`: a token bucket per key (`key:<principal>`, or `addr:<client>`
  without keys or on failed authentication), checked in `HarnessAPI._dispatch` after
  `/healthz` and `/readyz`. 429 carries `Retry-After`. Per process.
- `AgentOrchestrator.expire_pending_input()` lists `awaiting_input` top-level runs,
  follows a waiting subagent through `pending_approval`, and applies `on_input_timeout`.
  `deny` goes through `resolve_approval(actor="timeout")`; `Worker.expire_input()` runs
  after each reconcile and enqueues the runs that continue. CLI: `agent-harness expire`.
