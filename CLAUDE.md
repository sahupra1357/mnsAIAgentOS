# CLAUDE.md

Guidance for Claude Code in this repository. The canonical rules live in
[`AGENTS.md`](AGENTS.md) (architecture boundaries, implementation standards,
verification). Read it first; everything below supplements it and never overrides it.

## Project in one paragraph

`agent_harness` is a provider-neutral, durable, async agent runtime (Python ≥ 3.12,
stdlib only, no runtime dependencies). `AgentOrchestrator` drives a persisted
state machine (`queued → running → awaiting_input | completed | failed | cancelled |
budget_exhausted`) over three protocols in `domain.py`: `ModelResolver`,
`ToolExecutor`, `RunStore`. Models are selected by named profiles in `models.toml`;
tools go through `ToolRuntime` (schema validation, policy, timeouts, idempotency);
runs and append-only events persist in SQLite or memory. An optional coding mode adds
workspace-safe file/git/command tools with Docker sandboxing and approval pauses.

## Target LLM backends

The harness must work with these three model sources, all selected by config:

| Source    | Server                         | Default endpoint                         | Auth                                          |
|-----------|--------------------------------|------------------------------------------|-----------------------------------------------|
| Ollama    | `ollama serve`                 | `http://127.0.0.1:11434/v1`              | none                                          |
| llama.cpp | `llama-server --jinja ...`     | `http://127.0.0.1:8080/v1`               | optional `--api-key` bearer                   |
| Modal     | **vLLM or SGLang** on Modal    | `https://<ws>--<app>-<fn>.modal.run/v1`  | **both**: bearer key **and** Modal proxy headers |

All three expose OpenAI-compatible `/chat/completions`, so they share
`OpenAICompatibleClient`; provider quirks belong in thin adapters inside
`model_gateway/`, never in the orchestrator. The Modal adapter supports a
`server = "vllm" | "sglang"` setting and sends both the bearer key and the
`Modal-Key`/`Modal-Secret` proxy headers. An optional **LiteLLM** route (a proxy
through the existing adapter, or an SDK adapter as an optional extra) reaches providers
that are not OpenAI-compatible. Read `skills/llm-providers/SKILL.md` before touching
any of this.

## Hard rules for this repo

- **Never delete or rename existing files, tests, docs, or skills.** Extend instead.
  The original code was produced by Codex; preserve it and its tests.
- Plan before coding when the user asks for a plan; do not edit `src/` or `tests/`
  until they approve.
- Tests must stay network-free: use `ScriptedModelClient` or a fake `HttpTransport`.
  Never call a real Ollama, llama.cpp, or Modal endpoint from `tests/`.
- No real credentials, Modal workspace names, or tokens in source, tests, examples, or logs.
- Do not change `domain.py` contracts casually. If a new field is needed (e.g. on
  `ModelRequest`), add it with a default so existing callers and tests keep working.
- Stay stdlib-only unless a dependency clearly earns its place; ask first.

## Commands

```bash
# full test suite (must pass before handoff)
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests

# one module
PYTHONPATH=src python3 -m unittest tests.test_model_gateway -v

# CLI
PYTHONPATH=src python3 -m agent_harness profiles --config examples/models.toml
PYTHONPATH=src python3 -m agent_harness run --config models.toml --model-profile ollama "..."
PYTHONPATH=src python3 -m agent_harness resume  --config models.toml RUN_ID
PYTHONPATH=src python3 -m agent_harness approve --config models.toml RUN_ID --allow
```

## Where things are

| Concern                         | File(s)                                              |
|---------------------------------|------------------------------------------------------|
| Domain values & protocols       | `src/agent_harness/domain.py`                        |
| State machine, budgets, resume  | `src/agent_harness/orchestrator.py`                  |
| TOML/env config, `ModelProfile` | `src/agent_harness/config.py`                        |
| Adapter registry                | `src/agent_harness/model_gateway/gateway.py`         |
| OpenAI-compatible adapter       | `src/agent_harness/model_gateway/openai_compatible.py` |
| HTTP boundary (fakeable)        | `src/agent_harness/model_gateway/transport.py`       |
| Tool runtime/policy/validation  | `src/agent_harness/tools/`                           |
| Coding tools + Docker sandbox   | `src/agent_harness/coding/`                          |
| SQLite / memory stores          | `src/agent_harness/persistence/`                     |
| Composition root                | `src/agent_harness/application.py`                   |
| CLI                             | `src/agent_harness/cli.py`                           |

## Known pitfalls in the current code

- **Layered settings (6d).** `build_application` and the CLI also read
  `~/.agent-harness/settings.toml` and `/etc/agent-harness/settings.toml`. For
  hermetic tests, set `AGENT_HARNESS_USER_SETTINGS` / `AGENT_HARNESS_ENTERPRISE_SETTINGS`
  to nonexistent paths, as CI does. `load_config()` without `layered=True` reads only
  the given file.

- **One retry budget (Phase 3).** `OrchestratorSettings.max_model_attempts` bounds
  *provider requests* per step and profile. Adapters set
  `RetryableModelError.provider_attempts`, so adapter retries count against the same
  budget instead of multiplying. With `max_retries = 2`, the adapter's three attempts
  exhaust the default budget of 3, and the next fallback profile (if any) is used.
- **Model timeouts (fixed in Phase 1).** `ModelRequest.timeout_seconds` now defaults to
  `None`, which means "use the profile's timeout". Previously a hidden 60 s default
  capped every profile.
- **Cost is estimated (Phase 3).** Profiles with `pricing` get `Usage.cost_usd` from
  token prices and/or `gpu_usd_per_second × request latency`, which includes retries
  and cold start and is an upper bound under concurrency. Without `pricing`, cost stays
  unknown and `max_cost_usd` cannot trigger.
- **Repeated-call detection (Phase 3).** More than `max_identical_tool_calls` (5)
  identical calls since the last non-read-only call are refused, and the run fails
  after `max_refused_repeats` (2) refusals. Polling tools may need these raised, or set
  to `None`.
- **Runaway guards.** A reply with more than `max_tool_calls_per_turn` (32) tool calls is
  refused whole, and counts toward `max_refused_repeats`. Example profiles cap output with
  `parameters.max_tokens = 1024` (Ollama native: `num_predict`); an
  `AGENT_HARNESS_MODEL__X__PARAMETERS` variable replaces that table. Without a cap, vLLM on
  Modal keeps generating after the harness times out, because the proxy does not cancel
  abandoned requests. Adapter retries behind a success are recorded as
  `provider_attempts` / `retried_errors` in the event and telemetry.
- **Context window (added in Phase 2).** Only profiles with `context_window` are trimmed;
  others still send the full history. For Ollama's OpenAI-compatible mode, the server's
  own context (`OLLAMA_CONTEXT_LENGTH`) must also be at least that large, or the server
  truncates silently. See `skills/context-management/SKILL.md`.
- **Summarization (Phase 2.3).** It is off unless `[summarization] enabled = true`, and
  applies only to profiles with `context_window`. The summary is a model call that
  counts against the budget and is persisted on the run (`context_summary`). A test
  model must answer requests with `metadata["purpose"] == "summary"`.
- `ModelResolver.resolve` is synchronous (no `await`). `Execution.md` was brought up to date
  with the Phase 1–6 flow and now shows it correctly.
- **Subagent nesting (6c).** Subagents exist only when `[[agents]]` are defined; the
  model is then offered an `agent` tool. `[subagents] max_depth` defaults to 2, so a
  subagent is itself offered `agent` and can start its own children. Set
  `max_depth = 1` to let only the top-level run delegate. `examples/subagents.toml` is a
  complete example.
- **Quotas (Phase 10).** Admission is atomic in the store (`reserve`). A run's budget is
  capped to its grant, so a new run cannot overshoot the day's remainder. One model call can
  still overshoot, because tokens are known only after it returns. Changing the orchestrator's
  quota mapping takes effect at the next step boundary.
- **Docker mounts (fixed in Phase 10).** `--mount ...,rw` is invalid; bind mounts are read-write
  by default. The coding runtime used it, so its Docker backend failed against a real daemon.
  Coding images need the tools the agent runs: `python:3.12-slim` has no `git`.
- **Overrides (Phase 13).** `AGENT_HARNESS_SET__SECTION__KEY=value` sets any table value
  (lower-cased names, TOML-parsed values). Tests that build configs must not leak such
  variables; the unit stage already clears `AGENT_HARNESS_*`.
- **Context overflow (Phase 13).** A provider "context too long" error is retried with a
  smaller view (2 retries by default, counted in `max_model_attempts`). The cap is persisted in
  `model_responded` events and applies for the rest of the run on that profile.
- **Overflow caps (Phase 13–14).** A cap is raised until the pinned messages, tools, and
  newest turn fit (`_fitting_cap`); if nothing can be removed, the provider's error ends
  the step instead of a pointless retry. Server counting is skipped while the estimate is
  under half the limit.
- **Baselines and HumanEval (Phase 16b).**
  - `baselines/` and `.gstack/` are gitignored; baselines are per machine.
  - Every `[[benchmark]]` needs a `baseline` path (a test enforces it).
  - HumanEval checks `docker info` before it starts, and `SandboxError` (exit 125, daemon
    messages, or a timeout with Docker down) makes an item an error, not a wrong answer.
- **Validation progress (Phase 16c).**
  - `bench/progress.py` (`PipelineProgress`) draws the live status line.
  - `run_pipeline(progress=...)` runs subprocess stages with `asyncio.to_thread`, so the clock
    keeps ticking; do not call `_run_command` directly from async code.
  - Estimates come from the previous run's `timings.json`.
  - Print through `progress.write()` while it is active, so lines do not tear the status line.
- **Companion profiles (Phase 16).** `config.models` also holds hidden `NAME.embed` / `NAME.guard`
  profiles. Use `config.chat_models()` (not `config.models`) wherever profiles are listed or offered
  to users. A single chat profile with companions still counts as one profile for the default.
- **Durable idempotency (Phase 4a).** `ToolRuntime` backs its in-memory cache with an
  `IdempotencyStore`. `build_application` attaches the SQLite store. A crash *after* a
  handler succeeds but *before* the checkpoint no longer repeats the side effect. A
  crash *during* the handler still can, so downstream services should honor the
  idempotency key too.

## Implementation roadmap

Each phase goes on its own branch (`phase-N-<slug>`), is opened as a PR against `main`,
and is merged only after the user approves. Every phase must keep the full suite green
and add network-free tests. Each item's details are in the skill in brackets.

**Phase 0 – Housekeeping**
1. Commit `CLAUDE.md`, `.claude/`, and the new skills. Remove their lines from
   `.git/info/exclude`, and add `.mypy_cache/`/`.ruff_cache/` to `.gitignore`.

**Phase 1 – Model sources** [llm-providers, model-gateway] — implemented on `phase-1-model-sources`
1. Add optional `ModelProfile` fields with defaults: `headers_env`, `context_window`,
   `cold_start_timeout_seconds`, `pricing`, `fallback`, `server` (Modal only).
2. Refactor `OpenAICompatibleClient` so subclasses can add headers and normalization
   hooks without changing its behavior for existing tests.
3. Add `OllamaClient` (tolerate missing tool-call IDs and object arguments; optional
   native `/api/chat` mode for `num_ctx`).
4. Add `LlamaCppClient` (503-while-loading is retryable; `/health` probe).
5. Add `ModalClient` (bearer + `Modal-Key`/`Modal-Secret`, cold-start timeout,
   502/503/504 retryable, `server = vllm|sglang` selects its health path and
   tool-parser guidance).
6. Register `ollama`, `llama_cpp`, `modal` in `ModelGateway` defaults.
7. Add the CLI `doctor --model-profile X` command for a live reachability, model, and
   tool-call smoke test (never run in the unit suite).
8. Add example profiles for all three to `examples/models.toml` and `.env.example`,
   plus a `examples/modal/` deploy script for vLLM and SGLang with placeholder names.
9. Add tests with a fake transport for each provider's quirks and header handling.
10. Set up prefix caching in the Modal deploy script: vLLM `--enable-prompt-tokens-details`
    (prefix caching itself is on by default, so do not disable it); SGLang leaves the radix
    cache on. For Modal containers, use a long `scaledown_window`, optional
    `min_containers=1`, `@modal.concurrent(max_inputs=...)` to serve many requests from one
    container, and a low `max_containers` so a run's steps hit the same KV cache. Document
    `--gpu-memory-utilization`, `--max-model-len`, and `--kv-cache-dtype fp8`.
11. Parse cached-token counts in adapters (vLLM `usage.prompt_tokens_details.cached_tokens`,
    SGLang's equivalent; verify the field against the deployed version). `doctor` sends the
    same prompt twice and reports whether the second call hit the cache.

**Phase 2 – Context window** [context-management] — steps 1, 2, 4 implemented on `phase-2-context-window`; step 3 (summarization) on `phase-2-summarization`
1. Add the `ContextBuilder` protocol, and inject it with a default that sends
   everything (today's behavior).
2. Add a `BudgetedContextBuilder`: token estimate, keep system and first user message,
   keep tool-call/result pairs together, trim large tool outputs, drop oldest turns.
3. Optionally summarize with a persisted summary event so resume is deterministic.
4. Make trimming cache-friendly: keep the prefix fixed (system prompt, tool specs,
   first user message); trim in large chunks, only when a threshold is crossed, never a
   sliding window each step; then append-only again. No timestamps or run IDs in the
   system prompt. Tool specs in a fixed order with canonical JSON.

**Phase 3 – Reliability and cost** [durable-agent-orchestration, model-gateway] — implemented on `phase-3-reliability-cost`
1. Retry in one place only: the adapter retries transport errors, and the orchestrator
   retries only after the adapter gives up (or make adapter retries 0 by default).
2. Add cost estimation: per-token prices or Modal GPU-seconds × rate → `Usage.cost_usd`,
   so `max_cost_usd` works.
3. Add repeated-action detection: the same tool with the same arguments N times →
   inform the model, then stop the run.
4. Add model fallback chains (`fallback = ["modal", "llamacpp"]`) only on retryable
   exhaustion, with an event recording the switch.
5. Add capability routing: named routes (`fast`, `strong`) that map to profiles.

**Phase 4 – Operations** [harness-operations] — split into stacked PRs: 4a (items 1, 2, 10) on `phase-4a-durability-versioning`; 4b (items 3, 4) on `phase-4b-telemetry-streaming`; 4c (items 5, 6) on `phase-4c-api-workers`; 4d (items 7–9) on `phase-4d-artifacts-memory-evals`. All of Phase 4 is implemented.
1. Store tool idempotency results durably in the run store (SQLite table), not only in
   process memory.
2. Record prompt and config versions on each run: hashes of the system prompt, tool
   specs, and model profile.
3. Stream tokens: optional `ModelClient.stream()` using SSE; the orchestrator still
   checkpoints only the final assembled response.
4. Add observability: structured JSON logs, per-step trace/span IDs, and a metrics
   hook; OpenTelemetry exporter as an optional extra.
5. Add an HTTP API (`api.py`): create/get/cancel/resume runs, list events, approve or
   deny, and SSE event stream; server framework as an optional extra.
6. Add workers and a queue: lease-based claim of queued runs in SQLite so several
   workers can process runs safely; heartbeat and lease expiry → resume.
7. Store large tool outputs as artifacts, with references in events.
8. Add long-term memory: a `MemoryStore` protocol with a simple SQLite FTS
   implementation, exposed to the model as read-only tools.
9. Add an evaluation runner: scenario files → scripted or live model → pass/fail report.
10. Add an optional `Usage.cached_input_tokens` field (default `None`), persisted in
    events, plus a per-run and per-profile cache-hit-rate metric in telemetry.

**Phase 5 – Optional LiteLLM SDK adapter** [llm-providers] — implemented on `phase-5-litellm`
1. Add `adapter = "litellm"` behind an optional `[litellm]` extra, imported lazily, with
   `num_retries=0` (the harness owns retries) and cost taken from LiteLLM's response
   cost when present.

**Phase 6 – Interactive and extensibility features** [interactive-extensibility] — split into PRs that each target `main`:
6a (items 1, 2, 15) on `phase-6a-conversation-control`; 6b (items 3, 5, 6, 7) on `phase-6b-hooks-permissions`; 6c (items 4, 8) on `phase-6c-subagents-tasks`; 6d (items 9, 10, 11, 13, 14) on `phase-6d-instructions-settings-plugins`; 6e (item 12) on `phase-6e-rewind`; 6f (item 9 subdirectory discovery) on `phase-6f-instruction-discovery`. All of Phase 6 is implemented.
Order matters: 6.1 is the foundation for most of the rest.
1. Continue a conversation: `continue_run(run_id, message)` appends a user message to a
   `completed` run and moves it back to `running` (new event `USER_MESSAGE_ADDED`). Add a
   CLI `continue` command.
2. Cancel and steer mid-run: the loop reloads the run status or a signal before each model
   or tool step and stops cleanly on cancel. Queued steering messages are injected at the
   next model-turn boundary. Add a CLI `cancel` command.
3. Hooks: a `HookRunner` protocol with the events `session_start`, `user_prompt`,
   `pre_tool`, `post_tool`, `pre_model`, `post_model`, `pre_compact`, `stop`, and
   `subagent_stop`. Hooks can allow, deny, or modify, or add context. Hook commands
   use structured argv with a timeout, and every decision is recorded as an event.
4. Subagents: an `agent` tool spawns a child run (own profile, tools, instructions,
   budget; `parent_run_id` link); the parent pauses on it and receives only the final
   result. Child budgets count against the parent. Enforce a depth limit.
5. Detailed permission rules: allow/ask/deny rules matching tool name and
   argument patterns (e.g. `run_command(argv[0]=pytest)`), where deny beats ask beats
   allow, loaded from layered settings. Rules sit on top of the risk policy; they
   never override hard guardrails.
6. Permission modes: `default | accept_edits | plan | bypass` (bypass still keeps
   guardrails and deny rules). Mode is stored per run and can be changed with an event.
7. Plan mode: only read-only tools; the model ends with a plan (the `exit_plan` tool) →
   run pauses `awaiting_input` → approve switches the mode and continues, reject returns
   feedback.
8. Task tracking: a `todo_write` read-only-state tool whose list is persisted on the run,
   emitted as an event, and shown in the CLI/API.
9. Instruction files: load `AGENT.md`-style files (user → project → subdirectory) into the
   system prompt at run creation, and hash them into the Phase 4 version record.
10. Skills: discover `skills/*/SKILL.md`, put only the name and description in the
    context, and load the full body through a `load_skill` tool (progressive disclosure).
11. Prompt templates / slash commands: named templates with arguments (`/review <path>`),
    expanded at the CLI/API boundary into the user message.
12. Rewind: before each file-modifying tool call, snapshot the touched files to the
    artifact store. `rewind(run_id, step)` restores the files and truncates the
    conversation view, recorded as an event without deleting history.
13. Layered settings: enterprise → user → project → local merge, with defined precedence
    and a CLI `config show --config FILE` (there is no `--effective` flag; it always prints
    the merged layers with the layer each key came from, and keys left at code defaults are not listed).
14. Plugins: a manifest bundling templates, subagent definitions, hooks, and skills.
    Nothing is enabled unless explicitly installed and enabled in settings.
15. Usage display: `status`/API endpoint showing per-run tokens, cost, cache-hit rate,
    and steps against the budget.

**Phase 7 – Tool platform** [tool-platform] — on `phase-7-tool-platform`
1. Persistent tool catalog (`tool_catalog` table): what every source provides, spec
   hashes, trust, operator enable/disable that survives restarts, and unavailable
   tools kept for old runs. Each run records the tool versions it was offered.
2. Config-driven tool sources: `[[tools]]` with kind `python` (module:function), `http`
   (declarative JSON tools), and the built-in `web_search` (Tavily, key from env).
   CLI `tools list|sync|enable|disable`.
3. Stdlib MCP client (`mcp/`): stdio and streamable HTTP, tool discovery with
   pagination, schema normalization to the validator's subset, `server__tool` names,
   approval by default unless listed `read_only`, lazy server start, clean shutdown.
4. Untrusted tool output: `ToolSpec.trust`; untrusted results are wrapped in
   `<untrusted-tool-output>` tags and the system context says they are data.
5. Dynamic tool loading: `[tool_loading] mode = "dynamic"` with a pinned `core` set and
   `search_tools` / `load_tools` meta-tools (SQLite FTS5 ranking). Loaded tools are
   persisted on the run and appended in order (prefix-cache friendly).

**Phase 8 – Planning and delegation** [planning-delegation] — on `phase-8-planning-delegation`
1. Orchestrator-owned plan on the run (`Plan`, `PlanStep`) with `PLAN_CREATED`,
   `PLAN_STEP_UPDATED`, `PLAN_REVISED` events; `create_plan`, `update_plan`,
   `revise_plan` tools.
2. Plan-and-execute mode (`[planning]`, `run --plan`): plan first with read-only tools,
   then the orchestrator drives step by step, replans on failure (bounded), and asks for
   the final answer when all steps are done.
3. Orchestrator-initiated delegation: plan steps with an `agent` run as child runs
   without a model turn.
4. Parallelism: concurrent prefetch of read-only tool calls; several subagents (tool
   calls or ready plan steps) run concurrently with the remaining budget split.
5. Example subagents (`examples/subagents.toml`, `examples/agents/`) and the nesting
   note.

**Phase 9 – Production hardening** [production-hardening] — on `phase-9-production-hardening`
1. Versioned schema migrations for SQLite (and Postgres), plus a RunStore conformance
   suite run against every store.
2. Optional Postgres store behind a `[postgres]` extra, verified against a real server
   when `AGENT_HARNESS_TEST_POSTGRES_DSN` is set.
3. API principals with tenants and roles (viewer/operator/admin), run ownership, and
   the acting principal recorded on approvals and mode changes.
4. Per-tenant quotas (runs, tokens, cost per day; concurrent runs).
5. Secret references (`env:`, `file:`) and a network egress allowlist for HTTP tools,
   web search, and MCP over HTTP.
6. Redaction of secrets in persisted events and telemetry; retention with
   `agent-harness gc`. Encryption at rest is delegated to the volume or database.
7. Deployment: Dockerfile, compose file (API, workers, Postgres), `/readyz`, and
   `docs/DEPLOYMENT.md`.

**Phase 10 – Isolation, quota enforcement, memory backends** [production-hardening, tool-platform, coding-agent] — on `phase-10-isolation-quotas-memory`
1. Quota reservation: admission reads usage plus outstanding reservations, stores a
   reservation atomically (SQLite `BEGIN IMMEDIATE`, Postgres per-tenant advisory lock),
   and caps the run's budget to the grant; released when the run ends. `continue`
   re-admits. Tenant usage counts top-level runs only (subagents are in their parent).
2. Step-boundary quota checks: before each model turn the orchestrator re-checks the
   tenant's consumption (cached `quota_check_seconds`) and stops with `BUDGET_EXHAUSTED`.
3. Warm per-run containers (`sandbox/containers.py`): one container per run, reused with
   `docker exec`, removed when the run ends (or after `max_lifetime_seconds`).
   `[coding.docker] mode = "per_run"`; sandboxed Python tools
   (`[[tools]] kind = "python", sandbox = "container"`) run in the run's container via
   `agent_harness.sandbox_runner`. The current run is a context variable (`runcontext.py`).
4. Memory backends: `[memory] backend = "sqlite" | "postgres"` (Postgres full-text search);
   `agent-harness memory migrate --from sqlite:PATH` copies every note.

**Phase 11 – Operator and model-interface gaps** [harness-operations, llm-providers, model-gateway, interactive-extensibility] — four stacked PRs (11a → `main`, each later one on the previous branch):
11a (item 1) on `phase-11a-run-listing` — implemented; 11b (item 2) on `phase-11b-reasoning` — implemented;
11c (item 3) on `phase-11c-structured-output` — implemented; 11d (item 4) on `phase-11d-ask-user` — implemented. All of Phase 11 is implemented.
No classifier-based model routing: the user decided against it (extra call, loses the
prefix cache, overkill).
1. Run listing: an optional `RunLister` store protocol (`list_runs(RunQuery) -> RunPage`)
   in all three stores, with keyset cursors, top-level runs by default, CLI `runs`, and
   tenant-scoped `GET /runs`.
2. Reasoning content: `Message.reasoning` / `ModelResponse.reasoning` (default `None`),
   parsed from `reasoning_content` / `reasoning` / Ollama `thinking`, optional
   `<think>` splitting (`reasoning_tags`), `reasoning_history = "drop" | "keep"` (default
   `drop`, so the prefix cache is unaffected); CLI `--show-reasoning`.
3. Structured output: `ModelRequest.response_schema` and `tool_choice` (default `None`,
   so payloads are unchanged). A run's `output_schema` is sent as `response_format` when
   the run has no tools, otherwise as a `submit_result` tool. A plain answer that is valid
   JSON is accepted; otherwise the model is asked again (with tools, `tool_choice` forces
   `submit_result`), and the run fails after 2 rejections. The result goes to `run.output`
   (API `result`). Not combinable with plan-and-execute.
4. `ask_user(question, options?)` tool, off by default (`[interaction] ask_user = true`):
   pauses via the approval path with `kind = "question"`; `answer(run_id, call_id, text)`,
   CLI `answer`, API `POST /runs/{id}/answer`.

**Phase 12 – Guardrails, chat API, reliability, operations** [guardrails, harness-operations, durable-agent-orchestration, model-gateway] — PRs against `main`, in this order:
12a on `phase-12a-guardrails` — implemented; 12b (chat API) on `phase-12b-chat-api` — implemented; 12c (worker reliability) on
`phase-12c-worker-reliability` — implemented; 12d (gateway/operations) on `phase-12d-gateway-ops` — implemented.
All of Phase 12 is implemented. The `/ui` chat page is deferred.
1. Guardrails (`guardrails/`, `[guardrails]`, off by default; docs/GUARDRAILS.md): input guard on user text
   (injection, prompt extraction, hidden characters, encoded payloads, flooding, PII, denylist) before each model
   turn, tracked by `RunRecord.guarded_through`; output guard (canary marker in the system prompt, secrets, PII,
   denylist) before a reply is recorded; untrusted tool output scanned; taint mode (side-effect tools need approval
   after untrusted output, even in bypass). Blocked input gets `refusal_message` and the run completes
   (`guardrail_refused` state event). Optional guard model (Llama Guard 3 via a profile, `parallel` option).
   Events never contain matched text. `guardrails` bench suite for red-team detection rates.
2. Chat API: `GET /runs/{id}/stream` (persisted events plus `StreamHub` token previews from in-process
   workers), `GET /runs/{id}/messages` (chat transcript), `[api] cors_origins`, `user` / `RunRecord.end_user`
   (audit and listing filter; per-user quotas not included), `[notifications]` webhooks from workers.
3. Worker reliability: `run_queue.attempts` (claims) with dead-lettering past `max_attempts` (`run_dead_lettered`
   + `run_failed`); `create_run(enqueue=True)` → `create_run_and_enqueue` in one transaction and
   `RunRecord.dispatch = "queue"`; `Worker.reconcile` requeues stale orphaned queue runs (never inline/CLI runs);
   the heartbeat polls `is_cancelled` and stops in-flight calls.
4. Gateway/operations:
   - profile `max_concurrency`, `retry_jitter`, `max_retry_after_seconds` (Retry-After honoured), and
     `structured_output = strict|non_strict|prompt` (all hash-stable at their defaults);
   - images via `Message.parts`, `ContentPart`, CLI `--image`, and API `images`;
   - `doctor --max-tokens/--strict`, an "ran out of tokens" message, and a router model check; smoke `strict` and
     `max_tokens`;
   - bench resume re-runs `error`/`failed` items;
   - `GET /runs/{id}/artifacts/{aid}` (the global route is admin-only with tenant keys);
   - `gc` removes old unreferenced artifacts and snapshots;
   - eval scenarios take `output_schema`, `answers`, `[guardrails]`, and `expect.result`.

**Phase 13 – Limits and deadlines** [durable-agent-orchestration, context-management, harness-operations, production-hardening] — on `phase-13-limits`
1. Context-overflow recovery: `ContextOverflowError` (adapters classify vLLM/SGLang/llama.cpp/OpenAI
   messages and parse the stated sizes); the step is retried with a smaller, persisted input cap
   (`context_overflow_retries`, `context_overflow_shrink`, `context_overflow_patterns`).
2. Deadlines for paused runs: `[interaction] approval_timeout_seconds`, `question_timeout_seconds`,
   `on_timeout = deny|cancel|fail`; `expire_pending_input()`, worker reconcile, CLI `expire`.
3. API rate limits: `[api] rate_limit_per_minute`, `rate_limit_burst` (token bucket per key, 429 +
   Retry-After; `ratelimit.py`).
4. Per-user quotas: `[user_quotas]` (+ `[user_quotas.users.NAME]`), reserved with the tenant quota;
   stores gain `user_usage` and `reserve(..., end_user=)` (SQLite migration 7, Postgres 4).
Also `[orchestrator]` (all `OrchestratorSettings`) and `AGENT_HARNESS_SET__SECTION__KEY` overrides.

**Phase 14 – Context, evals, memory, tools** [context-management, model-gateway, harness-verification, harness-operations, tool-platform] — on `phase-14-context-evals-tools`
1. `[context]` section → `ContextPolicy` (`context_policy_from_config`), used by the app, the
   orchestrator's overflow cap, the summarizer, and benchmarks.
2. Profile `token_counting = "estimate" | "server"`: llama.cpp `/apply-template` + `/tokenize`,
   vLLM `/tokenize`; over the window → `CONTEXT_CAPPED` event (annotation) and a rebuilt view.
3. Scenario options `[orchestrator]`, `[interaction]`, `ask_user`, `wait_seconds`,
   `error = "context_overflow"`, `expect.event_payloads`; new examples in `examples/evals/`.
4. `qa` suite `match = "judge"` (`judge_profile`, `judge_max_tokens`, per-record `rubric`).
5. `[memory] search = keyword|semantic|hybrid`, `embedding_profile`, `vectors`;
   `semantic_memory.py`, client `embed()`, CLI `memory reindex`.
6. MCP `resources` / `prompts` (read-only `server__mcp_*` tools) and `list_changed` re-sync.
7. `[web_fetch]`: text of one page; private addresses refused; redirects re-checked per hop.
Live checks (on `phase-14b-live-checks`): bench suites `context` (overflows a small window with
JSON/code/prose filler; reports omitted/capped/overflow counts) and `memory` (save, embed, find
with different wording). Example profiles `llamacpp`/`modal` use `token_counting = "server"`, and
`[models.embed]` is `nomic-embed-text` on Ollama.

**Phase 15 – Defaults, feature validation, chat clients, web UI** [interactive-extensibility, harness-verification, harness-operations] — approved by the user; each part on its own branch and PR.
Code defaults stay off (opt-in). "Everything on" is a config file, not a change to dataclass
defaults, so existing tests, version hashes, and the prefix cache are unaffected.
- **15a CLI defaults** (`phase-15a-cli-defaults`) — implemented:
  1. `--config` becomes optional. Order: `--config` → `AGENT_HARNESS_CONFIG` → `./models.toml` →
     `~/.agent-harness/models.toml`. If none is found, the error lists the paths that were tried.
  2. `agent-harness "prompt"` means `agent-harness run "prompt"`. When the first argument is not a
     subcommand or an option, `run` is inserted before argparse runs. Existing subcommands keep working.
  3. `agent-harness` with no arguments starts an interactive chat REPL: the first message
     creates a run, and later ones use `continue_run`. Approvals and `ask_user` are asked
     inline. `/exit`, `/new`, `/status` are REPL commands; other `/name` inputs go to the templates.
  4. The default profile is still `default_model_profile` from the config; the examples use
     `llamacpp`. A profile is switched off with `enabled = false` (or
     `AGENT_HARNESS_MODEL__NAME__ENABLED=false`). It is removed at load, routes to it are
     dropped, and fallback chains skip it (`HarnessConfig.disabled_models`). The switch is
     not a `ModelProfile` field, so profile hashes do not change.
- **15b Full-feature config** (`phase-15b-full-config`) — implemented: the repository's root `models.toml`
  turns on every feature that needs nothing outside the harness. It is the file `agent-harness` finds
  from the repository root. Relative paths point into `examples/`: `hooks/audit_hook.py`, `agents/`,
  `skills/`, `templates/`, `tools/`. It covers:
  - profiles: llama.cpp default, Ollama fallback, `embed`; Modal and Llama Guard off via `enabled = false`
  - guardrails with taint mode, two audit hooks, permission rules, `ask_user` and deadlines, tasks
  - three subagents, instructions, skills, templates
  - summarization and `[context]`
  - python and http tool sources, `web_fetch`, `web_search` (needs `TAVILY_API_KEY` at call time),
    hybrid memory, artifacts, dynamic tool loading
  - coding in a `per_run` Docker container (`python:3.12`, which has git)
  - egress allowlist, redaction, quotas, API rate limit and CORS
  API keys, webhooks, MCP, the guard model, and Postgres are commented out at the end of the file,
  each with its prerequisite. Planning stays a run mode (`--plan`).
  `tests/test_full_config.py` pins each feature on, checks every referenced path, and builds the app.
  Code defaults are unchanged.
- **15c Per-feature validation** (`phase-15c-feature-validation`) — implemented:
  1. `[[feature]]` entries in `examples/validation.toml` (parsed by `bench/features.py`) give each
     feature its `tests` (modules), `scenarios` (file stems in `examples/evals/`), `smoke` (profiles),
     and `benchmarks` (names). The plan defines 14 features.
  2. `agent-harness-bench pipeline PLAN --feature NAME` (repeatable) runs only those checks, and skips
     the repo-wide static stage. `--offline` keeps static, unit, and scenarios.
     `agent-harness-bench features PLAN` lists the features. Without `--feature` the master run is unchanged.
  3. `scripts/validate.sh [feature ...] [--offline] [--list]` wraps the pipeline command;
     `VALIDATION_PLAN` selects another plan.
  4. `tests/test_feature_validation.py` fails when a test module, scripted scenario, smoke profile,
     or benchmark is in no feature.
  5. Application-backed scenarios:
     - **Keys:** `config`, `[harness]` (merged with `load_config(overrides=)`), and `[workspace]`.
     - **How they run:** `agent-harness eval` builds the app with
       `build_application(config_overrides=, model_resolver=)`. The scripted model is
       `_ScenarioModel`, which embeds with hashed-word vectors. Workspace and artifacts are temporary.
     - **Bare-only keys:** `guardrails`, `orchestrator`, `interaction`, `ask_user`, and
       `wait_seconds` are rejected next to `config`/`[harness]`.
     - **Eval input:** `eval` takes several files.
  6. 12 new app-backed scenarios: hooks, permission deny/ask, plan mode + snapshot, subagent, tasks,
     memory, skills, dynamic tool loading, `ask_user`, output redaction.
  7. The plan's live config is the root `models.toml`, with llama.cpp smoke and gsm8k/bfcl/guardrails/
     humaneval/context/memory benchmarks. `[scenarios] config = "models.toml"` keeps bare scenarios
     on `examples/models.toml`.
- **15d OpenAI-compatible chat endpoint** (`phase-15d-openai-chat`) — implemented in `openai_chat.py`,
  routed by `api.py`:
  - **Endpoints:** `GET /v1/models` lists `agent-harness` (the default profile), every profile, and
    every route. `POST /v1/chat/completions` handles JSON and SSE `chat.completion.chunk` replies.
  - **Conversation mapping:** a new optional `ConversationStore` protocol (`find_conversation`,
    `save_conversation`) backs the `chat_conversations` table. It is SQLite migration 8 and Postgres
    migration 5, is in the memory store too, and is purged with its run.
  - **Finding the run:** the digest of the earlier messages, or an `X-Agent-Harness-Run` header. A miss
    creates a run seeded with the client's history.
  - **Pauses:** they are answered by the next chat message (`decision_from_text`).
  - **Sampling options:** ignored. Client tools are refused.
  - **Worker race fixed:** a run continued or approved between its final checkpoint and the
    worker's queue release was dropped from the queue. `Worker._requeue_if_reopened` puts it back.
  - **Tests:** `tests/test_openai_chat.py` (8 cases with in-process workers), a store conformance
    case, and a worker regression test.
- **15e Built-in web UI** (`phase-15e-web-ui`) — implemented. `GET /ui` (and `/` → 307 `/ui`) serves
  `src/agent_harness/ui/index.html`:
  - **Packaging:** vanilla JS, no build step. It ships as `[tool.setuptools.package-data]` and
    is read with `importlib.resources` and cached per process (restart `serve` after editing it).
  - **Headers:** CSP `default-src 'none'; connect-src 'self'; frame-ancestors 'none'`.
  - **Key handling:** the key is entered in the page and kept in `sessionStorage` (`localStorage`
    only if "remember" is ticked).
  - **What it uses:** only existing endpoints; streams are read with `fetch` so the bearer header
    is sent. It never uses `innerHTML` for model text.
  - **Compose:** `deploy/docker-compose.yml` has an optional `open-webui` service (`--profile chat`)
    pointed at `/v1`.
  - **Tests:** `tests/test_web_ui.py`. The page was also checked in a headless browser against a
    live server: chat, follow-up turns, budget panel, events, and mobile layout.

**Phase 16 – Self-contained profiles** [llm-providers, model-gateway, harness-verification] — on `phase-16-self-contained-profiles`.
One provider profile serves everything; nothing in `models.toml` depends on a second provider.
1. Optional `ModelProfile` fields `embedding_model`, `embedding_endpoint`, `guard_model`, and
   `guard_endpoint`:
   - the endpoints default to `endpoint`;
   - env forms are `AGENT_HARNESS_MODEL__X__EMBEDDING_MODEL` etc.;
   - they are left out of the profile hash while unset.

   Modal names three deployments: `examples/modal/vllm_server.py`, `vllm_embed_server.py`
   (`--runner pooling`), and `vllm_guard_server.py`.
2. Companion profiles:
   - **What they are:** `load_config` adds `NAME.embed` / `NAME.guard` to `models`: same adapter,
     credentials, and headers, another model.
   - **Visibility:** they are in `HarnessConfig.companions` and hidden from
     `chat_models()`, `profiles`, `GET /profiles`, and `/v1/models`.
   - **Rules:** a companion cannot be the default profile, and a real profile with that name is
     a conflict.
3. Defaults follow the default profile (a route is resolved first):
   - `[memory]` semantic/hybrid with an empty `embedding_profile` uses `<default>.embed`, or is
     a `ConfigError` if that profile has no `embedding_model`.
   - An empty `[guardrails.model] profile` uses `<default>.guard` when `guard_model` is set.
   - The memory benchmark embeds with the profile under test (`BenchContext.embedding_profile`).
4. The root `models.toml`:
   - drops the `embed` and `guard` profiles and `fallback = ["ollama"]`;
   - llama.cpp's router (models-max >= 2) serves `nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0` next
     to the chat model;
   - Ollama uses `nomic-embed-text`;
   - Modal needs a second deployment named by `embedding_endpoint`.
5. Validation follows the default profile:
   - `[[smoke]]` / `[[benchmark]]` entries without `profile` use `default_model_profile`.
   - Benchmark names and baselines take `{profile}`.
   - The feature smoke name is `default`.
   - `AGENT_HARNESS_DEFAULT_MODEL_PROFILE=ollama scripts/validate.sh` validates another provider.
6. `doctor` also checks the profile's embedding and guard models (`companion_checks`).
Tests: `tests/test_self_contained_profiles.py`.

## Validation

`docs/VALIDATION.md` describes six stages: static, unit, scenarios, smoke, benchmarks,
and gates.
- **CI:** `.github/workflows/ci.yml` runs the offline stages.
- **Live stages:** `agent-harness-bench pipeline examples/validation.toml`.
- **Code:** `src/agent_harness/bench/`.
- **Tests:** never download datasets or call live models in `tests/`. Use the
  synthetic samples in `examples/bench/`.

## Skills (read before editing the matching area)

| Area                                          | Skill                                           |
|-----------------------------------------------|-------------------------------------------------|
| Orchestrator, persistence, budgets, resume    | `skills/durable-agent-orchestration/SKILL.md`   |
| Model gateway, adapters, routing              | `skills/model-gateway/SKILL.md`                 |
| Ollama / llama.cpp / Modal specifics          | `skills/llm-providers/SKILL.md`                 |
| Context window, truncation, summarization     | `skills/context-management/SKILL.md`            |
| Tools, policy, approval                       | `skills/tool-runtime/SKILL.md`                  |
| Coding tools, Docker sandbox                  | `skills/coding-agent/SKILL.md`                  |
| Streaming, tracing, API, workers, memory      | `skills/harness-operations/SKILL.md`            |
| Conversation, hooks, subagents, permissions   | `skills/interactive-extensibility/SKILL.md`     |
| Tool catalog, tool sources, MCP, tool loading | `skills/tool-platform/SKILL.md`                 |
| Plans, delegation, parallel execution         | `skills/planning-delegation/SKILL.md`           |
| Migrations, Postgres, tenancy, security, deploy | `skills/production-hardening/SKILL.md`        |
| Tests, evaluations, release checks            | `skills/harness-verification/SKILL.md`          |
| Guardrails, prompt injection, jailbreaks, PII | `skills/guardrails/SKILL.md`                    |

The same skills are exposed to Claude Code via symlinks in `.claude/skills/`.
