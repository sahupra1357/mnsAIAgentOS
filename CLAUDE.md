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
- **Context window (added in Phase 2).** Only profiles with `context_window` are trimmed;
  others still send the full history. For Ollama's OpenAI-compatible mode, the server's
  own context (`OLLAMA_CONTEXT_LENGTH`) must also be at least that large, or the server
  truncates silently. See `skills/context-management/SKILL.md`.
- `ModelResolver.resolve` is synchronous; `Execution.md` shows `await` but the code does not.
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

**Phase 2 – Context window** [context-management] — steps 1, 2, 4 implemented on `phase-2-context-window`; step 3 (summarization) deferred
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

**Phase 4 – Operations** [harness-operations] — split into stacked PRs: 4a (items 1, 2, 10) on `phase-4a-durability-versioning`; 4b (items 3, 4) on `phase-4b-telemetry-streaming`; 4c (items 5, 6) on `phase-4c-api-workers`; 4d (items 7–9)
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

**Phase 5 – Optional LiteLLM SDK adapter** [llm-providers]
1. Add `adapter = "litellm"` behind an optional `[litellm]` extra, imported lazily, with
   `num_retries=0` (the harness owns retries) and cost taken from LiteLLM's response
   cost when present.

**Phase 6 – Interactive and extensibility features** [interactive-extensibility]
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
    and a CLI `config show --effective`.
14. Plugins: a manifest bundling templates, subagent definitions, hooks, and skills.
    Nothing is enabled unless explicitly installed and enabled in settings.
15. Usage display: `status`/API endpoint showing per-run tokens, cost, cache-hit rate,
    and steps against the budget.

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
| Tests, evaluations, release checks            | `skills/harness-verification/SKILL.md`          |

The same skills are exposed to Claude Code via symlinks in `.claude/skills/`.
