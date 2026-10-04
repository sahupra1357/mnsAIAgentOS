# Execution flow

This document follows one run through the harness, from the command line to the final
answer. Diagrams of the same flow, including animated walkthroughs, are in
[`docs/diagrams/`](docs/diagrams/README.md).

The harness has four entry points:

1. CLI: `agent-harness run ...` (plus `continue`, `approve`, `resume`, `rewind`, and more)
2. Python: `build_application(...)`
3. HTTP API: `agent-harness serve` (needs the `[api]` extra)
4. Queue workers: `agent-harness worker`

All four use the same orchestrator, so a run behaves the same no matter how it started.

## 1. CLI entry point

Example:

```bash
agent-harness run \
  --config models.toml \
  --model-profile local \
  "Inspect the repository and explain the failing tests"
```

Useful options:
- **Permission mode:** `--mode plan|accept_edits|bypass`.
- **Planning:** `--plan` / `--no-plan` turn plan-and-execute on or off (section 18).
- **Output:** `--stream`.
- **Budgets:** `--max-steps`, `--max-wall-seconds`, `--max-tokens`, `--max-cost-usd`.
- **Templates:** `--arg key=value` fills a prompt template (`"/review app.py"`).
- **Storage:** `--database` takes a SQLite file or a `postgresql://` URL.

The call flow is:

```text
main()
  └── _run_command()
        ├── _build() → build_application()
        ├── expand_prompt()           # "/name args" → template text
        ├── create_run()
        ├── orchestrator.run()
        └── _print_result()           # answer to stdout, status to stderr
```

`--model-profile` accepts a profile name or a route name (`fast`, `strong`). A route is
resolved to its profile, and the route is recorded on the run.

Defaults (Phase 15a, 16e), applied in `main()` before any command runs:
- **`.env`:** when started as a program (`argv is None`), `load_dotenv()` (in `config/dotenv.py`)
  reads `./.env`, setting only variables that are not already set. `agent-harness-bench`
  does the same. `main([...])` (tests, the offline validation stages) never reads it.
- **Profile switch:** `load_config()` takes `AGENT_HARNESS_MODEL_PROFILE` (else
  `default_model_profile`) as the default profile. That profile's companions (`NAME.embed`, `NAME.guard`) then serve memory and the
  guard model, so one variable switches the whole provider.
- **Implicit command:** `_with_default_command()` turns `agent-harness` (no arguments)
  into `agent-harness chat`, and `agent-harness "prompt"` (first argument not a
  subcommand) into `agent-harness run "prompt"`.
- **Config discovery:** `discover_config_path()` (in `config/loader.py`) resolves a missing
  `--config` to `$AGENT_HARNESS_CONFIG`, `./models.toml`, or `~/.agent-harness/models.toml`,
  in that order. `load_config()` itself never searches.
- **Profile switches:** `load_config()` removes profiles with `enabled = false`, drops
  routes to them, and prunes them from fallback chains (`HarnessConfig.disabled_models`).

The chat flow (`cli/repl.py`):

```text
main() → _chat_command()
  └── ChatSession.loop()
        └── per message: send()
              ├── first message:  create_run()
              ├── later messages: continue_run(budget raised by one turn's steps and seconds)
              │                   (a failed or cancelled run starts a new conversation)
              ├── orchestrator.run()
              ├── _settle()        # approvals [y/N], plan approval, ask_user answers → resume()
              └── _show()          # streamed text or the last answer
```

Relevant files: `src/agent_harness/cli/main.py`, `src/agent_harness/cli/repl.py`

## 2. Application construction

`build_application()` creates the runtime components:

```text
Layered settings (enterprise → user → project → local)
    ↓
Environment overrides (AGENT_HARNESS_MODEL__*, AGENT_HARNESS_SET__SECTION__KEY)
    ↓
Profiles: enabled = false removed; companions NAME.embed / NAME.guard added from
embedding_model / guard_model; [memory] and [guardrails.model] default to the
default profile's companions (one provider serves everything)
    ↓
Plugins, templates, skills
    ↓
ModelGateway (Ollama, llama.cpp, Modal, OpenAI-compatible, LiteLLM)
    ↓
ToolRuntime (+ durable idempotency store, artifacts, memory, load_skill)
    ↓
Tool sources: [[tools]] python/http, web_search, [[mcp_servers]]
    ↓
Run store: SQLite, or Postgres (migrations run on open)
    ↓
Tool catalog sync (persistent; operator switches applied)
    ↓
AgentOrchestrator (context builder, summarizer, hooks, permissions, subagents,
                   file snapshots, system context, tool loading, planning,
                   [orchestrator] limits, quotas, input deadlines;
                   writes through the redacting store)
```

Relevant file: `src/agent_harness/application.py`

It performs these actions:

- **Settings:** loads `models.toml` merged with the layered settings. For hermetic runs,
  point `AGENT_HARNESS_USER_SETTINGS` and `AGENT_HARNESS_ENTERPRISE_SETTINGS` at missing
  files. `AGENT_HARNESS_SET__SECTION__KEY=value` then overrides any table value (for
  example `AGENT_HARNESS_SET__ORCHESTRATOR__MAX_TOOL_CALLS_PER_TURN=16`); `config show`
  lists each value with its source.
- **Summarizer:** validates the `[summarization]` section before opening any store.
- **Discovery:** finds enabled plugins, prompt templates, and skills.
- **Models:** builds the model gateway, with each profile's adapter, fallback chain, and
  routes.
- **Tools:** builds the default or coding tool runtime and attaches the SQLite idempotency
  store. It also registers `read_artifact`, the memory tools, and `load_skill` when
  enabled.
- **Instructions:** prepares the system context: instruction files (`AGENTS.md`, including
  subdirectory files when enabled) and the skills index.
- **Hooks and permissions:** configures hooks, permission rules, and the default permission
  mode.
- **Subagents and snapshots:** registers subagent definitions and the workspace snapshotter
  used by rewind.
- **Tool sources:** loads `[[tools]]` (Python modules and HTTP tools), the built-in
  `web_search`, and `[[mcp_servers]]`.
  - MCP tools come from the catalog cache until `tools sync`; the servers themselves start
    on first use.
  - Egress rules from `[network]` apply to HTTP tools, web search, and MCP over HTTP
    (section 19).
- **Store:** opens SQLite, or Postgres for a `postgresql://` URL or `[storage] url_env`.
  Versioned migrations run first. Configuration errors are raised before any connection
  is made.
- **Catalog:** records every registered tool in the persistent `tool_catalog` and hides
  tools an operator disabled.
- **Orchestrator:** creates it with a `BudgetedContextBuilder` for profiles that set
  `context_window`, the tool-loading and planning policies, the `[orchestrator]` loop
  limits, tenant and end-user quotas, the `[interaction]` deadlines, and a store wrapper
  that masks secrets in events (`[redaction]`).

The orchestration loop does not know whether the model is served by Ollama, llama.cpp,
Modal, or another provider.

## 3. Run creation

The CLI converts the user prompt into a message (after expanding a template, if one was
used):

```python
Message(
    role=Role.USER,
    content="Inspect the repository and explain the failing tests",
)
```

Then it creates a durable run:

```python
run = await app.orchestrator.create_run(
    model_profile="local",
    route=None,
    mode=None,                       # default permission mode from settings
    versions=app.run_versions("local"),
    messages=[message],
    budget=RunBudget(...),
    planning=None,                   # --plan / --no-plan, else [planning] enabled
    tenant="default",                # the API sets the caller's tenant
    created_by=None,                 # the API sets the caller's key name
)
```

`create_run` first **admits** the run against the tenant's quotas: runs, tokens, and cost
over 24 hours, plus concurrent runs.
- **Atomic:** in one store transaction it reads the tenant's usage plus other runs'
  outstanding reservations, grants at most the remainder, and records a reservation.
- **Budget:** the run's `max_tokens` and `max_cost_usd` are capped to the grant, so a new run
  cannot exceed what is left today.
- **Refusal:** HTTP 429 from the API, or exit code 2 from the CLI (section 20).
- **Release:** the reservation is released when the run finishes, and `continue` re-admits
  the run.

`create_run` does four things before any model is contacted:

1. **System context:** for top-level runs, the instruction files and the skills index are
   inserted as one system message. They are read once and stay fixed for the run. In
   plan-and-execute mode the planner instructions are added as well.
2. **`user_prompt` hook:** it can block the prompt or add context.
3. **Version record:** it computes the hashes of the system prompt, tool specs, model
   profile, and instruction context, plus the harness version.
4. **Persist:** it saves the run as queued, with a `RUN_CREATED` event:

```text
RunStatus.QUEUED
    ↓
RUN_CREATED event (versions, mode, route, template, planning, tenant,
                   tools: the spec hash of every tool offered)
```

This gives every run a stable `run_id`.

## 4. Run starts

`orchestrator.run(run_id)` loads the run from SQLite.

If the run is valid and not already finished or paused:

```text
QUEUED
  ↓
RUNNING
  ↓
RUN_STARTED event
  ↓
session_start hook (a deny fails the run; context is appended)
```

When a `RUNNING` run is resumed after a crash, the orchestrator first compares the current
config against the recorded versions and records `CONFIG_DRIFT` if they differ.

The orchestrator then enters its execution loop. Each iteration:

1. Reloads the run if another process changed it (a cancel, an approval, a steer).
2. Performs exactly one transition (`_advance`) and checkpoints it.

Relevant file: `src/agent_harness/orchestration/orchestrator.py`

## 5. Step boundary

At the start of each step the orchestrator:

- **Checks budgets:** steps, wall clock, tokens, and cost (section 13). It also checks the
  quotas: the end user's (`[user_quotas]`, for runs with an `end_user`) and the tenant's
  (`[[quotas]]`), consumption cached for a few seconds. Over a limit, the run stops with
  `BUDGET_EXHAUSTED`.
- **Runs pending tool calls first:** any tool calls from the last model response that have
  not run yet, which makes crash recovery exact.
- **Runs independent work in parallel:** when one response holds several read-only tool
  calls, or several `agent` calls, they run concurrently before the ordered loop records
  them (section 8).
- **Advances the plan:** in plan-and-execute mode the orchestrator may delegate ready
  agent steps, or give the model its next step, before calling it (section 18).
- **Applies steering:** messages queued with `agent-harness steer` are injected at this model
  turn boundary (`STEER_APPLIED`).

## 6. Model request

The request is checkpointed **before** the provider is contacted:

```text
MODEL_REQUESTED event
    ↓
pre_model hook (may deny, or add context for this request only)
    ↓
ContextBuilder builds the bounded view (+ summary, see below)
    ↓
overflow cap applied, if a previous request was rejected as too long
    ↓
token_counting = "server": the server counts the request; over the window → CONTEXT_CAPPED, rebuild
    ↓
client = self._models.resolve(run.model_profile)     # synchronous
    ↓
client.complete(request)  or  client.stream(request)
    ↓
Normalized ModelResponse
    ↓
MODEL_RESPONDED event (usage, cost, cached tokens, context stats,
                       provider_attempts / retried_errors when the adapter retried)
    ↓
post_model hook
```

The request contains the context view of the conversation, the tool specifications (in a
fixed order), and run metadata. Which tools are offered depends on the run:
- **Normally:** every enabled tool, minus any subagent tool restrictions.
- **`[tool_loading] mode = "dynamic"`:** only the `core` tools, `search_tools` /
  `load_tools`, and the tools this run has loaded. Loaded tools are appended in load
  order, so the list only grows at the end.
- **Plan-and-execute before a plan exists:** only read-only tools plus `create_plan`. The provider adapter converts it into the
provider-specific format; all supported servers use OpenAI-compatible `/chat/completions`.

- **Context window:** for profiles with `context_window`, the view is trimmed in large
  chunks (85% → 60% of the input budget) and keeps the system prompt, the first user
  message, and tool-call pairs. With `[summarization] enabled = true`, the omitted turns are
  replaced by a persisted summary (`CONTEXT_SUMMARIZED`) instead of being dropped. If even
  the minimum view does not fit, the run fails with `ContextWindowExceededError`.
- **Context overflow:** the view is built from an estimate, so a server can still reject
  it as too long (adapters recognise vLLM, SGLang, llama.cpp, and OpenAI messages, plus
  `[orchestrator] context_overflow_patterns`). The step is then retried, up to
  `context_overflow_retries` (default 2), with a smaller cap: the estimate times
  `context_overflow_shrink`, or the ratio the server stated less 10%. The cap is stored in
  the `MODEL_RESPONDED` event (`context_overflow: true`, `max_input_tokens`) and applies
  for the rest of the run on that profile. See `docs/diagrams/08-context-overflow.svg`.
- **Output cap:** the example profiles set `parameters.max_tokens = 1024`, so a runaway
  reply ends with `finish_reason = length` (Ollama native: `num_predict`).
- **Retries:** `max_model_attempts` (default 3) bounds provider requests per step and
  profile, including the adapter's own retries. A success after adapter retries records
  `provider_attempts`, `retried_errors`, and `attempt_timeout_seconds` in the event and
  telemetry.
- **Fallback:** when the retries run out, the next profile in the fallback chain is used
  (`MODEL_FALLBACK`).
- **Streaming:** with `--stream`, text is printed as it arrives, but only the final
  response is checkpointed.
- **Cost:** profiles with `pricing` get an estimated `cost_usd`. Without it, cost stays
  unknown.

Provider-specific code is isolated in `src/agent_harness/model_gateway`.

## 7. Model decision

The model response has two possible paths.

### Final answer path

If the assistant returns text without tool calls:

```text
Assistant response
    ↓
Steer queued meanwhile? → reopen the turn
    ↓
Plan-and-execute? → the answer completes the current step; the orchestrator
                    gives the next step (or asks for the final answer)
    ↓
stop hook (may ask to continue, at most 3 times)
    ↓
RUN_COMPLETED
    ↓
Return final answer
```

The final answer is printed to stdout.

### Tool-call path

If the model requests a tool:

```text
Assistant reply with tool calls
    ↓
More than max_tool_calls_per_turn (32)?  → refuse every call; ask again
    ↓
Repeated-call check (per call)
    ↓
pre_tool hook
    ↓
Permission rules and mode, then tool risk policy
    ↓
Execute, request approval, or refuse
```

The tool call is not executed blindly. A reply with more tool calls than
`[orchestrator] max_tool_calls_per_turn` is refused as a whole before any call runs: each
call gets a refusal result, and the run fails after `max_refused_repeats` such replies.

## 8. Tool decision and validation

**Parallel prefetch.** When one response holds several calls, eligible work starts
concurrently first:
- **Read-only calls:** they run together under the same idempotency keys, so the ordered
  loop below finds their results immediately.
- **`agent` calls:** their subagents are created one by one and then run together, with
  the parent's remaining tokens and cost split between them.

Calls that a rule denies or asks about are not prefetched, and nothing is prefetched when
a `pre_tool` hook is configured. Events and ordering are the same as without prefetch.

Each tool call then goes through these checks, in order:

1. **Duplicate in the same reply:** a read-only call identical to an earlier call in the same
   response is not run again. It gets that call's result (`duplicate_of`), and prefetch skips
   it too.
2. **Repeated-call check:** more than `max_identical_tool_calls` (5) identical calls since
   the last non-read-only call are refused. Each reply counts once, however many duplicates
   it holds. The run fails after `max_refused_repeats` (2) refusals.
3. **`pre_tool` hook:** it can deny the call or modify its arguments.
4. **Permission decision** (`extensions/permissions.py`); the first match wins:
   1. deny rules, which always win;
   2. permission mode: `plan` refuses non-read-only tools; `bypass` and `accept_edits`
      pre-approve;
   3. ask rules;
   4. allow rules;
   5. otherwise the tool's risk policy (by default, `high_risk` and `destructive` tools
      ask).
5. **Built-in tools:** the orchestrator handles these itself: `agent` (subagents),
   `exit_plan` (plan mode), `todo_write` (task list), `search_tools` / `load_tools`
   (dynamic loading, `TOOLS_LOADED`), and `create_plan` / `update_plan` /
   `revise_plan` (plan-and-execute). A call to a tool that exists but is not loaded
   tells the model to load it first.
6. **File snapshot:** before a tool that writes files, the touched files are snapshotted
   (`FILE_SNAPSHOT`) so the run can be rewound.
7. **ToolRuntime:** checks the tool name, argument schema and types, required fields,
   timeout, and idempotency key. The idempotency result is stored durably, so a replay
   after a crash reuses it.
8. **`post_tool` hook:** it can replace the result or add context.
9. **Record the result:**
   - Output from untrusted tools (web search, HTTP tools, MCP) is wrapped in
     `<untrusted-tool-output>` tags with a note that it is data.
   - Outputs larger than the artifact threshold are stored as artifacts and referenced
     from the event.

For coding tools, additional controls apply:

- Workspace boundary checks
- Protected file checks
- Command allowlists
- Dangerous command rejection
- Structured argv
- Docker sandbox execution

These guardrails apply in every permission mode, including `bypass`.

Relevant directories:

- `src/agent_harness/tools`
- `src/agent_harness/coding`
- `src/agent_harness/extensions/permissions.py`, `src/agent_harness/extensions/hooks.py`

## 9. Read-only tool execution

A read-only tool such as `read_file`, `search_files`, or `git_diff` can execute
immediately if it passes validation.

```text
TOOL_REQUESTED
    ↓
Tool policy check
    ↓
Tool handler
    ↓
Tool result
    ↓
TOOL_RESPONDED
```

The tool result is appended to the conversation as a tool message.

Then the model receives the updated conversation and decides what to do next.

```text
Tool result
    ↓
Next model request
```

## 10. Approval-required execution

A tool call pauses for approval when an ask rule matches or the risk policy requires it.
Plan approvals work the same way: the model calls `exit_plan`, and the run pauses with
`kind = "plan"`.

Example tools that usually need approval:

- `apply_patch`
- `run_tests`
- `run_command`
- `git_commit`
- `create_pull_request`

The handler is not executed before approval.

```text
Tool requires approval
    ↓
APPROVAL_REQUESTED event
    ↓
RunStatus.AWAITING_INPUT
    ↓
Persist checkpoint
    ↓
Process exits (code 3)
```

The CLI prints the run status and the pending request to stderr, for example:

```text
run_id=run-123 status=awaiting_input
approval_required={"arguments": {...}, "call_id": "call-456", "kind": "tool", "risk": "side_effect", "tool": "apply_patch"}
```

At this point, the run is safely paused in SQLite.

If a subagent needs approval, its parent is paused too. `agent-harness approve PARENT_ID`
shows and decides the subagent's pending call.

**Deadlines (optional).** With `[interaction] approval_timeout_seconds` (or
`question_timeout_seconds` for `ask_user`), a pause that waits longer is resolved by
`on_timeout`:
- `deny`: the call is denied (or the question declined) with `actor = "timeout"`, and the
  run continues;
- `cancel` or `fail`: the run ends.

Queue workers check at every reconcile and enqueue the runs that continue;
`agent-harness expire` does the same once (from cron, or without workers).

## 11. Resume after approval

Approve the action:

```bash
agent-harness approve \
  --config models.toml \
  --database agent-harness.sqlite3 \
  RUN_ID \
  --allow
```

Deny the action, optionally telling the model why:

```bash
agent-harness approve \
  --config models.toml \
  --database agent-harness.sqlite3 \
  RUN_ID \
  --deny --feedback "Use the staging database instead"
```

For a plan approval, `--mode default` or `--mode accept_edits` sets the permission mode
the run switches to.

The approval flow is:

```text
approve command
    ↓
pending_approval()
    ↓
resolve_approval(call_id=..., approved=..., feedback=..., mode=...)
    ↓
APPROVAL_RESOLVED event
    ↓
resume(run_id)
    ↓
Continue orchestration
```

If approved, the tool runs.

If denied, a policy-denied tool result (with any feedback) is sent back to the model,
allowing it to explain the denial or choose another approach. A rejected plan returns the
feedback so the model can revise the plan.

Approval does not bypass dangerous-command guardrails. Commands such as `rm`, `sudo`,
shell execution, unrestricted Docker access, and unsafe Git operations remain blocked.

## 12. Loop continuation

After a successful tool call:

```text
Tool result
    ↓
Append tool message
    ↓
Save checkpoint
    ↓
Call model again
```

The normal loop is:

```text
Step boundary (budgets, steering)
  ↓
Model
  ↓
Tool decision
  ↓
Validate
  ↓
Approve if needed
  ↓
Execute tool
  ↓
Persist result
  ↓
Model again
```

This continues until the model returns a final answer or the run reaches a stopping
condition.

## 13. Stopping conditions

Every run is bounded by its `RunBudget`:

| Budget             | Default | Notes                                                         |
|--------------------|---------|---------------------------------------------------------------|
| `max_steps`        | 20      | model turns                                                   |
| `max_wall_seconds` | 300     | idle time between `continue` turns is not counted             |
| `max_tokens`       | none    | includes summary calls and subagent usage                     |
| `max_cost_usd`     | none    | only works for profiles with `pricing`                        |

A run also stops on:

- **Retries and repeats:** model retries exhausted with no fallback left
  (`max_model_attempts`); tool retries exhausted (`max_tool_attempts`); repeated identical
  tool calls or oversized replies (`max_refused_repeats`); context-overflow retries used up
  (`context_overflow_retries`). All are set in `[orchestrator]`.
- **Quotas:** the tenant's or the end user's daily tokens or cost used up
  (`BUDGET_EXHAUSTED`).
- **Deadlines:** an approval or question that expired with `on_timeout = "cancel"` or
  `"fail"`.
- **Cancellation:** `agent-harness cancel`, the API, or a parent run being cancelled.
- **Permanent errors:** a permanent model or tool error, including
  `ContextWindowExceededError`.
- **Plan failure:** a plan step fails after the plan was already revised `max_replans`
  times.
- **Hooks:** a `session_start`, `user_prompt`, or `pre_model` hook that denies.

Possible persisted statuses are:

```text
QUEUED
RUNNING
AWAITING_INPUT
COMPLETED
FAILED
CANCELLED
BUDGET_EXHAUSTED
```

Because every event and checkpoint is stored, an interrupted run can be resumed without
losing its previous messages or tool results.

## 14. After a run: follow-ups

A run is not necessarily over when it stops:

| Command                           | What it does                                                         |
|-----------------------------------|----------------------------------------------------------------------|
| `continue RUN_ID "message"`       | Adds a user message to a completed run and runs it again (`USER_MESSAGE_ADDED`). With a larger budget it also reopens a `budget_exhausted` run. |
| `steer RUN_ID "message"`          | Queues a message for a running run, applied at its next model turn.  |
| `cancel RUN_ID`                   | Cancels the run and any unfinished subagents.                        |
| `mode RUN_ID plan`                | Changes the permission mode (`MODE_CHANGED`).                        |
| `resume RUN_ID`                   | Continues a run after a crash or restart.                            |
| `rewind RUN_ID --to-step N`       | Restores files and the conversation to before step N (dry run without `--yes`). History is archived, not deleted (`REWOUND`). |
| `status RUN_ID`                   | Shows tokens, cost, cache-hit rate, steps against budget, the task list, subagents, and the plan. |
| `tools list / sync / enable / disable` | Shows the persistent tool catalog, discovers MCP tools, and switches tools on or off across restarts. |
| `gc --older-than-days N [--yes]`  | Retention: deletes finished runs older than N days (dry run without `--yes`). |
| `expire [--no-resume]`            | Resolves approvals and questions past their `[interaction]` deadline, then continues denied runs. |
| `config show`                     | Effective settings and where each value came from, including `AGENT_HARNESS_SET__*` overrides. |

**Subagents.** When the model calls the `agent` tool, the orchestrator creates a child run
with its own prompt, tools, profile, and a budget capped by what the parent has left. The
parent waits (`SUBAGENT_WAITING`) and receives only the child's final answer
(`SUBAGENT_FINISHED`), and the child's usage rolls up into the parent.
- **Limits:** depth and count are limited by `max_depth` (default 2, so subagents may
  start their own) and `max_per_run`.
- **Parallel calls:** several `agent` calls in one response run concurrently.
- **Examples:** `examples/subagents.toml` has three working definitions.

## 15. Other entry points

- **HTTP API:** `agent-harness serve` exposes the same operations and runs in-process queue
  workers.
  - **Principals:** each API key (`[[api.keys]]`) is a principal with a tenant and a role.
  - **Roles:** `viewer` reads; `operator` also creates, continues, steers, cancels,
    resumes, and approves; `admin` also changes modes and rewinds.
  - **Tenant isolation:** runs in other tenants return 404.
  - **Audit:** approvals and mode changes record the acting key.
  - **Rate limit:** `[api] rate_limit_per_minute` per key (per address on failed
    authentication): over it, `429` with `Retry-After`.
  - **Health:** `GET /healthz` and `GET /readyz` need no key and are never rate limited.

  Endpoints:
  - Create a run with `POST /runs`, and read it with `GET /runs/{id}`.
  - Events: `GET /runs/{id}/events`, or live over SSE with `GET /runs/{id}/events/stream`.
  - Control a run with `POST /runs/{id}/cancel`, `.../resume`, `.../mode`, and `.../rewind`.
  - Messages: `POST /runs/{id}/messages` continues a run, and `POST /runs/{id}/steer`
    steers one that is still running.
  - Approvals: read with `GET /runs/{id}/approval`, decide with `POST /runs/{id}/approval`.
  - Usage: `GET /runs/{id}/usage`.
  - OpenAI-compatible (Phase 15d, `server/openai_chat.py`): `GET /v1/models` and
    `POST /v1/chat/completions`. `OpenAIChat.complete()` handles one request:

    ```text
    parse_chat_request()                 # messages, model → profile/route, stream
    _start_turn()
      ├── X-Agent-Harness-Run header, or find_conversation(digest of messages[:-1])
      ├── run awaiting input   → answer() / resolve_approval(decision_from_text(...))
      ├── completed/exhausted  → continue_run(budget + one turn)
      └── otherwise            → create_run(seeded with the client's history, enqueue=True)
    subscribe to StreamHub (stream) → enqueue → wait until run_is_settled()
    reply: newest answer, or pause_text(pending)
    save_conversation(digest of messages + reply, run_id)
    ```
- **Web chat:** `GET /ui` serves `src/agent_harness/server/ui/index.html` (no key needed for the page;
  every call it makes is authenticated). It drives the same endpoints: `POST /runs`,
  `GET /runs/{id}/stream` (read with `fetch`, so the bearer header is sent), `/messages`,
  `/approval`, `/answer`, `/usage`, `/events`, `/cancel`, `/steer`.
- **Queue workers:** `agent-harness worker` claims queued runs with leases, so several
  workers can share one database. On Postgres the claim uses `FOR UPDATE SKIP LOCKED` and
  workers can run on many hosts. An expired lease makes the run available to
  another worker, which resumes it. Each reconcile also requeues orphaned runs and expires
  overdue approvals and questions.
- **`agent-harness doctor --model-profile X`:** a live reachability and tool-call check
  against a real server.
- **`agent-harness eval`:** runs scenario evaluations, scripted by default and live with
  `--live`. It takes one or more files or directories.
  - **Bare scenarios** run on an orchestrator with the config's tools.
  - **Application-backed scenarios** (`config`, `[harness]`, `[workspace]`) go through
    `scenario_application()`. It calls `build_application(config_overrides=..., model_resolver=...)`
    in a temporary directory, so the run passes through the configured hooks, permission rules,
    subagents, memory, skills, and tool sources with the scripted model.
- **`scripts/validate.sh [feature ...]`:** runs `agent-harness-bench pipeline` for the named
  `[[feature]]` entries, or all of them (`docs/VALIDATION.md`).

## 16. Programmatic usage

You can use the harness directly from Python:

```python
import asyncio

from agent_harness.application import build_application
from agent_harness.domain import Message, Role, RunStatus


async def main() -> None:
    async with build_application(
        config_path="models.toml",
        database_path="agent-harness.sqlite3",
    ) as app:
        run = await app.orchestrator.create_run(
            model_profile="local",
            messages=[
                Message(
                    role=Role.USER,
                    content="Inspect the repository and summarize the architecture.",
                )
            ],
        )

        result = await app.orchestrator.run(run.id)

        if result.status is RunStatus.AWAITING_INPUT:
            approval = await app.orchestrator.pending_approval(run.id)
            print("Approval required:", approval)
            return

        print("Run status:", result.status)
        print("Messages:", result.messages)


asyncio.run(main())
```

For an approval-aware application:

```python
result = await app.orchestrator.run(run.id)

if result.status is RunStatus.AWAITING_INPUT:
    approval = await app.orchestrator.pending_approval(run.id)

    user_allowed = ask_user(approval)

    await app.orchestrator.resolve_approval(
        run.id,
        call_id=approval.call_id,
        approved=user_allowed,
        feedback=None,        # optional: returned to the model on denial
    )

    result = await app.orchestrator.resume(run.id)
```

For a follow-up turn on a completed run:

```python
await app.orchestrator.continue_run(run.id, "Now add tests for it.")
result = await app.orchestrator.run(run.id)
```

## 17. Process exit codes

The CLI distinguishes run results from process errors:

| Condition                                           | Exit code |
|-----------------------------------------------------|-----------|
| Run completed successfully                          | 0         |
| Run failed, cancelled, or budget exhausted          | 1         |
| Invalid configuration or command-line input, or a quota refusal | 2 |
| Approval is required (the run is `awaiting_input`)  | 3         |

The final assistant response is written to stdout. Operational information such as
`run_id`, status, errors, and `approval_required=...` is written to stderr. `doctor` and
`eval` exit with 1 when a check or scenario fails.

## 18. Plan-and-execute

Turn it on with `run --plan`, `"planning": true` on the API, or `[planning] enabled = true`.
The model proposes the plan, and the orchestrator owns and drives it. Diagrams:
`docs/diagrams/06-plan-and-execute`.

```text
Explore with read-only tools
    ↓
create_plan(goal, steps with ids, depends_on, optional agent)
    ↓ validated: unique ids, known dependencies, no cycles, known agents
PLAN_CREATED (the plan is stored on the run)
    ↓
Orchestrator loop over ready steps (dependencies completed)
    ├── agent steps → child runs started by the orchestrator, several at once,
    │                 results collected without a model turn
    ├── other steps → "[Plan] Execute step …" for the main model;
    │                 update_plan or a final answer completes the step
    └── a failed step → "[Plan] … failed" → revise_plan (at most max_replans)
    ↓
All steps finished → "[Plan] … write the final answer" → final answer → completed
```

- **One message per boundary:** the orchestrator adds at most one user message per
  boundary. Delegated results wait in `plan.outbox` until the next message, so a pause or
  crash loses nothing.
- **Children:** plan children have deterministic ids. A resume finds them instead of
  starting new ones, and an approval inside a child pauses the parent.
- **Events:** `plan_created`, `plan_step_updated`, and `plan_revised`. `status` and
  `GET /runs/{id}` show the plan.

## 19. Tool sources, the catalog, and MCP

```text
[[tools]] python / http    [web_search] [web_fetch]    [[mcp_servers]] stdio / http
          └──────────────┬───────┴───────────────┘
                   ToolRuntime registry (handlers are code)
                         ↓
         tool_catalog table: source, spec hash, risk, trust,
         enabled (operator switch), available (still provided)
                         ↓
         offered to the model (all, or core + loaded in dynamic mode)
```

- **Persistence:** the catalog survives restarts. Every process re-attaches handlers from
  the configured sources and then syncs the catalog.
- **Switches:** `tools disable NAME` persists.
- **Removed tools:** a tool its source stopped providing is marked unavailable, not
  deleted.
- **Sandboxed Python tools** (`sandbox = "container"`): tool definitions are discovered in a
  one-shot container. Calls run in the run's **warm container** (`docker exec`), with no
  network by default and a read-only mount of the module. The container is removed when the
  run ends.
- **MCP servers:**
  - Tools are named `server__tool`; their schemas are normalized to the validator's
    subset.
  - They ask for approval unless listed in `read_only`, and their output is untrusted.
  - A stdio server sees only the environment variables named in `env` (plus
    `PATH`/`HOME`).
  - Servers start on first use and are synced before runs execute when `sync_on_start`
    is set.
  - With `resources = true` / `prompts = true`, a server's resources and prompts become
    read-only `server__mcp_*` tools. A `list_changed` notification re-syncs the server in
    the background.
- **`web_fetch`:** one page as text. Private and loopback addresses are refused unless
  `allow_private`; each redirect hop is re-checked against that rule and `[network]`.

## 20. Production controls

| Control | Where it acts |
|---|---|
| Quotas (`[[quotas]]`, `[user_quotas]`) | Tenant and end-user limits in one atomic reservation at create and continue (API 429, CLI exit 2); re-checked at every step |
| API rate limit (`[api] rate_limit_*`) | Every API request except health checks; 429 with `Retry-After` |
| Input deadlines (`[interaction] *_timeout_seconds`) | Worker reconcile or `agent-harness expire`; `on_timeout` deny / cancel / fail |
| Loop limits (`[orchestrator]`) | Provider attempts, repeated calls, tool calls per reply, context-overflow retries |
| Environment overrides (`AGENT_HARNESS_SET__SECTION__KEY`) | At config load; shown by `config show` |
| Memory backend (`[memory] backend`) | SQLite FTS5 or Postgres full-text search; `memory migrate` copies notes |
| Warm containers (`mode = "per_run"`, sandboxed tools) | Started on a run's first container call; removed when the run ends |
| API principals (`[[api.keys]]`) | Every API request: role check, then tenant check on the run |
| Secret references (`env:`, `file:`) | Whenever a model or tool credential is read |
| Egress allowlist (`[network]`) | Before HTTP tools, web search, or MCP over HTTP send a request |
| Redaction (`[redaction]`) | As events, run errors, and telemetry are written |
| Migrations | When a store is opened (a newer schema is refused) |
| Retention (`gc`) | On demand. Events stay append-only except during a purge |

See `docs/DEPLOYMENT.md` for the container, compose file, and upgrade procedure.

## Complete flow

```text
User command (CLI, API, or worker)
    ↓
Build application
    ↓
Load layered settings, plugins, templates, skills
    ↓
Create model gateway, tools + tool sources + MCP, store (migrate), tool catalog
    ↓
Rate limit (API) → quota reservation (tenant + end user) → create durable run
(system context, planner instructions, user_prompt hook, version hashes, RUN_CREATED)
    ↓
Start orchestrator (RUN_STARTED, session_start hook)
    ↓
Step boundary: budgets and quotas, pending tools (parallel prefetch), plan step or
delegation, steering
    ↓
pre_model hook → context view (+ summary, + overflow cap) → offered tools (all / core + loaded)
→ call model (max_tokens cap, retries recorded, fallback, cost;
  "too long" → smaller cap and retry)
    ↓
Model returns text or tool call
    ├── Text
    │     ↓
    │   plan-and-execute: complete the step → next step or final answer
    │     ↓
    │   stop hook (may continue)
    │     ↓
    │   Complete run
    │     ↓
    │   Print final answer
    │
    └── Tool call
          ↓
        Too many calls in the reply? → refuse all, ask again
          ↓
        Repeated-call check → pre_tool hook
          ↓
        Permission rules and mode → risk policy → guardrails
          ├── Denied
          │     ↓
          │   Return policy error to model
          │
          ├── Approval required (or plan review)
          │     ↓
          │   Persist paused run
          │     ↓
          │   Exit with code 3 → approve → resume
          │   (or the deadline expires → deny / cancel / fail)
          │
          └── Allowed
                ↓
              Snapshot files → execute tool (validation, timeout, idempotency)
                ↓
              post_tool hook → persist result (untrusted output wrapped,
              artifact if large, secrets masked in events)
                ↓
              Next step
```
