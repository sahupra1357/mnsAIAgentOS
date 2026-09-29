# Agent Harness

A small, durable agent runtime that can point at different model endpoints without changing the orchestration loop.

It provides:

- Named, configurable model profiles.
- An OpenAI-compatible adapter for hosted and local model servers.
- A resumable model/tool state machine.
- In-memory and SQLite run stores.
- Typed tools with validation, policy, timeouts, and idempotency.
- Explicit budgets for steps, wall time, tokens, and cost.
- Append-only run events and optimistic checkpointing.
- Network-free tests with scripted models.

## Architecture

```text
CLI / application
       │
       ▼
AgentOrchestrator ─────► RunStore
       │                (memory / SQLite)
       ├───────────────► ModelGateway
       │                (named provider adapters)
       └───────────────► ToolRuntime
                        (registry + policy + limits)
```

The orchestrator uses protocols from `domain.py`; it does not know which model provider, database, or tool implementation is behind them.

## Quick Start

Requires Python 3.12 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
cp examples/models.toml models.toml
```

Edit `models.toml` so the endpoint and model name match your server. The default `local` profile does not require a credential.

List the configured profiles:

```bash
.venv/bin/agent-harness profiles --config models.toml
```

Run an agent:

```bash
.venv/bin/agent-harness run \
  --config models.toml \
  --model-profile local \
  "Use the calculator to multiply 27 by 19."
```

The run ID and status are written to stderr. The final assistant response is written to stdout. State is saved by default in `agent-harness.sqlite3`.

Resume an interrupted run:

```bash
.venv/bin/agent-harness resume \
  --config models.toml \
  --database agent-harness.sqlite3 \
  RUN_ID
```

## Pointing at Different Models

Define one TOML table per model:

```toml
default_model_profile = "local"

[models.local]
adapter = "openai_compatible"
model = "local-model"
endpoint = "http://127.0.0.1:8000/v1"

[models.production]
adapter = "openai_compatible"
model = "production-model"
endpoint = "https://provider.example/v1"
api_key_env = "PRODUCTION_MODEL_API_KEY"
timeout_seconds = 60
max_retries = 2
retry_backoff_seconds = 0.25

[models.production.parameters]
temperature = 0.2
```

Then choose a profile per run:

```bash
export PRODUCTION_MODEL_API_KEY="..."
.venv/bin/agent-harness run \
  --config models.toml \
  --model-profile production \
  "Summarize the deployment plan."
```

The endpoint can be a base ending in `/v1` or a full `/chat/completions` URL. Credentials are resolved at request time from the environment variable named by `api_key_env`; secrets do not belong in TOML.

Configuration can also be overridden through environment variables:

```text
AGENT_HARNESS_DEFAULT_MODEL_PROFILE
AGENT_HARNESS_MODEL__<PROFILE>__ADAPTER
AGENT_HARNESS_MODEL__<PROFILE>__MODEL
AGENT_HARNESS_MODEL__<PROFILE>__ENDPOINT
AGENT_HARNESS_MODEL__<PROFILE>__API_KEY_ENV
AGENT_HARNESS_MODEL__<PROFILE>__TIMEOUT_SECONDS
AGENT_HARNESS_MODEL__<PROFILE>__MAX_RETRIES
AGENT_HARNESS_MODEL__<PROFILE>__RETRY_BACKOFF_SECONDS
AGENT_HARNESS_MODEL__<PROFILE>__PARAMETERS
```

`PARAMETERS` is a JSON object. Environment profile names are normalized to lowercase.

## Ollama, llama.cpp, and Modal

Built-in adapters: `openai_compatible`, `ollama`, `llama_cpp`, and `modal`.
`examples/models.toml` has a working profile for each.

| Adapter     | Server                                  | Notes                                                                                     |
|-------------|-----------------------------------------|-------------------------------------------------------------------------------------------|
| `ollama`    | `ollama serve`                          | Handles tool calls without IDs. `api_mode = "native"` uses `/api/chat` and sends `context_window` as `num_ctx`. |
| `llama_cpp` | `llama-server -m model.gguf --jinja -c N` | `--jinja` is required for tool calls. 503 while loading is retried.                        |
| `modal`     | vLLM or SGLang on Modal (`server = "vllm"` or `"sglang"`) | Requires both the server bearer key (`api_key_env`) and Modal proxy auth (`headers_env` for `Modal-Key` and `Modal-Secret`). The first request uses `cold_start_timeout_seconds` (default 600). |

Deploy scripts for Modal are in `examples/modal/`. They keep prefix caching on, turn
on cached-token reporting, and set container options that keep a run on one warm KV
cache.

Extra profile fields (all optional):

```toml
headers_env = { X-Header = "ENV_VAR_NAME" }  # values read from the environment per request
context_window = 32768
cold_start_timeout_seconds = 600
server = "vllm"            # modal: vllm | sglang
api_mode = "openai"        # ollama: openai | native
pricing = { gpu_usd_per_second = 0.0 }       # validated now, used by cost estimation later
fallback = ["llamacpp"]    # validated now, used by fallback routing later
```

Check a profile against a live server:

```bash
.venv/bin/agent-harness doctor --config models.toml --model-profile modal
```

`doctor` checks, in order:
1. The credential variables are set.
2. The server's health endpoint responds.
3. The configured model is listed.
4. A plain completion works.
5. A structured tool call works. A tool call returned as plain text means the server's
   tool-call parser is missing.
6. A repeated prompt is served from the prefix cache.

It exits 1 if any check fails.

## Retries, Cost, Fallback, and Routes

- **One retry budget:** `OrchestratorSettings.max_model_attempts` (default 3) counts
  every provider request in a step, including the adapter's own `max_retries`, so
  retries never multiply. Only retryable errors (timeouts, throttling, 5xx) are retried.
- **Fallback:** when retryable failures exhaust that budget, the run switches to the
  next profile in the original profile's `fallback` list. It records a
  `model_fallback` event and stays on that profile. Permanent errors (auth,
  configuration, malformed output) never fall back.

  ```toml
  [models.modal]
  fallback = ["llamacpp", "ollama"]
  ```

- **Cost:** add `pricing` to a profile, and each response gets an estimated
  `cost_usd`, so `--max-cost-usd` works.

  ```toml
  [models.modal.pricing]
  gpu_usd_per_second = 0.0003        # your GPU's rate; latency includes cold start
  # input_usd_per_million_tokens = 0.2
  # output_usd_per_million_tokens = 0.6
  ```

- **Repeated calls:** after 5 identical calls to the same tool (since the last
  side-effecting call), further identical calls are refused with an explanation to the
  model. After 2 refusals the run fails. Tune this with
  `OrchestratorSettings(max_identical_tool_calls=..., max_refused_repeats=...)`, or
  disable it with `None`.
- **Routes:** name capabilities instead of profiles. A route can be used anywhere a
  profile name is accepted, including `default_model_profile` and `--model-profile`.
  The run stores the resolved profile and records the route. Environment override:
  `AGENT_HARNESS_ROUTE__FAST=ollama`.

  ```toml
  [routes]
  fast = "llamacpp"
  strong = "modal"
  ```

## Conversations, Steering, and Status

```bash
agent-harness run      --config models.toml "Summarize the incident report"   # prints run_id
agent-harness continue --config models.toml RUN_ID "Now draft the customer email"
agent-harness steer    --config models.toml RUN_ID "Focus on the billing impact"
agent-harness cancel   --config models.toml RUN_ID
agent-harness status   --config models.toml RUN_ID
```

- **`continue`** reopens a completed run with a follow-up message. The model sees the
  whole conversation. Budgets are cumulative, so pass `--max-steps` and similar flags
  to raise them; that is also how a `budget_exhausted` run is continued. Idle time
  between turns does not count against the wall-clock budget.
- **`steer`** queues a message for a run that is still working, for example one being
  executed by a worker or waiting for approval. It is added at the next model turn,
  after any pending tool results, and exactly once, even across resumes. A steer that
  arrives while the model is writing its final answer reopens the turn.
- **`cancel`** works from any process. A worker executing the run stops at its next
  step boundary instead of crashing.
- **`status`** shows steps, tokens (including cached), cost, and active time against
  the budget.

The same actions are available over HTTP: `POST /runs/{id}/messages`,
`POST /runs/{id}/steer`, `POST /runs/{id}/cancel`, and `GET /runs/{id}/usage`.

## Permissions, Modes, Plan Mode, and Hooks

```toml
[permissions]
mode = "default"                          # default | accept_edits | plan | bypass
allow = ["read_file", "run_command(argv.0=pytest)"]
ask   = ["read_file(path=*.env)"]
deny  = ["run_command(argv.0=rm)", "git_*(force=true)"]
edit_tools = ["apply_patch"]              # what accept_edits auto-allows

[[hooks]]
event = "pre_tool"                        # session_start | user_prompt | pre_model | post_model
matcher = "run_command"                   #   | pre_tool | post_tool | stop
command = ["python3", "hooks/check_command.py"]   # argv, never a shell
timeout_seconds = 10
# on_error = "deny"                       # guarding events fail closed by default
```

**Order of evaluation.** Deny rules win. Then the mode applies:
- `plan`: read-only tools only;
- `bypass`: everything allowed;
- `accept_edits`: the edit tools are allowed.

After that, `ask` rules pause for approval and `allow` rules skip the approval pause.
Anything left unmatched falls back to each tool's risk policy. "Allow" never skips
argument validation, workspace boundaries, or command guardrails.

**Plan mode.** Run with `agent-harness run --mode plan ...`. The model investigates
with read-only tools and then calls `exit_plan`, which pauses the run for approval:
- `agent-harness approve RUN_ID --allow --mode accept_edits` approves the plan and
  switches mode;
- `--deny --feedback "..."` sends the feedback to the model to revise the plan.

`agent-harness mode RUN_ID bypass` changes a run's mode at any time.

**Hooks.** A hook receives JSON on stdin and may print JSON on stdout:
`{"decision": "allow" | "deny" | "modify" | "continue", "reason", "arguments",
"content", "context"}`.
- `pre_tool` can deny a call, or rewrite its arguments; permission rules are then
  checked against the rewritten call.
- `user_prompt` can reject or rewrite a prompt.
- Other events can add context.
- `stop` can ask the model to keep working, a bounded number of times.

Every decision is recorded as a `hook_decided` event. A resumed run replays the
recorded decisions; it does not re-run the hooks.

## Subagents and Task Lists

```toml
[tasks]
enabled = true                    # offer todo_write: the model keeps a visible task list

[subagents]
max_depth = 2                     # how deep delegation may nest
max_per_run = 8

[[agents]]
name = "researcher"
description = "Finds facts in the repository with read-only tools."
prompt_file = "agents/researcher.md"      # or prompt = "..."
tools = ["read_file", "search_files"]     # omit for all tools
profile = "fast"                          # optional: a cheaper model or route
max_steps = 10
```

**Delegation.** When agents are configured, the model gets an `agent` tool and can
hand a self-contained task to a subagent:
- The subagent is a separate run with its own prompt, tool subset, model, and budget.
  Its budget is capped by what the parent has left.
- Only its final answer returns to the parent, and its tokens and cost count toward
  the parent's budget.
- Subagents inherit the parent's permission mode, so a plan-mode parent gets
  read-only subagents.
- If a subagent needs approval, the parent pauses too. `agent-harness approve
  PARENT_ID` shows and decides the subagent's pending call.
- Cancelling the parent cancels its subagents. Resuming the parent resumes an
  unfinished subagent instead of starting a new one.

**Task list.** `agent-harness status RUN_ID` and `GET /runs/{id}` show the task list
and the subagents.

## Instructions, Skills, Templates, Settings Layers, and Plugins

```toml
[instructions]                    # project conventions for every new run
enabled = true
files = ["~/.agent-harness/AGENTS.md", "AGENTS.md"]   # later = more specific

[skills]                          # skills/<name>/SKILL.md with name/description frontmatter
enabled = true
paths = ["skills"]

[templates]                       # templates/<name>.md, used as: /name args
paths = ["templates"]

[plugins]
enabled = ["reviewkit"]           # installed plugins take effect only when listed
```

**Instructions and skills.** Both are read once when a run starts and fixed for that
run, so the prompt prefix stays cacheable, and their hash is recorded. The model sees
only each skill's name and description, and loads a full skill with `load_skill`.

**Templates.** `agent-harness run --arg focus=security "/review app.py"` fills
`$ARGUMENTS`, `$1`, and `${focus}`, and records the template in the run. List them
with `agent-harness templates`.

**Settings layers.** Settings merge in this order:
1. user: `~/.agent-harness/settings.toml`
2. project: your `--config` file
3. local: `models.local.toml`, for personal overrides that are not committed
4. enterprise: `/etc/agent-harness/settings.toml`, applied last so it wins

Rule lists (permissions, hooks, agents) accumulate across layers. Run
`agent-harness config show --config models.toml` to see each effective value and
the layer it came from.

**Plugins.** A plugin is a directory with a `plugin.toml` manifest that bundles
templates, skills, subagents, and hooks.
- `agent-harness plugin install ./my-plugin --config models.toml` validates it and
  copies it into `.agent-harness/plugins/`. It does nothing until you add its name to
  `[plugins] enabled`.
- Plugin hooks and agents follow the same permission rules and guardrails as
  everything else.

## LiteLLM (optional)

LiteLLM is opt-in for each profile. Only profiles with `adapter = "litellm"` use it;
every other profile keeps calling its server directly, and without such a profile the
package is never imported.

```bash
pip install -e '.[litellm]'
```

```toml
[models.claude]
adapter = "litellm"
model = "anthropic/claude-sonnet-5"     # LiteLLM provider/model name
api_key_env = "ANTHROPIC_API_KEY"       # passed as api_key
# endpoint = "https://gateway/v1"       # optional, passed as api_base
# headers_env = { X-Team = "TEAM_ID" }  # passed as extra_headers

[models.modal]
fallback = ["claude"]                   # use it only when Modal keeps failing
```

**How it behaves inside the harness:**
- **Retries:** LiteLLM's own retries are turned off. The profile's `max_retries`, the
  harness retry budget, and fallback apply as with every other adapter.
- **Errors:** LiteLLM errors are mapped to the harness error types (auth, throttling,
  timeout, context window, provider).
- **Cost:** comes from LiteLLM's per-call cost, unless the profile sets `pricing`.
- **Everything else works as usual:** context trimming, cached tokens, streaming,
  telemetry, routes, and `doctor`.

Alternatively, run the LiteLLM Proxy yourself and point an ordinary
`openai_compatible` profile at it.

## Context Window

Profiles that set `context_window` get a bounded view of the history on every model call.
The stored run history is never modified.

- **Always kept:** system messages and the first user message.
- **Tool call pairs:** an assistant tool call always stays together with its tool results.
- **Large tool outputs:** shortened in the view to head + tail (default 16,000 characters).
- **Trimming:** when the estimate reaches 85% of the input budget, the oldest turns are
  omitted until it falls to 60%, and a short notice records the omission. Trimming in
  chunks keeps the prompt prefix stable between trims, so vLLM, SGLang, and llama.cpp
  keep reusing their prefix cache.
- **Input budget:** `context_window` minus the output reserve (`parameters.max_tokens`,
  default 1024) and the tool definitions.
- **Too large to fit:** if the system prompt, first user message, and latest turn alone
  do not fit, the run fails with `ContextWindowExceededError`. It does not send a
  truncated request.

Tune the behavior with `build_application(..., context_policy=ContextPolicy(...))`.

## Programmatic Composition

The default application uses SQLite, the configured model gateway, and the example tools:

```python
from agent_harness.application import build_application
from agent_harness.domain import Message, Role

async with build_application(
    config_path="models.toml",
    database_path="runs.sqlite3",
) as app:
    run = await app.orchestrator.create_run(
        model_profile="local",
        messages=[Message(role=Role.USER, content="Calculate 12 * 8")],
    )
    result = await app.orchestrator.run(run.id)
```

For tests or custom deployments, compose `AgentOrchestrator` directly with any implementations of `ModelResolver`, `ToolExecutor`, and `RunStore`.

## Adding a Model Adapter

Implement the `ModelClient` protocol and register a factory:

```python
gateway = ModelGateway(
    config,
    adapter_factories={"my_provider": lambda profile: MyModelClient(profile)},
)
```

Provider-specific payloads remain inside the adapter; return the normalized `ModelResponse` domain type.

## Coding-Agent Mode

Set the coding section in `models.toml`:

```toml
[coding]
enabled = true
workspace = "."
execution_runtime = "docker"
approval_mode = "side_effects"
enable_git_commit = true
enable_pull_requests = false

[coding.docker]
image = "python:3.12-slim"
network_enabled = false
memory = "1g"
cpus = 1.0
pids_limit = 256
```

Then run a coding request:

```bash
.venv/bin/agent-harness run \
  --config models.toml \
  "Inspect the repository, fix the failing tests, and summarize the diff."
```

The agent can read and search files, inspect Git status/diffs, apply exact text patches, run allowlisted commands/tests, create commits, and optionally create pull requests through `gh`. Read-only tools run automatically. Writes and execution pause the durable run and print an approval request:

```text
approval_required={"call_id":"...","tool":"apply_patch","risk":"side_effect",...}
```

Approve or deny it explicitly:

```bash
.venv/bin/agent-harness approve \
  --config models.toml \
  --database agent-harness.sqlite3 \
  RUN_ID --allow
```

Use `--deny` to return the denial to the model. Approval is never a bypass for dangerous-command guardrails. The default Docker runtime uses no network, a workspace-only bind mount, dropped capabilities, `no-new-privileges`, CPU/memory/PID limits, a read-only root filesystem, and a bounded `/tmp`. Local execution requires `allow_local_execution = true` and is intended only for trusted development.

The coding tools and policy details are in [`skills/coding-agent/SKILL.md`](skills/coding-agent/SKILL.md). Pull requests are disabled by default; enabling them requires adding `gh` to `coding.allowed_commands`, setting `coding.enable_pull_requests = true`, explicitly enabling Docker networking, and providing GitHub credentials outside the repository.

## Adding Tools

Tool registration is explicit:

```python
registry = ToolRegistry()
registry.register(tool_spec, async_handler)
runtime = ToolRuntime(registry)
```

Read-only tools may run automatically. Side-effecting tools require an idempotency key. High-risk and destructive tools require an approval-aware application policy; the default orchestrator does not grant approval.

The included `echo` and `calculator` tools are deterministic examples. In production, isolate untrusted tools out of process and back side-effect idempotency with durable storage or the downstream service.

## HTTP API and Workers

Runs can be submitted over HTTP and executed by queue workers. Many workers,
in-process or in separate processes, can share one SQLite database. Each claimed run
is leased: a worker renews the lease while it works, and if the worker dies, the lease
expires and another worker resumes the run from its last checkpoint.

```bash
pip install -e '.[api]'                        # uvicorn
export AGENT_HARNESS_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/agent-harness serve --config models.toml --workers 2        # API + in-process workers
.venv/bin/agent-harness worker --config models.toml --workers 4       # optional extra worker processes
```

| Method and path | Purpose |
|---|---|
| `GET /healthz` | Liveness (no auth) |
| `GET /profiles` | Profiles (adapter, model, context window, fallback) and routes |
| `POST /runs` | `{"prompt", "system"?, "model_profile"?, "budget"?}` → `202 {"id", ...}` (queued) |
| `GET /runs/{id}` | Status, usage, error, final `output`, and `pending_approval` |
| `GET /runs/{id}/events?after=N` | Event log from index N |
| `GET /runs/{id}/events/stream?after=N` | Server-sent events until the run completes or pauses |
| `POST /runs/{id}/cancel` | Cancel |
| `POST /runs/{id}/resume` | Re-enqueue an interrupted run |
| `GET /runs/{id}/approval` | The pending tool call, with its arguments |
| `POST /runs/{id}/approval` | `{"call_id", "decision": "allow" \| "deny"}`, then re-enqueue |

All routes except `/healthz` require `Authorization: Bearer $AGENT_HARNESS_API_TOKEN`.
`--insecure-no-auth` is accepted only with a loopback `--host`. Inputs are validated at
the boundary: unknown fields, oversized text, and out-of-range budgets are rejected. A
run that keeps raising unexpected errors is removed from the queue after 3 attempts;
it can be resumed explicitly after investigation.

## Artifacts, Memory, and Evaluations

Enable either feature in `models.toml`. Both are off by default:

```toml
[artifacts]
enabled = true
directory = "artifacts"      # relative to the config file
threshold_bytes = 16384      # larger tool results are stored as artifacts

[memory]
enabled = true
namespace = "my-project"     # isolates notes between projects
# database = "memory.sqlite3"  # defaults to the run database
```

**Artifacts.** A tool result larger than the threshold is stored content-addressed.
The model sees a preview plus the artifact ID, and can page through the full text
with the `read_artifact` tool. Events record `artifact_id`, and the API serves the
content at `GET /artifacts/{id}`.

**Memory.** Adds the `memory_search` (read-only) and `memory_write` (side-effect)
tools, backed by SQLite full-text search. Search results are labeled as notes, not
instructions, because anything a model saved earlier is untrusted input.

**Evaluations.** Scenario files (TOML) run the real orchestrator, with the same tools
and context policy, against scripted responses (offline, suitable for CI) or live
models:

```bash
.venv/bin/agent-harness eval --config models.toml examples/evals --report eval.json
.venv/bin/agent-harness eval --config models.toml examples/evals --live --model-profile ollama
```

Scenarios can assert:
- the final status;
- the tool calls made, in order;
- text the output must or must not contain;
- the maximum number of steps;
- required events;
- expected error text.

They can also auto-approve or auto-deny approval pauses. See `src/agent_harness/evals.py`
for the file format and `examples/evals/` for samples. The command exits 1 if any
scenario fails.

## Telemetry and Streaming

Stream the answer as it is generated, and write sanitized telemetry:

```bash
.venv/bin/agent-harness run --config models.toml --stream \
  --telemetry-file telemetry.jsonl "Explain the failing test"
```

**What gets recorded.** Each JSON line is one of:
- a persisted run `event`;
- a `model_call` (profile, attempt, ok, tokens, cached tokens, cost, latency,
  error category);
- a `tool_call` (tool, status, risk, attempt, latency).

The run ID is the trace ID. Records never contain prompts, message content, tool
arguments or outputs, or credentials.

**Programmatic use.** Pass a sink to `build_application(telemetry=...)` or
`AgentOrchestrator(telemetry=...)`:

| Sink                                                                | What it does                                              |
|---------------------------------------------------------------------|-----------------------------------------------------------|
| `JsonLinesTelemetry(stream)`                                        | Writes one JSON line per record.                          |
| `MetricsTelemetry()`                                                | In-process counters, `cache_hit_rate(profile=...)`, `snapshot()`. |
| `MultiTelemetry([...])`                                             | Sends records to several sinks.                           |
| `OpenTelemetryTelemetry()` (`pip install -e '.[otel]'`; configure the SDK/exporter as usual) | Spans plus duration, token, and cost metrics. |

A failing sink never affects a run.

**Streaming details.** Streaming uses server-sent events on any OpenAI-compatible
server. Pass a `StreamListener` to `build_application(stream_listener=...)` to receive
text increments. Streamed text is a preview: only the final assembled response is
checkpointed, and a retried call may show text again. Ollama's `api_mode = "native"`
does not stream.

## Budgets and Durability

Each run can limit:

- Model steps
- Wall-clock duration
- Consumed tokens
- Estimated cost

Run state and boundary events are checkpointed after accepted model and tool results. SQLite checkpoints update the run snapshot and append events atomically, using a version field to reject concurrent stale writes.

**Durable tool results.** Successful tool results are stored in the run database
(`tool_results` table), keyed by tool name and idempotency key
(`run_id:step:call_id`). If a worker crashes after a tool ran but before its result was
checkpointed, the resumed run gets the stored result instead of running the tool again.
Reusing a key with different arguments is rejected. A crash *during* a tool's execution
can still repeat it, so external services should also honor the idempotency key.

**Versions.** Every run records, in its `run_created` event:
- `system_prompt_sha256`
- `tools_sha256`
- `profile_sha256` (the profile's non-secret settings)
- `harness_version`

If a run is resumed with different tool definitions, a `config_drift` event records the
old and new hashes.

**Cached tokens.** `Usage.cached_input_tokens` reports prompt tokens served from the
server's prefix cache (when the server reports it), and is included in `model_responded`
events.

## Validation

See [`docs/VALIDATION.md`](docs/VALIDATION.md). CI runs these stages offline on every
PR: lint, types, unit tests, and scripted scenarios. With your model servers and
downloaded public datasets you can also run:
- live `doctor` smoke checks;
- benchmarks: GSM8K (agent loop and tools), BFCL (tool-calling accuracy per provider),
  HumanEval (coding in the Docker sandbox), and generic Q&A;
- regression gates against saved baselines.

```bash
agent-harness-bench suites
agent-harness-bench run --suite gsm8k --data datasets/gsm8k/test.jsonl --config models.toml --model-profile ollama --limit 200
agent-harness-bench pipeline examples/validation.toml --update-baselines
```

## Development

Run the complete suite without network access:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
```

Multi-agent development instructions are in [`AGENTS.md`](AGENTS.md). Component-specific reusable skills live under [`skills/`](skills/):

- `durable-agent-orchestration`
- `model-gateway`
- `tool-runtime`
- `harness-verification`

Before changing a component, agents should read the corresponding `SKILL.md` and respect the ownership and boundary rules in `AGENTS.md`.

## Current Scope

This first version is a solid local and single-service foundation. Horizontal worker scaling would add a queue/lease layer around persisted run IDs. The coding mode includes a Docker sandbox, while distributed idempotency, telemetry exporters, and a production approval UI/notification channel remain deliberate extension points.
