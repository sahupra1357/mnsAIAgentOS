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
