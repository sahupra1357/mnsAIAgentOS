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
reasoning_tags = false     # split <think>...</think> out of the answer
reasoning_history = "drop" # or "keep": send earlier reasoning back to the model
```

### Reasoning ("thinking") models

When a server returns a thinking model's reasoning separately, the harness keeps it on
the assistant message (`Message.reasoning`) instead of discarding it. It is read from
`reasoning_content` (vLLM, SGLang with `--reasoning-parser`, llama.cpp with
`--reasoning-format deepseek`, LiteLLM), `reasoning` (Ollama's OpenAI mode, newer
vLLM), or `thinking` (Ollama `api_mode = "native"` with `parameters.think = true`).
Field names vary between server versions, so check yours with one request.

- **Server without a reasoning parser:** `reasoning_tags = true` splits a leading
  `<think>...</think>` block (or reasoning ending in `</think>`) out of the answer.
  Streamed text is shown before the split, so the tags can appear in `--stream` output.
- **History:** `reasoning_history = "drop"` (default) never sends earlier reasoning
  back. Most chat templates strip it anyway, and the prompt prefix stays cacheable.
  Use `"keep"` for models that expect their reasoning back (sent as
  `reasoning_content`, or `thinking` for native Ollama); it is then counted against
  `context_window`.
- **Records:** the text is saved in the run snapshot. The `model_responded` event holds
  only `reasoning_chars`.
- **CLI:** `run`/`continue --show-reasoning` prints it to stderr, as it streams with
  `--stream`. Custom stream listeners can implement `on_reasoning(run_id, step, text)`.

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
- **`--strict`:** exit 1 on warnings too (use it in CI).
- **`--max-tokens N`:** raises the probes' output limits. Thinking models need about
  512, and doctor reports "ran out of tokens" when they hit the default limit.
- **llama.cpp routers** (`llama serve`, several models) get a real model check; a
  single-model `llama-server` is skipped.

### Gateway options per profile

```toml
max_concurrency = 2            # concurrent requests to this server from one process (default unlimited)
retry_jitter = 0.2             # spread retry delays by +/-20% (default 0)
max_retry_after_seconds = 60   # longest server Retry-After to honour on 429/503
structured_output = "strict"   # "strict" | "non_strict" | "prompt" (for servers that reject response_format)
```

- **Retry delays:** exponential backoff, but at least the server's `Retry-After`
  (seconds or an HTTP date, capped).
- **Hash-stable fields:** these fields are left out of the profile hash at their
  defaults, so older runs do not report config drift.

### Images (vision models)

```bash
agent-harness run --config models.toml --model-profile llamacpp --image photo.png "What is in this picture?"
```

- **API:** `POST /runs` and `POST /runs/{id}/messages` take
  `"images": ["https://...", "data:image/png;base64,..."]`, up to 16.
- **Formats:**
  - Files can be PNG, JPEG, GIF, or WebP, and are sent as base64 `data:` URLs (under
    15 MB each).
  - `https://` links are passed to the server.
  - Native Ollama (`api_mode = "native"`) accepts data URLs only.
- **Storage and context:** images are stored with the message, and each counts as
  about 1,000 tokens toward `context_window`.
- **Transcript:** `GET /runs/{id}/messages` lists them on user turns.
- **Requirement:** the model must be a vision model. Example: a llama.cpp model
  started with `--mmproj`.

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
agent-harness runs     --config models.toml --status awaiting_input,failed
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
- **`runs`** lists runs, newest first: ID, status, profile, steps, tokens, cost, and a
  title taken from the first user message. Only top-level runs are shown by default;
  `--children` includes subagent runs, and `--parent RUN_ID` lists one run's subagents.
  Filter with `--status` (repeat or comma-separate) and `--tenant`. `--limit` sets the
  page size, and a longer listing prints a `--cursor` value for the next page. Use
  `--json` for scripts.

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

## Asking the User (`ask_user`)

With `[interaction] ask_user = true`, runs are offered an `ask_user(question, options?)`
tool for clarifying questions. It is off by default, because a headless worker has nobody
to answer.

```bash
agent-harness run --config models.toml "Deploy the new build"
# run_id=... status=awaiting_input
# question: Which environment should I deploy to?
#   - staging
#   - production
agent-harness answer --config models.toml RUN_ID "staging"   # continues the run
agent-harness answer --config models.toml RUN_ID --decline   # the model proceeds on its own
```

- The run pauses in `awaiting_input` through the approval path. The pending request has
  `kind = "question"`, and its `arguments` hold the question and options. It shows up
  in `GET /runs/{id}` (`pending_approval`) and `GET /runs/{id}/approval`.
- Answer over HTTP with `POST /runs/{id}/answer {"call_id", "answer"}` (operator role). The
  run is re-enqueued. `POST /runs/{id}/approval` with `"decision": "deny"` declines. In
  code: `orchestrator.answer(run_id, call_id=..., answer=...)`, then `run`/`resume`.
- The answer reaches the model as the tool result (`The user answered: ...`). Events
  record only `answer_chars` and the acting principal.
- Time spent waiting does not count against `max_wall_seconds`. A question from a
  subagent pauses its parent and is answered through the parent's run ID.
- `ask_user` is read-only, so it is also available in plan mode and while planning.
- Find runs waiting for an answer or approval with
  `agent-harness runs --status awaiting_input`.

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
- **Nothing is offered until agents are defined.** Without `[[agents]]` the model never
  sees the `agent` tool.
- **Nesting.** With the default `max_depth = 2`, a subagent is also offered `agent` and
  can start its own subagents. Set `max_depth = 1` to let only the top-level run
  delegate.
- **Working example.** `examples/subagents.toml` defines a `researcher`, `reviewer`, and
  `test-runner` with prompt files in `examples/agents/`.

**Task list.** `agent-harness status RUN_ID` and `GET /runs/{id}` show the task list
and the subagents.

## Instructions, Skills, Templates, Settings Layers, and Plugins

```toml
[instructions]                    # project conventions for every new run
enabled = true
files = ["~/.agent-harness/AGENTS.md", "AGENTS.md"]   # later = more specific
subdirectories = true             # also AGENTS.md in each directory down to the cwd
filenames = ["AGENTS.md"]

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
With `subdirectories = true`, a run started from `services/api/` also loads
`services/AGENTS.md` and then `services/api/AGENTS.md` (relative to the config file's
directory), after `files`. Directories outside the project, and symlinks that point
outside it, are ignored.

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

## Rewind

With coding tools enabled, every file an edit tool is about to change is snapshotted
first. Rewind undoes a run back to just before a model step:

```bash
agent-harness rewind --config models.toml RUN_ID --to-step 3          # dry run: shows what would change
agent-harness rewind --config models.toml RUN_ID --to-step 3 --yes    # apply
agent-harness continue --config models.toml RUN_ID "Try a smaller change instead"
```

**What a rewind does:**
- **Files:** each changed file goes back to how it was before that step. Files created
  since are deleted.
- **Conversation:** messages from that step on move to the run's archive; nothing is
  deleted. The run is then ready to continue.
- **Side effects it cannot undo:** commands, commits, and external calls are listed.

Over HTTP: `POST /runs/{id}/rewind {"to_step": 3, "confirm": true}`. Without
`confirm`, this is a dry run.

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

## Guardrails

`[guardrails] enabled = true` checks user input, model replies, and untrusted tool
output for prompt injection, jailbreaks, system-prompt leaks, secrets, and PII. A
blocked turn gets a refusal message, and the conversation continues. After untrusted
tool output, side-effect tools need approval (taint mode). An optional guard model
(Llama Guard 3 on Ollama) adds content-safety classification. Detection rates are
measured with the `guardrails` benchmark suite.

On the bundled 30-prompt sample, patterns plus `llama-guard3:1b` caught 14 of 15
attacks with no false positives. See [docs/GUARDRAILS.md](docs/GUARDRAILS.md) for the
detectors, actions, events, and limits.

## Structured Output

A run can be required to end with a JSON result that matches a schema:

```bash
agent-harness run --config models.toml --output-schema examples/schemas/incident_summary.json \
  "Summarize the incident in incident.md"
# stdout: the validated JSON result
```

Over HTTP, pass `"output_schema": {...}` to `POST /runs`; `GET /runs/{id}` returns the
validated value as `result` (and the final text as `output`, as before). In code:
`create_run(..., output_schema=schema)`, then read `run.output`.

- **Schema vocabulary:** the same subset tools use (`type`, `properties`, `required`,
  `additionalProperties`, `items`, `enum`, min/max bounds, `pattern`). Anything else is
  rejected when the run is created.
- **Run without tools:** the schema is sent as `response_format` (`json_schema`), or as
  `format` for native Ollama, so the server constrains the reply.
- **Run with tools:** constraining every reply would stop tool calls, so the model gets a
  `submit_result` tool whose parameters are the schema (non-object schemas are wrapped
  as `{"result": ...}`). A valid call ends the run without another model turn. An
  invalid call returns the validation error to the model.
- **Plain-text answers:** a final answer that parses as valid JSON is accepted in both
  modes. Otherwise the model is asked again; with tools, that request sets
  `tool_choice` to `submit_result`. After 2 rejected answers the run fails. Each
  rejection is an `output_rejected` event.
- **Continue:** a continued run must produce a new result.
- **Limits:** `output_schema` cannot be combined with plan-and-execute (`--plan`).
  Servers differ in how strictly they enforce `response_format` and `tool_choice`
  (Ollama's native API has no `tool_choice`), so the harness always validates the result
  itself.

`ModelRequest.response_schema` and `ModelRequest.tool_choice` (`"auto" | "none" |
"required" | <tool name>`) are also available to custom callers of a model client. Both
default to `None`, which leaves the request payload unchanged.

## Planning and Parallel Delegation

**Plan-and-execute.** Turn it on for one run with `agent-harness run --plan "..."` or
`"planning": true` on the API, or for every new run with `[planning] enabled = true`.
The model proposes the plan; the orchestrator owns it and drives execution:

1. **Plan.** The model explores with read-only tools, then calls `create_plan`: a goal,
   plus steps with ids, dependencies, and an optional `agent` per step.
   - Nothing that changes state can run before the plan exists.
   - The plan is validated: unique ids, known dependencies, no cycles, known agents.
2. **Execute.** The orchestrator picks the steps that are ready.
   - **Agent steps:** steps naming an agent are delegated to that subagent directly, with
     no model turn. Ready agent steps run **in parallel**, and the parent's remaining
     tokens and cost are split between them.
   - **Other steps:** the orchestrator gives the main model one step at a time. The model
     reports with `update_plan`, or a final answer completes the step.
3. **Replan.** A failed step asks the model to call `revise_plan`. Completed steps are
   kept. There are at most `max_replans` revisions (default 2); after that the run fails
   with a clear reason.
4. **Finish.** When every step is done, the orchestrator asks for the final answer.

The plan is stored on the run and changes are recorded as `plan_created`,
`plan_step_updated`, and `plan_revised` events. `agent-harness status RUN_ID` and
`GET /runs/{id}` show it. A crash or an approval pause resumes at the same step.

```toml
[planning]
enabled = false        # default for new runs; --plan / --no-plan override per run
max_plan_steps = 20
max_replans = 2
```

**Parallel work without a plan.** When one model response asks for several things, the
orchestrator runs independent work concurrently:
- **Read-only tool calls** run together. Results are recorded in the original order, with
  the same events.
- **Several `agent` calls** start their subagents together.

Calls that a permission rule denies or asks about are not run early, and nothing is run
early when a `pre_tool` hook is configured, since a hook could change the arguments.

## Tools: Catalog, Sources, MCP, and Dynamic Loading

Tools come from **sources** declared in config. `examples/tools.toml` shows all of them.

| Source | Config | Tool names |
|---|---|---|
| Built-in tools | always (coding tools when `[coding] enabled`) | `echo`, `read_file`, ... |
| Web search (Tavily) | `[web_search] enabled = true`, key in `TAVILY_API_KEY` | `web_search` |
| Your Python code | `[[tools]] kind = "python"`, `module = "pkg.mod:register"` | whatever you register |
| Your Python code, sandboxed | the same, plus `sandbox = "container"` and `path = "tools"`: runs in the run's warm container, no network by default | whatever you register |
| HTTP APIs | `[[tools]] kind = "http"` with `url`, `method`, `input_schema` | the entry's `name` |
| MCP servers | `[[mcp_servers]]` with `command = [...]` (stdio) or `transport = "http"` | `<server>__<tool>` |

**The catalog is persistent.** Every tool is recorded in the run store (`tool_catalog`
table): its source, spec hash, risk, and trust.
- **Handlers are code:** they are re-attached from their source at every start.
- **Switches persist:** `agent-harness tools disable NAME` survives restarts.
- **Removed tools are kept:** a tool its source no longer provides is marked unavailable,
  not deleted.
- **Runs record versions:** each run records the version hash of every tool it was
  offered (`RUN_CREATED.tools`).

```bash
agent-harness tools list    --config models.toml          # --all includes disabled ones
agent-harness tools sync    --config models.toml          # discover MCP servers' tools
agent-harness tools disable web_search --config models.toml
```

**MCP servers.** The client is built in, and needs no extra dependency.
- **Transports:** stdio subprocesses and streamable HTTP, with tool discovery across
  pages.
- **Lifecycle:** servers start on first use. Commands that execute runs (`run`,
  `continue`, `resume`, `approve`, `serve`, `worker`) sync servers that set
  `sync_on_start`. A server that is down keeps the tools from its last sync.
- **Approval:** MCP tools ask for approval unless you list them in `read_only` (or set
  `trust_annotations = true` to accept the server's read-only hints).
- **Schemas:** they are reduced to the subset the harness validates. A tool whose schema
  cannot be reduced is skipped and reported by `tools sync`.
- **Secrets:** they are passed by environment-variable **name** (`env = ["GITHUB_TOKEN"]`,
  `headers_env = {...}`). A stdio server sees only those variables plus `PATH`, `HOME`,
  and locale variables.

**Sandboxed Python tools.** With `sandbox = "container"`, your module runs in a container
instead of the harness process:
- **Isolation:** no network unless `network = true`, a read-only root, dropped capabilities,
  CPU/memory/process limits, and your module mounted read-only.
- **Discovery:** its tool definitions are read inside a one-shot container during
  `tools sync` and cached in the catalog.
- **Calls:** each call runs in the run's warm container, about 50–150 ms per call once warm.

Docker must be available where runs execute.

**Untrusted output.** Output from web search, HTTP tools, and MCP servers is wrapped in
`<untrusted-tool-output>` tags, with a note that it is data, not instructions. This is a
defence against prompt injection. Set `trust = "trusted"` on an HTTP tool only for
services you control.

**Hundreds of tools.** With `[tool_loading] mode = "dynamic"` the model is offered only the
`core` tools plus `search_tools` and `load_tools`.
- **Finding tools:** `search_tools` ranks the catalog by name and description (SQLite
  FTS5).
- **Loading tools:** `load_tools` adds tools to the run. They are saved on the run and
  appended to the end of the tool list, so prompt caches keep working and a resumed run
  offers the same tools.
- **Unloaded tools:** calling a tool that is not loaded returns a message telling the
  model to load it first.

## Production: Storage, Access, Quotas, Security, Deployment

The full guide is in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), and `deploy/` has a
compose setup with Postgres, the API, and two workers.

| Concern | What the harness does |
|---|---|
| **Storage** | SQLite by default. With `pip install -e '.[postgres]'`, `--database postgresql://...` or `[storage] url_env` uses Postgres. Both run versioned migrations on start, and both pass one conformance suite (`tests/test_store_conformance.py`). |
| **API access** | `[[api.keys]]`: each key is a principal with a tenant and a role (viewer, operator, or admin). Keys are configured by SHA-256 or read from the environment. Principals only see their tenant's runs, and approvals and mode changes record who acted. |
| **Quotas** | `[[quotas]]` limits runs, tokens, and cost per tenant over 24 hours, plus concurrent runs. Admission **reserves** budget atomically and caps the run's budget to the grant. The orchestrator re-checks at every step. Refusals are HTTP 429 or CLI exit 2. |
| **Memory** | `[memory] backend = "sqlite"` (FTS5) or `"postgres"` (full-text search). `agent-harness memory migrate --from sqlite:memory.sqlite3` copies every note. |
| **Sandboxing** | `[coding.docker] mode = "per_run"` keeps one warm container per run for coding commands. Sandboxed Python tools run in the run's container. The container is removed when the run ends. |
| **Secrets** | Credential fields accept `NAME`, `env:NAME`, or `file:/run/secrets/name`. |
| **Egress** | With `[network] restrict_egress = true`, HTTP tools, web search, and MCP over HTTP may only reach `allowed_hosts`. |
| **Redaction** | On by default. Tokens and keys are masked in persisted events, run errors, and telemetry. |
| **Retention** | `agent-harness gc --older-than-days N [--yes]` deletes finished runs, then artifacts and file snapshots that no remaining run references (only those older than the cutoff, so a running step's new artifact is never removed). |
| **Artifacts** | `GET /runs/{id}/artifacts/{artifact_id}` serves an artifact that run references, to principals that own the run. With tenant keys, `GET /artifacts/{id}` is limited to all-tenant (`"*"`) keys, because a content hash says nothing about ownership. |
| **Health** | `/healthz` (liveness) and `/readyz` (store reachable, schema version). |
| **Container** | `Dockerfile`: non-root, with the `api` and `postgres` extras. No secrets are baked in. |

## Architecture Diagrams

Flow diagrams are in [`docs/diagrams/`](docs/diagrams/README.md), each as a static SVG, a
self-running animated SVG, and an editable Excalidraw file. They cover:
- the end-to-end flow;
- tool permissions and approval;
- subagents;
- context summarization;
- the tool platform;
- plan-and-execute;
- API access and tenancy.

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
- **Too large to fit:** if the system prompt, first user message (with any summary), and
  latest turn alone do not fit, the run fails with `ContextWindowExceededError`. It does not send a
  truncated request.

Tune the behavior with `build_application(..., context_policy=ContextPolicy(...))`.

**Summarization (optional).** Instead of dropping the omitted turns, a model can summarize
them:

```toml
[summarization]
enabled = true
profile = "fast"          # profile or route that writes summaries; empty = the run's own
max_output_tokens = 1024  # keep well below the smallest context_window
max_input_chars = 0       # 0 = size the transcript from the summarizing profile's window
```

- **When:** only when a trim would omit messages, so at most once per trim point, never
  every step.
- **Where it goes:** the summary is appended to the first user message. The view is then
  the system prompt, the first user message with the summary, and the turns after it.
  Later summaries fold in the previous one.
- **Durable:** the summary is stored on the run and in a `context_summarized` event, so
  resume reuses it instead of calling the model again. Its tokens and cost count against
  the run budget.
- **Failures never fail the run:** the view falls back to plain omission, and the same
  trim point is not retried.
- **Hooks:** a `pre_compact` hook runs first and can deny the summary.
- **Rewind:** rewinding behind the summarized range clears the summary.

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
| `GET /runs` | List runs newest first: `status`, `parent`, `children`, `created_after`, `created_before`, `limit` (≤ 200), `cursor` → `{"runs", "next_cursor"}` |
| `POST /runs` | `{"prompt", "system"?, "model_profile"?, "budget"?, "output_schema"?, "user"?}` → `202 {"id", ...}` (queued) |
| `GET /runs/{id}` | Status, usage, error, final `output`, and `pending_approval` |
| `GET /runs/{id}/events?after=N` | Event log from index N |
| `GET /runs/{id}/events/stream?after=N` | Server-sent events until the run completes or pauses |
| `GET /runs/{id}/stream?after=N` | The same, plus live `delta` token previews (`text`/`reasoning`) for chat UIs |
| `GET /runs/{id}/messages` | Chat transcript: user and assistant turns, `ask_user` questions and answers (`?reasoning=true` adds reasoning) |
| `POST /runs/{id}/cancel` | Cancel |
| `POST /runs/{id}/resume` | Re-enqueue an interrupted run |
| `GET /runs/{id}/approval` | The pending tool call, with its arguments |
| `POST /runs/{id}/approval` | `{"call_id", "decision": "allow" \| "deny"}`, then re-enqueue |

`GET /runs` lists only the caller's tenant. An all-tenant (`"*"`) key may add
`tenant=` to pick one; any other key gets `403` for another tenant.

All routes except `/healthz` require `Authorization: Bearer $AGENT_HARNESS_API_TOKEN`.
`--insecure-no-auth` is accepted only with a loopback `--host`. Inputs are validated at
the boundary: unknown fields, oversized text, and out-of-range budgets are rejected.

**Worker reliability.**
- **Dead-lettering.** The queue counts every claim of a run. A run claimed more than
  5 times without finishing is **dead-lettered**: it fails with a `run_dead_lettered`
  event instead of crash-looping forever. That covers a worker killed each time (OOM)
  and a lease lost each time. A run that keeps raising errors (3 times) is
  dead-lettered too. Resume it explicitly after investigating.
- **Atomic enqueue.** `POST /runs` creates the run and puts it on the queue in one
  transaction.
- **Reconciler.** Each worker requeues *queue-dispatched* runs that are unfinished,
  idle for 5 minutes, and missing from the queue. That happens when a process stopped
  between saving a run and enqueueing it, for example after an answer or approval.
  Runs started from the CLI (`agent-harness run`) are never picked up by workers.
- **Cancel.** A cancelled run's in-flight model or tool call is stopped within about
  2 seconds, instead of at the next step.

### Building a chat front end on the API

| Chat action | Endpoint |
|---|---|
| New chat | `POST /runs {"prompt", "user": "<end-user id>"}` |
| Send a message | `POST /runs/{id}/messages {"content"}` |
| Show the reply as it is written | `GET /runs/{id}/stream` |
| Render the conversation | `GET /runs/{id}/messages` |
| Sidebar | `GET /runs?user=<id>` |
| Stop | `POST /runs/{id}/cancel` |
| Answer a question or approve | `POST /runs/{id}/answer` and `POST /runs/{id}/approval` |

- **Token previews.**
  - `delta` events in `GET /runs/{id}/stream` come from runs executed **in the API
    process** (`agent-harness serve --workers N`); separate `worker` processes publish
    none, and clients then see each reply when it is recorded.
  - Previews are not authoritative: render the final text from `model_responded` or
    `GET /runs/{id}/messages`.
  - With output guardrails on, replies are checked before they are shown, so no
    previews are sent.
- **`user`** records which end user a service created the run for. It goes on the
  run, in `run_created` for audit, and is filterable in `GET /runs?user=` and
  `agent-harness runs --user`. It is set by an operator key on the user's behalf;
  it is not authentication.
- **CORS.** Allow browser origins with `[api] cors_origins = ["https://chat.example.com"]`
  (`"*"` allows any origin). Preflight requests are answered without authentication;
  every other request still needs a bearer key.
- **Webhooks.** These avoid polling for paused or finished runs:
  ```toml
  [notifications]
  webhook_url = "https://hooks.example.com/agent-harness"
  secret_env = "AGENT_HARNESS_WEBHOOK_SECRET"   # adds X-Agent-Harness-Signature: sha256=<HMAC of the body>
  statuses = ["awaiting_input", "completed", "failed", "cancelled", "budget_exhausted"]
  ```
  - Workers POST `{"run_id", "status", "tenant", "end_user", "parent_run_id", "error",
    "updated_at"}`, never message text.
  - Delivery is best effort (2 retries) and never affects the run.

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

Each pipeline run is kept in its own folder, `validation-results/runs/<time>/`.
`validation-results/index.html` is the run history. Watch a run, then debug from that
run's report, which includes
per-item transcripts of failures and a telemetry summary: see
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).


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

*Update:* the queue and lease layer (Phase 4c), durable idempotency (Phase 4a), an
optional OpenTelemetry exporter (Phase 4b), and a Postgres store for multi-host
deployments (Phase 9) have since been added. Approvals are still collected by polling
(`GET /runs/{id}/approval` or `agent-harness runs --status awaiting_input`); there is no
push notification channel yet.
