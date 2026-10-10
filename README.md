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

Edit `models.toml` so the endpoint and model name match your server. The default profile
is `llamacpp` (a local llama.cpp server, no credential). Switch any profile off with
`enabled = false` in its table, or `AGENT_HARNESS_MODEL__NAME__ENABLED=false`; a disabled
profile cannot be selected, and routes and fallbacks to it are dropped.

With `./models.toml` in place, no subcommand or `--config` is needed:

```bash
agent-harness                                   # interactive chat (the `chat` subcommand)
agent-harness "Multiply 27 by 19."              # same as: agent-harness run "..."
agent-harness --stream --model-profile ollama "Hello"
```

The config file is found in this order: `--config FILE`, `$AGENT_HARNESS_CONFIG`,
`./models.toml`, `~/.agent-harness/models.toml`.

**Local settings in `.env`.** `agent-harness`, `agent-harness-bench`, and
`scripts/validate.sh` read `./.env` at startup (`cp .env.example .env`). It holds what must
not be committed:
- credentials (`MODAL_LLM_API_KEY`, the Modal proxy token, `TAVILY_API_KEY`, ...);
- real endpoints (for example `AGENT_HARNESS_MODEL__MODAL__ENDPOINT`);
- the profile switch, `AGENT_HARNESS_MODEL_PROFILE`.

Variables already set in your shell win over `.env`. `.env` and `.env.*` are gitignored
(only `.env.example` is tracked), and unit tests and the offline validation stages never
read it. Every subcommand accepts `--config` to
override it. In `chat`, each message continues the same durable run; approvals and
`ask_user` questions are asked inline; `/new`, `/status`, `/help`, and `/exit` are chat
commands, and any other `/name` expands a prompt template. Each message gets
`--max-steps` (20) more steps and `--max-wall-seconds` (300) more seconds.
The repository's own `models.toml` turns every feature on (see "Everything On" below).

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

## Everything On (`models.toml`)

The code defaults keep every optional feature off, for these reasons:
- **Side effects.** Coding tools write files and hooks run programs.
- **Cost.** Summaries, guard models, and subagents each make extra model calls.
- **Outside services.** Docker, a Tavily key, an embedding model, and MCP servers are needed by some features.
- **Behaviour changes.** Guardrails can refuse, and permission rules pause runs.
- **Stable hashes.** Defaults feed the version hashes and the prompt prefix.

The repository's `models.toml` opts into all of them, so `agent-harness` run from the
repository root starts with everything on:

| On | Section |
|----|---------|
| llama.cpp default, serving chat and embeddings from one router; Ollama self-contained as an alternative; Modal switched off (`enabled = false`) | `[models.*]`, `[routes]` |
| Coding tools in a warm per-run Docker container (`python:3.12`, has git); `ls`, `cat`, `head`, `wc`, and `find` allowlisted and pre-approved | `[coding]`, `[coding.docker]`, `[permissions]` |
| Permission rules, an audit hook (`examples/hooks/audit_hook.py`), `ask_user`, approval and question deadlines, task list | `[permissions]`, `[[hooks]]`, `[interaction]`, `[tasks]` |
| Three subagents (`examples/agents/`), nesting depth 2, at most 2 running at once | `[[agents]]`, `[subagents]` |
| Planning offered when it helps (`mode = "auto"`, with a reminder after 6 tool steps); an answer on the last step and near the time limit; today's date in the first message | `[planning]`, `[orchestrator]` |
| Instruction files, skills (`examples/skills/`), templates (`examples/templates/`) | `[instructions]`, `[skills]`, `[templates]` |
| Summarization, context policy | `[summarization]`, `[context]` |
| Python and HTTP tool sources, `web_fetch`, `web_search`, hybrid memory, artifacts, dynamic tool loading | `[[tools]]`, `[web_fetch]`, `[web_search]`, `[memory]`, `[artifacts]`, `[tool_loading]` |
| Guardrails with taint mode | `[guardrails]` |
| Egress allowlist, redaction, tenant and user quotas, API rate limit and CORS | `[network]`, `[redaction]`, `[[quotas]]`, `[user_quotas]`, `[api]` |

Some features need something the harness cannot provide. They are at the end of the
file, commented out, each with what it needs:
- per-tenant API keys
- webhooks
- MCP servers (Node.js)
- the Llama Guard classifier
- Postgres

`web_search` is on but needs `TAVILY_API_KEY` at call time. `agent-harness config show`
prints every value and the file it came from. `tests/test_full_config.py` checks that
every listed feature stays on and every path in the file exists.

Turn a feature off by editing its section, or for one process with
`AGENT_HARNESS_SET__SECTION__KEY=value`, e.g. `AGENT_HARNESS_SET__GUARDRAILS__ENABLED=false`.

**Coding is on for every run.** With `[coding] enabled = true`, every run (CLI, `/ui`,
`/v1`, the REPL) gets the coding system prompt and the coding tools, including
`run_command`, even for a plain chat question. There is no per-run switch yet; turn it
off with `AGENT_HARNESS_SET__CODING__ENABLED=false` for a chat-only server.

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
AGENT_HARNESS_MODEL_PROFILE          # which profile serves everything (chat, embeddings, guard)
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

## One Profile Serves Everything

Each provider profile is self-contained:
- **Chat:** `model`.
- **Embeddings:** `embedding_model`, used by `[memory]` hybrid/semantic search and the memory
  benchmark.
- **Guard model:** `guard_model`, used by `[guardrails.model]`.

Switching the profile switches all three; no setting points at a second provider. The
switch is one variable, normally kept in `.env`:

```bash
AGENT_HARNESS_MODEL_PROFILE=llamacpp   # or: ollama | modal
```

| `AGENT_HARNESS_MODEL_PROFILE` | Chat, embeddings, and guard model come from |
|---|---|
| `llamacpp` | the llama.cpp router (`:9931`) |
| `ollama` | Ollama (`:11434`) |
| `modal` | the three Modal deployments (endpoints in `.env`) |

It overrides `default_model_profile` in `models.toml` everywhere: `agent-harness`, chat,
`serve`, `doctor`, and the live validation stages. A shell variable overrides `.env` for one
command: `AGENT_HARNESS_MODEL_PROFILE=ollama agent-harness "hi"`.

```toml
[models.llamacpp]
adapter = "llama_cpp"
endpoint = "http://127.0.0.1:9931/v1"                         # llama serve router
model = "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"
embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"   # same router
```

`agent-harness profiles` shows what each profile serves:

```text
llamacpp *: llama_cpp / ggml-org/Qwen3.8-27B-GGUF:Q4_K_M @ http://127.0.0.1:9931/v1
  embed: nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0 @ http://127.0.0.1:9931/v1
```

Per provider:
- **llama.cpp router (`llama serve --models-preset`).**
  - Add the embedding model to the preset with `embedding = true`. The Llama app reads
    `~/.config/llama/models.user.ini`.
  - Allow two loaded models (`--models-max 2`).
  - A single-model `llama-server` needs a second `--embedding` server, named by
    `embedding_endpoint`.
- **Ollama.** `ollama pull nomic-embed-text` on the same server.
- **Modal.** One deployment serves one model, so the `modal` profile names three, all with
  the same Modal auth:
  - `deploy/modal/vllm_server.py` (chat, `endpoint`)
  - `vllm_embed_server.py` (`embedding_endpoint`)
  - `vllm_guard_server.py` (Llama Guard, `guard_endpoint`; needs an `agent-harness-hf`
    secret with `HF_TOKEN`, because the model is gated)

`agent-harness doctor` also checks the embedding and guard models.
`AGENT_HARNESS_MODEL_PROFILE=ollama scripts/validate.sh` validates another provider end
to end: smoke and benchmark entries without a `profile` follow the default.

## Ollama, llama.cpp, and Modal

Built-in adapters: `openai_compatible`, `ollama`, `llama_cpp`, and `modal`.
`examples/models.toml` has a working profile for each.

| Adapter     | Server                                  | Notes                                                                                     |
|-------------|-----------------------------------------|-------------------------------------------------------------------------------------------|
| `ollama`    | `ollama serve`                          | Handles tool calls without IDs. `api_mode = "native"` uses `/api/chat` and sends `context_window` as `num_ctx`. |
| `llama_cpp` | `llama-server -m model.gguf --jinja -c N` | `--jinja` is required for tool calls. 503 while loading is retried.                        |
| `modal`     | vLLM or SGLang on Modal (`server = "vllm"` or `"sglang"`) | Requires both the server bearer key (`api_key_env`) and Modal proxy auth (`headers_env` for `Modal-Key` and `Modal-Secret`). The first request uses `cold_start_timeout_seconds` (default 600). |

Deploy scripts for Modal are in `deploy/modal/` (`examples/modal/` keeps the reference
copies). They keep prefix caching on, turn on cached-token reporting, and set container
options that keep a run on one warm KV cache.

**The Modal chat model.** `deploy/modal/vllm_server.py` serves the model the local profile
runs, Qwen 3.6 35B-A3B (`Qwen/Qwen3.6-35B-A3B`, BF16, 71.9 GB of weights, public), on one
**H200**, with `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` from the vLLM
recipe, which lists the H200 as tested hardware. The earlier Qwen2.5-7B on an L40S often
answered without using tools. Only the chat model moved: the embedding and guard
deployments stay separate, on an L4 each, and scale to zero when idle.

| Setting | Value | Where |
|---|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` | `MODEL_NAME` in the script = `[models.modal] model` (a test checks it) |
| GPU and price | `H200` (141 GB; about 55 GB left for the KV cache), $0.001261/s | `GPU` in the script, `[models.modal.pricing] gpu_usd_per_second` |
| Context | 32768 | `MAX_MODEL_LEN` = `context_window` |
| First start | up to 20 minutes (71.9 GB download into the `agent-harness-hf-cache` volume) | `cold_start_timeout_seconds = 1200` |
| Thinking | returned as reasoning; off for summaries | `REASONING_PARSER`, `[summarization] parameters` |

To deploy:
1. `modal deploy deploy/modal/vllm_server.py` (and the embed and guard scripts, once).
2. `agent-harness doctor --model-profile modal`, then restart `serve`.

**Cheaper GPU option.** The script also has an FP8 preset (`Qwen/Qwen3.6-35B-A3B-FP8`, 37.5 GB,
on one `A100-80GB`, $0.000694/s). It is untested: the A100 has no FP8 hardware, and vLLM
documents weight-only FP8 (Marlin) for dense models, not for this MoE. The model also has a
vision encoder; `EXTRA_ARGS = ["--language-model-only"]` skips it if your vLLM accepts the
flag.

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
- **Tool calls per reply:** a reply with more than 32 tool calls is refused as a whole
  (none of them run) and the model is asked again. After `max_refused_repeats` such
  replies the run fails. Tune it with `OrchestratorSettings(max_tool_calls_per_turn=...)`,
  or disable it with `None`.
- **Output cap:** every example profile sets `max_tokens = 1024` in its `parameters`, so a
  runaway reply stops with `finish_reason=length` instead of generating until the context
  fills or the request times out. Ollama's native mode sends it as `num_predict`. An
  `AGENT_HARNESS_MODEL__X__PARAMETERS` variable replaces the whole table, so include
  `max_tokens` there too.
- **Visible retries:** when the adapter retried before succeeding, the response, the
  `model_responded` event, and the `model_call` telemetry record carry
  `provider_attempts`, `retried_errors` (e.g. `["timeout"]`), and `attempt_timeout_seconds`.
  The report's telemetry table counts them as `retried timeout: N`.
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
max_parallel = 2                  # children of one parent running at once (0 = no limit)

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
- **How many at once.** `max_parallel` caps the children of one parent that run
  together; the rest wait for a slot. A child that waited gets no more time than its
  parent has left. A local llama.cpp server shares one KV cache (`ctx-size`) between
  concurrent requests: five 16K-token children overflowed a 32K cache ("Context size has
  been exceeded"), so the root config uses 2.
- **Ending with an answer.** With `[orchestrator] final_answer_on_last_step = true` (root
  config), a subagent's last allowed step is an answer: a `[Budget]` note and
  `tool_choice = "none"`. A child that still does not finish reports what it had found
  (its context summary) to the parent, not only its error.

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
Each file gets a header naming the file only (`## Instructions from CHAT.md`); a file
found in a subdirectory keeps its relative path (`services/api/AGENTS.md`), since its
rules apply there. A full path such as `.agent-harness/instructions/CHAT.md` in the
prompt led a model to read "explain how agent-harness works" as a question about that
folder.

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
mode = ""              # "off" | "always" | "auto"; empty keeps `enabled` (true = "always")
auto_reminder_after_steps = 0   # "auto": remind once after N tool steps in a turn (0 = never)
max_plan_steps = 20
max_replans = 2
```

**Planning when it helps (`mode = "auto"`, the root config).** Top-level runs without an
output schema are offered `create_plan` but are not asked to plan. A successful call turns
the run into a planned run (`plan_created` with `auto: true`). With
`auto_reminder_after_steps = 6`, a turn that has made 6 tool-calling steps without a plan,
or whose context was summarized, gets one `[Plan]` reminder: plan and give the reading to
a subagent, or answer from what it has.

**Finishing.** With `[orchestrator] final_answer_on_last_step = true`, the answer step of a
finished plan is sent with `tool_choice = "none"`, so the model writes the answer instead
of reading more. The `[Budget]` last-step note also comes early when less than
`final_answer_reserve_seconds` (default 120, at most a quarter of the time budget) is
left, and an answer to it ends a planned run even with steps left.

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
| Web page fetch | `[web_fetch] enabled = true` | `web_fetch` |
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
- **Resources and prompts:** with `resources = true` and/or `prompts = true` on the
  server entry, a server that offers them gets four read-only tools:
  `<server>__mcp_list_resources`, `<server>__mcp_read_resource` (by URI),
  `<server>__mcp_list_prompts`, and `<server>__mcp_get_prompt` (name plus string
  arguments). Servers that offer only resources or prompts are accepted too.
- **Changes:** when a server sends `notifications/tools/list_changed` (or the
  resources/prompts equivalent), the harness re-syncs that server in the background:
  new tools are registered, and tools it no longer lists are marked unavailable.

**Fetching web pages.** `[web_fetch] enabled = true` adds `web_fetch(url, max_chars?)`.
It returns the page's text, title, and final URL, and reads HTML, plain text, JSON, and
XML.
- **Safety:** private, loopback, and link-local addresses are refused unless
  `allow_private = true`. Redirects are followed one hop at a time (up to
  `max_redirects`), and every hop is checked against that rule and against
  `[network] allowed_hosts`.
- **Limits:** `max_bytes` (download size, default 2 MB) and `max_chars` (text returned,
  default 20,000).
- **Trust:** the result is untrusted output, like web search.

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

## Limits, Deadlines, and Overrides

Every limit below is set in the config file, and any table value can be changed
without editing it: `AGENT_HARNESS_SET__SECTION__KEY=value` (more `__` parts for nested
tables; the value is read as TOML, so `60`, `1.5`, `true`, and `["a"]` work, and other
text is a plain string; names are lower-cased).

```toml
[orchestrator]                    # loop limits; 0 turns an optional check off
max_model_attempts = 3            # provider requests per step and profile (incl. adapter retries)
max_tool_attempts = 3
max_identical_tool_calls = 5      # identical calls before they are refused
max_refused_repeats = 2           # refusals before the run fails (-1: never)
max_tool_calls_per_turn = 32      # a larger reply is refused whole
context_overflow_retries = 2      # retries with a smaller context when the server says "too long"
context_overflow_shrink = 0.5     # used when the server does not say by how much
context_overflow_patterns = []    # extra regexes that mark an error as an overflow

[interaction]
approval_timeout_seconds = 0      # 0 waits forever
question_timeout_seconds = 0      # for ask_user questions
on_timeout = "deny"               # deny | cancel | fail

[api]
rate_limit_per_minute = 0         # per API key (or client address); 0 = off
rate_limit_burst = 0              # 0 = rate_limit_per_minute

[user_quotas]                     # per end user (`user` on POST /runs), within the tenant; 0 = no limit
max_runs_per_day = 0
max_tokens_per_day = 0
max_cost_usd_per_day = 0.0
max_concurrent_runs = 0
# [user_quotas.users.alice]       # replaces the values above for one user
# max_runs_per_day = 100
```

For example, `AGENT_HARNESS_SET__API__RATE_LIMIT_PER_MINUTE=120` or
`AGENT_HARNESS_SET__USER_QUOTAS__USERS__ALICE__MAX_RUNS_PER_DAY=100`.

- **Context overflow:** when a server rejects a request as larger than its context
  (vLLM, SGLang, llama.cpp, and OpenAI messages are recognized), the step is retried with
  a smaller view. The new input cap is the estimate times `context_overflow_shrink`, or the
  ratio the server states less 10%. The cap is persisted in the `model_responded` event
  (`context_overflow`, `max_input_tokens`) and applies to the rest of the run on that
  profile, so later turns stay append-only between trims. This also covers profiles
  without `context_window`. The retries count against `max_model_attempts`.
- **Deadlines:** a run paused for an approval or an `ask_user` answer past its timeout is
  resolved by `on_timeout`. `deny` denies the tool (or declines the question) and the run
  continues; `cancel` and `fail` end it. Workers check at every reconcile (60 s by
  default) and enqueue runs that continue; `agent-harness expire` does the same once (for
  cron, or without workers). The resolution is recorded with `actor = "timeout"`.
- **Rate limits:** requests beyond the burst get HTTP 429 with `Retry-After`. Failed
  authentication is limited by client address. `/healthz` and `/readyz` are never
  limited. The state is per API process.
- **Per-user quotas:** checked with the tenant's `[[quotas]]` in the same atomic
  reservation, and again at each step. A refusal names the user (HTTP 429, CLI exit 2).
  Runs without `end_user` are not limited by them.

## Production: Storage, Access, Quotas, Security, Deployment

The full guide is in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), and `deploy/` has a
compose setup with Postgres, the API, and two workers.

| Concern | What the harness does |
|---|---|
| **Storage** | SQLite by default. With `pip install -e '.[postgres]'`, `--database postgresql://...` or `[storage] url_env` uses Postgres. Both run versioned migrations on start, and both pass one conformance suite (`tests/test_store_conformance.py`). |
| **API access** | `[[api.keys]]`: each key is a principal with a tenant and a role (viewer, operator, or admin). Keys are configured by SHA-256 or read from the environment. Principals only see their tenant's runs, and approvals and mode changes record who acted. |
| **Quotas** | `[[quotas]]` limits runs, tokens, and cost per tenant over 24 hours, plus concurrent runs. `[user_quotas]` adds the same limits per end user. Admission **reserves** budget atomically and caps the run's budget to the grant. The orchestrator re-checks at every step. Refusals are HTTP 429 or CLI exit 2. |
| **Rate limits** | `[api] rate_limit_per_minute` / `rate_limit_burst` per API key: HTTP 429 with `Retry-After`. |
| **Deadlines** | `[interaction] approval_timeout_seconds` / `question_timeout_seconds` with `on_timeout`: workers (or `agent-harness expire`) resolve paused runs that waited too long. |
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
- API access and tenancy (keys, rate limits, tenant and user quotas);
- context-overflow recovery;
- limits and deadlines (where each limit applies, and how paused runs expire).

## Context Window

Profiles that set `context_window` get a bounded view of the history on every model call.
The stored run history is never modified.

- **Always kept:** system messages and the first user message. When a summary or trim
  covers the newest message the user wrote (`[Plan]`/`[Budget]` notes do not count), its
  text is added after the summary as `[Latest user message - answer this one]`, so the
  second question of a chat is never answered as the first.
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
- **When the server still says "too long":** trimming and summarization work from an
  estimate (3.5 characters per token), and profiles without `context_window` are not
  trimmed at all. Code, JSON, and non-English text use more tokens per character, and the
  server's chat template and tool rendering add tokens of their own. A provider
  rejection is therefore retried with a smaller, persisted cap; see
  [Limits, Deadlines, and Overrides](#limits-deadlines-and-overrides) and
  [the diagram](docs/diagrams/08-context-overflow.svg).

Tune the behavior in `[context]` (or `AGENT_HARNESS_SET__CONTEXT__CHARS_PER_TOKEN=2.5`):

```toml
[context]
chars_per_token = 3.5          # lower it (about 2.5) for code, JSON, or non-English text
high_water_ratio = 0.85        # trim when the estimate passes this share of the budget
low_water_ratio = 0.60         # ... down to this share
message_overhead_tokens = 8
max_tool_output_chars = 16000  # 0 = never shorten tool outputs in the view
```

`build_application(..., context_policy=ContextPolicy(...))` still takes precedence.

**Exact counts from the server (optional).** With `token_counting = "server"` on a
profile that sets `context_window`, the request is counted by the server's tokenizer before
it is sent, whenever the estimate is above half the limit:
- **llama.cpp:** the prompt is rendered with `/apply-template` (when available), then
  counted with `/tokenize`.
- **vLLM:** `/tokenize` counts the chat messages and tools.
- **Not available:** Ollama, and SGLang on Modal. Setting it there is a configuration
  error.

If the count exceeds the window less the output reserve, a `context_capped` event
records a smaller cap and the request is rebuilt to fit, the same way as after an
overflow. A failed count is ignored, so the estimate and overflow recovery still apply.
It costs one or two extra local HTTP calls on large turns.

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

**Commands.** `run_command` runs only commands in `allowed_commands`: by default `cargo`,
`git`, `go`, `make`, `npm`, `npx`, `pnpm`, `py.test`, `pytest`, `python`, `python3`, `uv`,
`yarn`. The setting replaces that list, so repeat the defaults when you add to it. The root
config adds `ls`, `cat`, `head`, `wc`, and `find`, and pre-approves them with
`[permissions] allow` rules (`run_command(argv.0=ls)`, ...), while `ask` rules keep a pause
for a `.env` argument. `find` may only list files: `-delete`, `-exec`, `-execdir`, `-ok`,
`-okdir`, `-fls`, and `-fprint*` are refused, because the workspace is mounted read-write
and permission rules only see `argv[0]` (`find . -exec rm {} ;` would pass an `rm` deny
rule). The allowlist is checked when the command runs, after any approval, so a command
outside it is refused even if you approved it.

**Wrong paths.** File tools answer in words the model can act on, using the path as it was
given: `no such file or directory: src/x.py; search_files can find the right path`,
`file could not be read: X (Permission denied)`, or `not a UTF-8 text file`. A bad `cwd`
for `run_command` or the `git_*` tools gets the same message. `search_files` skips files it
cannot read.

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
- **Deadlines.** With `[interaction] approval_timeout_seconds` or
  `question_timeout_seconds`, each reconcile also resolves approvals and questions that
  waited too long (`on_timeout`), and enqueues the runs that continue.
- **Rate limits.** With `[api] rate_limit_per_minute`, a caller over the limit gets
  `429` with `Retry-After`; health checks are exempt.

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

### Built-in web chat (`/ui`)

`agent-harness serve` serves a chat page at **http://127.0.0.1:8000/ui** (`/` redirects
there). It is one static file in the package, with no build step, no CDN, and a strict
Content-Security-Policy. Open its Settings and paste an API key. The key is kept in
`sessionStorage`, or in `localStorage` if you tick "remember". The page calls the same
authenticated API as any other client.

| Area | What it shows |
|---|---|
| Left | Your runs (newest first), a **New chat** button, and the profile or route for new chats |
| Center | The conversation, with streamed replies and reasoning. Approval and plan cards (Approve/Deny with a reason) and `ask_user` cards (option buttons or free text) appear inline. **Stop** cancels the run. |
| Right | Run facts; steps, tokens, cost, and time against the budget; plan steps; tasks; subagents (click to open); and the event log |

Sending a message picks the right call for the run's state:
- a new chat → `POST /runs`
- a finished run → `POST /runs/{id}/messages`
- a working run → `/steer`
- an open question → `/answer`

The page follows the system light/dark setting. The `deploy/` compose file can also start
Open WebUI next to the harness (`--profile chat`).

### Using it from any chat window (OpenAI-compatible)

`agent-harness serve` also speaks the OpenAI chat API, so any client that lets you set an
"OpenAI-compatible" base URL can use the harness as a model. That includes Open WebUI,
LibreChat, Chatbox, AnythingLLM, Continue (VS Code), Msty, and the OpenAI SDKs. Tools,
hooks, guardrails, subagents, and approvals all run on the server; the client only sees
the conversation.

```bash
pip install -e '.[api]'
export AGENT_HARNESS_API_TOKEN="$(openssl rand -hex 24)"   # at least 16 characters
agent-harness serve --host 127.0.0.1 --port 8000 --workers 2
```

| Client setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8000/v1` |
| API key | the value of `AGENT_HARNESS_API_TOKEN` (or an `[[api.keys]]` key) |
| Model | `agent-harness` (the default profile), or any profile or route from `GET /v1/models` |

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=TOKEN)
reply = client.chat.completions.create(model="agent-harness",
                                       messages=[{"role": "user", "content": "Explain cli/repl.py"}])
```

- **One conversation is one durable run.** Clients resend the whole history on every
  message.
  - **Continuing:** the harness looks up a digest of the earlier messages
    (`chat_conversations` table) and continues that run. `X-Agent-Harness-Run: RUN_ID`
    names the run explicitly, and every response carries that header.
  - **New conversations:** an unknown history (a new chat, or an edited or regenerated
    one) starts a new run seeded with those messages.
  - **System prompt:** the client's system message becomes the run's system prompt.
- **Approvals and questions arrive as chat messages.**
  - **Approving:** a paused run replies "Approval needed: `tool` ...". Answer **approve**
    (or yes/ok) to allow it, or **deny** / no with an optional reason. Any other text
    denies the call with that text as feedback.
  - **Plans:** a plan approval accepts approve, or what to change.
  - **Questions:** an `ask_user` question takes your next message as the answer.
- **Streaming.** With `"stream": true`, tokens arrive as `chat.completion.chunk` events.
  Reasoning goes in `delta.reasoning_content`.
  - **When previews are missing:** they come only from in-process workers
    (`serve --workers N`), and output guardrails suppress them. Without previews the
    answer arrives in one chunk.
- **Ignored and refused fields.** Sampling options in the request (temperature,
  max_tokens) are ignored; the profile's `parameters` apply. Client-side `tools` are
  refused, because the harness runs its own.
- **Auth.** `GET /v1/models` needs a viewer key; `POST /v1/chat/completions` needs an operator key.

Open WebUI in Docker, pointed at a harness on the host:

```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY="$AGENT_HARNESS_API_TOKEN" \
  -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

Serve on `0.0.0.0` (with a key) when the client runs in a container that must reach the host.

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
# search = "hybrid"            # keyword (default), semantic, or hybrid
# embedding_profile = "embed"  # a profile whose /embeddings endpoint is used
# vectors = ""                 # SQLite file for vectors (default: next to the notes)
```

**Artifacts.** A tool result larger than the threshold is stored content-addressed.
The model sees a preview plus the artifact ID, and can page through the full text
with the `read_artifact` tool. Events record `artifact_id`, and the API serves the
content at `GET /artifacts/{id}`.

**Memory.** Adds the `memory_search` (read-only) and `memory_write` (side-effect)
tools, backed by SQLite full-text search. Search results are labeled as notes, not
instructions, because anything a model saved earlier is untrusted input.

**Searching memory by meaning (optional).** With `search = "semantic"` or `"hybrid"`, each
note is also embedded through `embedding_profile`. That works with any OpenAI-compatible
`/embeddings` endpoint:
- Ollama, with an embedding model such as `nomic-embed-text`;
- llama.cpp started with `--embeddings`;
- vLLM serving an embedding model.

How the modes work:
- **`semantic`:** ranks notes by cosine similarity.
- **`hybrid`:** merges the keyword and semantic rankings, which helps when a query names
  an exact identifier.
- **Storage:** vectors are kept per embedding model in a small SQLite table.
- **Existing notes:** `agent-harness memory reindex --config models.toml` embeds notes
  saved before (or while the embedding server was down).
- **Outages:** if embedding fails, search falls back to keywords and new notes are still
  saved.
- **Scale:** search is a linear scan, which suits thousands of notes, not millions.

```toml
[models.embed]
adapter = "ollama"
model = "nomic-embed-text"
endpoint = "http://127.0.0.1:11434/v1"
```

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

They can also auto-approve or auto-deny approval pauses. Scenarios can also use these
options:
- `[orchestrator]` limits;
- `ask_user = true` without scripted answers;
- `wait_seconds` (time that passes while a run is paused) with `[interaction]` deadlines;
- a scripted `error = "context_overflow"`;
- `expect.event_payloads`, which requires an event with given values, for example
  `{ type = "approval_resolved", actor = "timeout" }`.

`examples/evals/` covers these behaviours offline:
- overflow recovery;
- tool calls per reply;
- repeated-call refusal;
- question deadlines;
- guardrails;
- structured output;
- `ask_user`.

The scenarios stage of every pipeline run checks them. See `src/agent_harness/bench/evals.py`
for the file format. The command exits 1 if any scenario fails.

**Model-graded answers.** For open-ended questions, the `qa` benchmark suite can ask a
model to grade each answer against the references (and an optional per-record `rubric`)
instead of matching text:

```toml
[[benchmark]]
name = "qa-judged"
suite = "qa"
data = "bench/qa_sample.jsonl"
profile = "ollama"
[benchmark.options]
match = "judge"
judge_profile = "modal"      # default: the profile under test
judge_max_tokens = 1024      # leave room for thinking models
```

Each item records the verdict and reason (`details.judge`). An unreadable verdict counts
as incorrect.

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

Around the budget, the loop has its own guards (`[orchestrator]`, see
[Limits, Deadlines, and Overrides](#limits-deadlines-and-overrides)): provider attempts per
step, repeated identical tool calls, tool calls per reply, and context-overflow retries.
Tenant and end-user quotas cap the budget at admission and are re-checked at every step.

**Ending in an answer.** By default a run that reaches `max_steps` or `max_wall_seconds`
stops with `budget_exhausted`. With `[orchestrator] final_answer_on_last_step = true` (on
in the root config), the last allowed step, and any step taken when less than
`final_answer_reserve_seconds` is left, gets a `[Budget]` note ("This is your last step:
tools are no longer available...") and `tool_choice = "none"`. The tools stay in the
request, so the cached prompt prefix holds. Runs with an `output_schema` are unaffected.

**Answers cut off at the output limit.** `parameters.max_tokens` caps every reply, so a long
final answer can end mid-sentence (`finish_reason: "length"`). With `[orchestrator]
max_answer_continuations = N` (2 in the root config; 0 = off in code), the harness sends a
`[Budget]` note asking the model to continue exactly where it stopped, with
`tool_choice = "none"`, and **joins the continuation to the cut answer**, so the run still
has one answer message (the CLI, API, `/ui`, `/v1`, a subagent's parent, and the plan all see
the whole text). At most N extra turns, and only while the step budget allows. An answer
still cut off after that completes with `run_completed {"answer_truncated": true}`, shown in
`/ui` as "answer cut off at the output limit". Runs with an `output_schema` are not continued.

**Unreadable tool arguments.** A tool call whose arguments are not a JSON object does not
fail the run: the call gets a validation error ("The arguments for X were not valid JSON
... Send the call again with a valid JSON object."), it is not run or prefetched, and the
model can try again.

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

**Datasets.** Run `scripts/fetch_datasets.sh` once. It downloads GSM8K, a BFCL v4 mix, and
HumanEval into `datasets/` (gitignored), which the live benchmarks use.

**Per-feature validation.** `scripts/validate.sh` is the master script. With no
arguments it runs every stage for every feature. Name one or more features to run only
their checks (unit-test modules, scenarios, smoke profiles, and benchmarks, as listed
under `[[feature]]` in `examples/validation.toml`):

```bash
scripts/validate.sh --list                 # core, providers, context, guardrails, hooks-permissions, ...
scripts/validate.sh                        # master: every feature, every stage (needs model servers)
scripts/validate.sh guardrails             # one feature
scripts/validate.sh memory tools --offline # several features, no model server
```

While it runs, a status line shows the stage, elapsed time, a benchmark progress bar, and
time left (estimated from the previous run). Baselines (`baselines/`) are local and gitignored: record your own with
`scripts/validate.sh --update-baselines`. HumanEval checks that Docker answers before it starts.
The live stages use the repository's `models.toml` (llama.cpp default). Start
`llama-server`/the router (and Ollama for the `ollama` and `embed` profiles) first. See
[docs/VALIDATION.md](docs/VALIDATION.md#per-feature-validation).

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

*Update (Phase 12–13):* `[notifications]` webhooks now report runs that pause or finish,
and paused runs can expire (`[interaction] approval_timeout_seconds`). Phase 13 also added
context-overflow recovery, API rate limits, and per-user quotas. Phase 14 added `[context]`
settings, optional server-side token counting, scenario coverage for the Phase 12–13
behaviours, model-graded benchmark answers, semantic memory search, MCP resources and
prompts, and `web_fetch`.
