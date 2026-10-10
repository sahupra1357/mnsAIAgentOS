# Troubleshooting

How to find out *where* a problem is (your environment, a model server, the harness,
or the model itself) and fix it. Most entries come from real validation runs.

## The debugging loop

1. **Read the report.** Open `validation-results/index.html`, the run history. Pick a
   run (the newest is marked "latest") to see which stage failed, which benchmark
   items failed, and a summary of model and tool calls.
2. **Find the layer.** Use the table in the next section. The right layer is usually
   obvious from the first failing stage.
3. **Open the evidence** for that layer. See "Where the evidence is".
4. **Reproduce narrowly.** Rerun one stage, one profile, or one benchmark item, rather
   than the whole pipeline.
5. **Fix, then verify** with the same narrow command, and finally the full pipeline.

| First failing stage | Layer | Look at |
|---|---|---|
| `static`, `unit` | Harness code or your Python environment | The stage detail line; run the command yourself |
| `scenarios` | Harness behaviour (scripted models, no network) | `<run>/scenarios.json` |
| `smoke:<profile>` | Model server or connection | The `doctor` lines in the detail; the server log |
| `residency:*` | Swapping local models | The Ollama and llama.cpp logs; `ollama ps`; `curl localhost:9931/models` |
| `benchmark:<name>` with errors over 0% | Server under load, budgets, or the harness | Failed items, their transcripts, telemetry |
| `benchmark:<name>` with low accuracy and 0% errors | The model's answers | Transcripts: expected vs predicted |
| `gate:<name>` | A change against the baseline | Gate lines; the baseline vs the new `report.json` |

## Where the evidence is

Each pipeline run has its own folder, `validation-results/runs/<YYYY-MM-DD_HH-MM-SS>/`,
written `<run>` below. `validation-results/latest` (and `latest.txt`) points to the
newest run. Earlier runs are kept, so you can compare a failing run with the last good
one.

| Evidence | Location | What it holds |
|---|---|---|
| Run history | `validation-results/index.html` | Every run, newest first, with scores per benchmark and the change from the previous run |
| HTML report | `<run>/index.html` | Everything below for that run, summarized, with links |
| Stage results | `<run>/pipeline-summary.json` | Every stage, its status, and its detail |
| Benchmark items | `<run>/<name>/results.jsonl` | One line per item: correct, expected, predicted, status, error, latency, tokens. Written live. |
| **Transcripts** | `<run>/<name>/transcripts/<item>.json` | The **full run** of every failed item: messages, tool calls and results, reasoning, events, and the error. Use `--transcripts all` to keep passing items too. |
| Benchmark report | `<run>/<name>/report.json` | Accuracy with a 95% CI, error rate, per-category scores, latency, tokens, cost, dataset hash |
| Telemetry | `<run>/telemetry.jsonl` (or the file given to `--telemetry-file`) | One JSON line per model call (profile, latency, tokens, cached tokens, `finish_reason`, `error_category`) and per tool call (tool, status, latency). Never message text. |
| Scenario results | `<run>/scenarios.json` | Per scenario: outcome and failure reasons |
| Ollama log | `~/.ollama/logs/server.log` | Model loads, out-of-memory errors, request errors |
| llama.cpp router log | `/tmp/llama-server.log` (the `--log-file` of `llama serve`) | Loads and unloads, `Compute error`, Metal out of memory |
| Modal log | `modal app logs <app>` | Image build, cold start, vLLM startup errors, requests |
| A run from `agent-harness run`, `serve`, or `/ui` | The SQLite database (`--database`, default `agent-harness.sqlite3` in the directory the command started in) | `agent-harness status RUN_ID`, `agent-harness runs --status failed`, and the event log (below) |
| Messages, tool calls, and the model's reasoning | `runs.snapshot_json` in the same database (`messages[i].content`, `.tool_calls`, `.reasoning`); `GET /runs/{id}/messages?reasoning=true` | The text the events leave out. Reasoning is stored even with `reasoning_history = "drop"`, which only keeps it out of later requests |
| Tool calls as the hooks saw them | `.agent-harness/hook-audit.jsonl` (root config's audit hook) | Every tool call with its arguments and decision |

The event log of a stored run is the most detailed record of what the harness did:

```bash
sqlite3 agent-harness.sqlite3 \
  "select step, type, payload_json from run_events where run_id = 'RUN_ID' order by sequence"
```

Over HTTP, use `GET /runs/{id}/events`. For a chat, use `GET /runs/{id}/messages`.

## Watching a run live

```bash
agent-harness-bench pipeline examples/validation.local.toml
# progress lines go to stderr; each result is printed as soon as it is known
# at the end it prints the run's report and the history page
```

In other terminals:

```bash
tail -f validation-results/latest/<benchmark>/results.jsonl   # items as they finish
tail -f validation-results/latest/telemetry.jsonl             # every model and tool call
tail -f ~/.ollama/logs/server.log   # or /tmp/llama-server.log, or: modal app logs <app>
```

## Smoke checks (`doctor`)

Rerun one profile directly. Add `--strict` to treat warnings as failures:

```bash
agent-harness doctor --config examples/models.toml --model-profile llamacpp --skip-cache
```

| Line | Meaning | Fix |
|---|---|---|
| `[FAIL] credentials` | A variable named by the profile (`api_key_env`, `headers_env`) is not set | Load `.env` into **this** shell (`set -a && source .env && set +a`) |
| `[FAIL] health: … unreachable: Connection refused` | Nothing is listening on that host and port | Start the server; check the endpoint and port in the profile or `.env` |
| `[FAIL] health: … CERTIFICATE_VERIFY_FAILED` | The Python running `agent-harness` has no CA certificates | `which agent-harness` must point into this repo's `.venv`. If the venv was copied or moved, recreate it. For python.org builds, run `Install Certificates.command`. |
| `[FAIL] health: … /v2/models responded 404` | The endpoint path is wrong | Endpoints end in `/v1`, the OpenAI API path. `v1`/`v2` in `modal app history` are deployment versions, not paths. |
| `[FAIL] health: … responded 401` | The server's bearer key is wrong | `MODAL_LLM_API_KEY` must equal `VLLM_API_KEY` in the Modal secret |
| `[FAIL] health: … responded 403` | Modal proxy auth was rejected | Check `MODAL_PROXY_TOKEN_ID` and `MODAL_PROXY_TOKEN_SECRET` |
| `[FAIL] model: 'X' is not served; available: …` | The server, or a llama.cpp router, does not have that model | Fix `model` in the profile, or `ollama pull X` |
| `[FAIL] completion: transient model provider error (500): Compute error` | llama.cpp ran out of GPU memory (`Insufficient Memory` in its log) | Unload other large models (`ollama stop …`), or turn on `[residency]`. Then **restart the router**: a failed instance stays in an error state. |
| `[FAIL] completion: … timed out` (Modal) | Cold start: the image is pulled and the model loaded | Run again; watch `modal app logs`; raise `cold_start_timeout_seconds` |
| `[WARN] completion: ran out of tokens (16)` | A thinking model (Qwen3, DeepSeek-R1, gpt-oss) reasons before answering | `--max-tokens 512`, or `max_tokens = 512` on the `[[smoke]]` entry |
| `[WARN] tool_call: tool call came back as plain text` | The server's tool-call parser is off | llama.cpp: `--jinja`. vLLM: `--enable-auto-tool-choice --tool-call-parser <family>`. SGLang: `--tool-call-parser`. |
| `[WARN] tool_call: model did not call the tool` | The model ignores tools, or ran out of tokens | Try `--max-tokens 512`; otherwise the model may not support tools |
| `[WARN] prefix_cache: server did not report cached tokens` | Cache reporting is off | vLLM: `--enable-prompt-tokens-details`. SGLang: `--enable-cache-report`. Ollama never reports it. |

## Benchmark items

Open the failed item's transcript (the HTML report links it) and read it in this order:

1. **`status` and `error` at the top.** `failed` with an error means the run broke; read
   the error first. `completed` means the model answered, so compare the answers.
2. **Events, in order.** The sequence tells you what happened:

   | You see | Means |
   |---|---|
   | `model_requested` → `model_responded` with `ok: false, retryable: true`, repeated | Server errors or timeouts; look at the server log at that time |
   | `model_fallback` | The profile gave up and the run switched to its `fallback` |
| `model_responded` with `provider_attempts: 2`, `retried_errors: ["timeout"]` | The adapter timed out and retried; the call's latency includes the lost attempt |
| `finish_reason: "length"` | The reply hit the output cap (`max_tokens`); raise it if the answers are legitimately long |
| Tool results `Refused: your reply made N tool calls` | The model emitted more than `max_tool_calls_per_turn` calls in one reply (a runaway) |
   | `tool_requested` → `tool_responded` with a status other than `success` | A tool failed; the tool message in `messages` says why |
   | `budget_exhausted` | Steps, tokens, time, or cost ran out; raise `max_steps` / `max_wall_seconds` on the benchmark |
   | `output_rejected` | The reply did not match `output_schema`; the next request asks again |
   | `guardrail_refused` | Guardrails blocked the input; the `guardrail_decided` events show which detector |
   | `approval_requested`, and nothing after | A tool needed approval; benchmarks have nobody to approve |
3. **Messages.** What the model was asked, which tools it called with which arguments,
   what came back, its reasoning (if returned), and its final answer.

**Which layer is it?**

| Pattern | Layer |
|---|---|
| Many items `error`/`failed` with the same `error_category` (`timeout`, `provider`) | **Server or environment.** Check the server log; resume reruns them once fixed. |
| Errors only under `concurrency > 1` | **Server capacity.** Lower `concurrency`, or set the profile's `max_concurrency` |
| A harness exception in `error` (`harness raised …`, a traceback) | **Harness bug.** Reproduce with one item, then report it |
| `completed`, but predicted ≠ expected | **The model.** Is the answer wrong, or the extraction? Read the final message. |
| A tool call with the wrong arguments, or no tool call where one was needed | **The model, or the server's tool parser.** Compare with another backend on the same item. |

**Reproduce one item** with everything kept:

```bash
agent-harness-bench run --suite gsm8k --data examples/bench/gsm8k_sample.jsonl \
  --config examples/models.toml --model-profile llamacpp --offset 2 --limit 1 \
  --transcripts all --telemetry-file /tmp/one.jsonl --out /tmp/one-item
```

You can also send the item's prompt (from the transcript's first user message) to a
run you can inspect afterwards:

```bash
agent-harness run --config examples/models.toml --model-profile llamacpp --database /tmp/debug.sqlite3 \
  --show-reasoning --telemetry-file /tmp/debug.jsonl "<the prompt>"
agent-harness status --config examples/models.toml --database /tmp/debug.sqlite3 RUN_ID
```

Each pipeline run starts benchmarks fresh, in a new run folder. `pipeline --resume`
continues the latest run's folder instead, re-running only items that ended in `error`
or `failed`.
`agent-harness-bench run` always resumes from its `--out` directory; delete that
directory to start over.

## Gates

| Gate line | Meaning |
|---|---|
| `max_error_rate: 0.333 > 0.100` | Too many items errored. This is an environment or server problem, not the model's accuracy. Fix the errors first. |
| `max_accuracy_drop: … (drop -0.333 …)` | Accuracy fell compared with the baseline. With n=3, one item is 33 points: check the 95% CI in the report before trusting it. |
| `same_dataset: … differ` | The dataset file or the selected items changed, so the comparison would be meaningless. Use the same file, `offset`, and `limit`, or record a new baseline. |
| `(no baseline yet)` | Only the absolute limits were checked. `--update-baselines` records one. |

When a change is **intended** (a new model, quantization, or prompt), record new
baselines with `--update-baselines`. A baseline is written only when its gate passes or
none exists. A baseline recorded from a broken run must be deleted by hand.

## Known issues and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| llama.cpp `500 Compute error` right after an Ollama step | Two large models in GPU memory (Metal out of memory) | `[residency] enabled = true` in the plan; restart the router once |
| Every llama.cpp request fails after one out-of-memory error | The router keeps the broken instance | Restart the Llama app, or wait out `--sleep-idle-seconds` |
| `unit` fails only inside the pipeline, with many errors | `AGENT_HARNESS_*` variables leaking into tests (fixed: the unit stage now runs isolated) | Update to the current `main` |
| Modal image build: `CUDA_HOME is not set` | pip fell back to a vLLM release that must be compiled | Keep `VLLM_PACKAGE` pinned in `deploy/modal/vllm_server.py` |
| vLLM: `Could not find nvcc` at startup | FlashInfer sampler needs the CUDA toolkit | `VLLM_USE_FLASHINFER_SAMPLER=0` (already in the script) |
| `CERTIFICATE_VERIFY_FAILED` only for Modal | Wrong Python on `PATH` (a moved venv) | Recreate `.venv`; `which agent-harness` |
| One call takes about twice `timeout_seconds` (e.g. 124 s with 120 s); the report shows `retried timeout` | The first attempt timed out and the retry succeeded. With no output cap the model may be running away | Keep `max_tokens` in the profile's `parameters`; check the server log for long generations |
| A run fails with `too many tool calls in one reply`, or vLLM keeps generating after the harness gave up | A runaway reply; Modal's proxy does not cancel abandoned vLLM requests | Keep `max_tokens` set; `modal app stop <app>` stops requests still generating |
| Results ignore a TOML edit | An `AGENT_HARNESS_MODEL__*` or `AGENT_HARNESS_SET__*` variable overrides it | Environment wins over the file: `agent-harness config show` names the variable; comment it out in `.env` |
| `model_responded` with `context_overflow: true`, then a shorter view | The server said the request was too long; the step was retried under a smaller cap | Expected. If it happens often, set `context_window` to the server's real context, or enable `[summarization]` |
| A run fails with `model context exceeded` | The overflow retries (`[orchestrator] context_overflow_retries`) were used up, or the pinned head alone is too large | Raise the server's context, shorten the system prompt or first message, or raise the retries |
| A server's "too long" error fails the run instead of being retried | Its message is not one the adapters recognise | Add a regex to `[orchestrator] context_overflow_patterns` |
| Frequent `context_overflow` or `context_capped` events on one profile | The estimate undercounts this content (code, JSON, other languages) | Lower `[context] chars_per_token` (about 2.5), or set `token_counting = "server"` on the profile (llama.cpp, vLLM) |
| `adapter ... cannot count tokens on the server` | `token_counting = "server"` on Ollama or SGLang | Use `"estimate"` there |
| `memory_search` finds nothing that obviously matches | Keyword search; or notes saved before semantic search was enabled | `[memory] search = "hybrid"` with an `embedding_profile`, then `agent-harness memory reindex` |
| `web_fetch` refuses a URL as private | The host resolves to a private, loopback, or link-local address | Intended (SSRF protection); set `[web_fetch] allow_private = true` only for trusted internal pages |
| Judge accuracy looks too high | The model is grading its own answers | Set `judge_profile` to a different, stronger profile |
| API clients get `429` with `Retry-After` | `[api] rate_limit_per_minute` (per key; per address after failed authentication) | Raise the limit or `rate_limit_burst`; with several API processes each counts separately |
| `429` naming `user '...'` | That end user's `[user_quotas]` limit for the last 24 hours | Raise it, or add `[user_quotas.users.NAME]` for that user |
| A paused run was denied or ended by itself; `approval_resolved` has `actor: "timeout"` | `[interaction] approval_timeout_seconds` / `question_timeout_seconds` expired, applied by `on_timeout` | Raise the timeout, or set it to 0 to wait forever |
| Context errors or silent truncation on llama.cpp | `context_window` is larger than the server's `ctx-size` | Keep them equal |
| Benchmark fails at 100% errors, and resume "fixes nothing" | Older versions reused errored items | Current versions rerun them; delete the bad baseline |
| Benchmarks print `resuming: … already has results` and finish instantly; telemetry is empty | Older versions of the pipeline reused earlier answers | Current versions use a new run folder each time, unless `--resume` is given |
| A chat asks to approve `ls` or `find`, then each one fails with `command 'ls' is not allowlisted` | The allowlist is checked when the command runs, after approval; `ls`/`find` were not in `allowed_commands` | Fixed in the root config: they are allowlisted and pre-approved. For your own config, add read-only commands to `[coding] allowed_commands` (repeating the defaults) and to `[permissions] allow` |
| `find ... is forbidden: find may only list files` | `-delete`, `-exec`, `-execdir`, `-ok`, `-okdir`, `-fls`, `-fprint*` would change the host's files through the read-write mount | Intended. Use `find` to list, and other tools to read or change files |
| A plain chat question starts with `run_command` or reads the workspace | `[coding] enabled = true` gives every run (including `/ui`) the coding prompt and tools | Expected for now; `AGENT_HARNESS_SET__CODING__ENABLED=false` for a chat-only server |
| The model explores `.agent-harness/` when asked about "agent-harness" | Older versions put the instruction file's full path in the system prompt header | Update: headers now name the file only (`## Instructions from CHAT.md`) |
| `Tool execution failed.` for `read_file`, or `Tool returned an invalid result.` for a `cwd` | Older versions did not translate missing files, bad directories, binary or unreadable files | Update: the message now says `no such file or directory: X; search_files can find the right path` (and similar) |
| Every `search_files` call fails | Older versions stopped at the first file they could not read | Update: unreadable files are skipped |
| In a chat's second turn, the model answers the first question again | Older versions kept only the first user message verbatim; a summary turned the new question into a passing remark | Update: the newest request follows the summary as `[Latest user message - answer this one]` |
| Subagents fail with `Context size has been exceeded` (llama.cpp, retryable 500) | Concurrent children share one llama.cpp KV cache (`ctx-size`) | `[subagents] max_parallel = 2` (root config), or raise `ctx-size` |
| A run failed with `MalformedResponseError` about tool-call arguments | Older versions failed the run when the model wrote broken JSON arguments | Update: the call gets a validation error and the model can retry |
| A run hits `max_steps` or the time limit with no answer | `final_answer_on_last_step` is off | `[orchestrator] final_answer_on_last_step = true` (root config); `final_answer_reserve_seconds` (120) sets how early the time-limit answer comes |
| The model re-reads files it already read, right after a `context_summarized` event | The summary replaced the file contents in the view, so the model fetches them again | Known. A larger input budget (smaller `parameters.max_tokens`, larger `context_window`) makes summaries rarer |
| `agent-harness-bench run --option repeats=3` fails with `options.repeats must be a positive integer` | `--option` values are passed as strings | Known bug. Put the options in a `[[benchmark]]` entry of a plan TOML and run `pipeline PLAN --stages benchmarks` |
| `guardrail_decided` with `action: flag` and `violent_crimes` on ordinary coding chats | Llama Guard 3 1B misreads code and repository text | Expected with `action = "flag"` (no effect on the run); the 8B model does not do this |

## Reporting a harness bug

Include:
- the failing stage line;
- the item's transcript;
- the matching telemetry lines (`grep RUN_ID telemetry.jsonl`);
- `git rev-parse HEAD`;
- the profile, with any keys removed.

Check whether the offline suite passes:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

If it does, the problem is most likely in the environment or a server, not in the
harness.
