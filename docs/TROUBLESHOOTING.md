# Troubleshooting

How to find out *where* a problem is (your environment, a model server, the harness,
or the model itself) and fix it. Most entries come from real validation runs.

## The debugging loop

1. **Read the report.** Open `validation-results/index.html`. It shows which stage
   failed, which benchmark items failed, and a summary of model and tool calls.
2. **Find the layer.** Use the table in the next section. The right layer is usually
   obvious from the first failing stage.
3. **Open the evidence** for that layer. See "Where the evidence is".
4. **Reproduce narrowly.** Rerun one stage, one profile, or one benchmark item, rather
   than the whole pipeline.
5. **Fix, then verify** with the same narrow command, and finally the full pipeline.

| First failing stage | Layer | Look at |
|---|---|---|
| `static`, `unit` | Harness code or your Python environment | The stage detail line; run the command yourself |
| `scenarios` | Harness behaviour (scripted models, no network) | `validation-results/scenarios.json` |
| `smoke:<profile>` | Model server or connection | The `doctor` lines in the detail; the server log |
| `residency:*` | Swapping local models | The Ollama and llama.cpp logs; `ollama ps`; `curl localhost:9931/models` |
| `benchmark:<name>` with errors over 0% | Server under load, budgets, or the harness | Failed items, their transcripts, telemetry |
| `benchmark:<name>` with low accuracy and 0% errors | The model's answers | Transcripts: expected vs predicted |
| `gate:<name>` | A change against the baseline | Gate lines; the baseline vs the new `report.json` |

## Where the evidence is

| Evidence | Location | What it holds |
|---|---|---|
| HTML report | `validation-results/index.html` | Everything below, summarized, with links |
| Stage results | `validation-results/pipeline-summary.json` | Every stage, its status, and its detail |
| Benchmark items | `validation-results/<name>/results.jsonl` | One line per item: correct, expected, predicted, status, error, latency, tokens. Written live. |
| **Transcripts** | `validation-results/<name>/transcripts/<item>.json` | The **full run** of every failed item: messages, tool calls and results, reasoning, events, and the error. Use `--transcripts all` to keep passing items too. |
| Benchmark report | `validation-results/<name>/report.json` | Accuracy with a 95% CI, error rate, per-category scores, latency, tokens, cost, dataset hash |
| Telemetry | The file given to `--telemetry-file` | One JSON line per model call (profile, latency, tokens, cached tokens, `finish_reason`, `error_category`) and per tool call (tool, status, latency). Never message text. |
| Scenario results | `validation-results/scenarios.json` | Per scenario: outcome and failure reasons |
| Ollama log | `~/.ollama/logs/server.log` | Model loads, out-of-memory errors, request errors |
| llama.cpp router log | `/tmp/llama-server.log` (the `--log-file` of `llama serve`) | Loads and unloads, `Compute error`, Metal out of memory |
| Modal log | `modal app logs <app>` | Image build, cold start, vLLM startup errors, requests |
| A run from `agent-harness run` | The SQLite database (`--database`, default `agent-harness.sqlite3`) | `agent-harness status RUN_ID`, `agent-harness runs --status failed`, and the event log (below) |

The event log of a stored run is the most detailed record of what the harness did:

```bash
sqlite3 agent-harness.sqlite3 \
  "select step, type, payload_json from run_events where run_id = 'RUN_ID' order by sequence"
```

Over HTTP, use `GET /runs/{id}/events`. For a chat, use `GET /runs/{id}/messages`.

## Watching a run live

```bash
agent-harness-bench pipeline examples/validation.local.toml --telemetry-file validation-results/telemetry.jsonl
# progress lines go to stderr, and each result is printed as soon as it is known
```

In other terminals:

```bash
tail -f validation-results/<benchmark>/results.jsonl          # items as they finish
tail -f validation-results/telemetry.jsonl                    # every model and tool call
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

Each pipeline run starts benchmarks fresh. `pipeline --resume` continues an interrupted
run instead, re-running only items that ended in `error` or `failed`.
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
| Modal image build: `CUDA_HOME is not set` | pip fell back to a vLLM release that must be compiled | Keep `VLLM_PACKAGE` pinned in `examples/modal/vllm_server.py` |
| vLLM: `Could not find nvcc` at startup | FlashInfer sampler needs the CUDA toolkit | `VLLM_USE_FLASHINFER_SAMPLER=0` (already in the script) |
| `CERTIFICATE_VERIFY_FAILED` only for Modal | Wrong Python on `PATH` (a moved venv) | Recreate `.venv`; `which agent-harness` |
| Results ignore a TOML edit | An `AGENT_HARNESS_MODEL__*` variable overrides it | Environment wins over the file: comment it out in `.env` |
| Context errors or silent truncation on llama.cpp | `context_window` is larger than the server's `ctx-size` | Keep them equal |
| Benchmark fails at 100% errors, and resume "fixes nothing" | Older versions reused errored items | Current versions rerun them; delete the bad baseline |
| Benchmarks print `resuming: … already has results` and finish instantly; telemetry is empty | Older versions of the pipeline reused earlier answers | Current versions start fresh unless `--resume` is given |

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
