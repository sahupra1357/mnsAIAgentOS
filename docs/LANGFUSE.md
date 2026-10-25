# Langfuse: traces, feedback, and evaluations

The harness sends its runs to [Langfuse](https://langfuse.com) (cloud or self-hosted) with the
standard library only: no SDK, no OpenTelemetry packages.

| What | How | Where in Langfuse |
|---|---|---|
| Every turn of every top-level run | OTLP/HTTP JSON to `/api/public/otel/v1/traces`, header `x-langfuse-ingestion-version: 4` (real time) | Tracing → one trace per turn, session = the run |
| 👍/👎 in `/ui` | `POST /runs/{id}/feedback` → `user_feedback` score on that turn's root observation | Scores, and on the trace |
| Benchmark results | `agent-harness-bench run ... --langfuse` | Datasets → `agent-harness/<suite>/<file>` → Experiments |
| LLM-as-a-judge on live traffic | configured in the Langfuse UI (below) | Evaluators → rules on root observations |

Langfuse v4 note (checked 2026-10-11): organizations created on or after 2026-09-16 run v4. Its
trace ingestion is OpenTelemetry; the older `/api/public/ingestion` trace events are deprecated,
delayed by up to 15 minutes, and sunset on Langfuse Cloud on 2026-11-16. Score writes
(`score-create`) and the dataset item API stay. Dataset runs became experiments, created with
experiment attributes on spans. The harness uses only the v4 paths.

## Setup

1. Create a Langfuse project; copy its public and secret keys.
2. In `.env` (never in `models.toml`):
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # or https://cloud.langfuse.com (EU), or your host
   ```
   (`LANGFUSE_HOST` is read when `LANGFUSE_BASE_URL` is not set.)
3. The root `models.toml` already has it on:
   ```toml
   [langfuse]
   enabled = true
   content = "redacted"   # "none" | "redacted" | "full"
   sample_rate = 1.0      # share of runs sent, the same runs every time
   environment = "local"
   # max_text_chars = 8000, flush_seconds = 2.0, timeout_seconds = 10.0
   # base_url_env / public_key_env / secret_key_env name other variables
   ```
4. Restart `serve` (`.env` is read at start). Without the keys nothing is sent and the log says
   so once; runs are never affected.

## What a trace contains

One trace per turn (the first request, then each `continue`), IDs derived from the run ID and
turn number. Every span carries the trace-wide attributes (trace name, session = run ID, user,
tags = profile, environment, release, metadata `run_id`, `turn`, `status`, `model_profile`,
`tenant`), as Langfuse v4 expects.

```
agent run / agent turn        (agent, the root: user message in, answer out)
├─ model modal_shared         (generation: model, tokens, cost, finish reason, new messages, reply)
├─ guard modal_shared.guard   (generation: guard check tokens; embedding: memory tool embeddings)
├─ tool read_file             (tool: arguments, result, status, risk, permission)
├─ guardrail_decided          (guardrail: category, action, detector; never the matched text)
├─ approval_requested / ...   (event)
├─ agent researcher           (agent: the subagent's task and answer)
│  ├─ model ...               (the child run's own generations and tools)
│  └─ tool ...
└─ model modal_shared
```

- **Content.** `redacted` masks secrets (the `[redaction]` patterns plus built-ins) and PII (the
  guardrail detectors: email, phone, card, SSN, IBAN), then truncates each text to
  `max_text_chars`. `none` keeps only timings, tokens, cost, and metadata.
- **When.** A turn is sent when it ends (`run_completed`, `run_failed`, `run_cancelled`,
  `budget_exhausted`), from the run store, on a background thread. A paused run (approval,
  question) is sent once it finishes.
- **Once.** Langfuse v4 keeps every copy of a re-sent span ID, so the exporter never sends a span
  twice. A turn that fails and is then resumed keeps the first export.
- **Cost.** Generations carry the harness's estimated cost (`cost_details.total`); Langfuse shows
  it as is.
- **Guard and embeddings.** Each guard check (`guard <profile>`) and each embedding (`embedding`) is a
  generation with its own tokens, so a trace's total includes them (from `usage_recorded` events).

## User feedback (`/ui`)

With `[langfuse]` on and keys set, `GET /profiles` reports `"feedback": true` and `/ui` shows
👍/👎 under the last answer of each turn. A vote sends a `user_feedback` boolean score (1 or 0)
to that turn's root observation; voting again the same day replaces it.

```bash
curl -X POST -H "Authorization: Bearer $KEY" localhost:8765/runs/RUN_ID/feedback \
     -d '{"value": "up", "message_index": 7, "comment": "spot on"}'
```

`message_index` (the `index` from `GET /runs/{id}/messages`) picks the turn; without it the
newest finished turn is rated. Comments go through the content policy.

## Benchmarks as experiments

```bash
agent-harness-bench run --suite tasks --data examples/bench/tasks_sample.jsonl --config models.toml \
    --model-profile modal_shared --max-steps 40 --max-wall-seconds 900 \
    --langfuse --langfuse-run "modal_shared tasks 2026-10-11"
```

Or `langfuse = true` (and optionally `langfuse_run = "..."`) on a `[[benchmark]]` in a plan.

- **Dataset** `agent-harness/<suite>/<data file>`: every selected item is upserted (stable IDs).
- **Experiment** `--langfuse-run` (default `<profile> <UTC time>`): one trace per item, environment
  `experiment`, every span tagged with the experiment ID and name, dataset ID, item ID, and root
  observation; the root holds the item input, the answer, and the expected output (also as
  `metadata.expected_output`, for Langfuse judges).
- **Scores** on the root observation: `correct`, plus each boolean the suite reports
  (`answer_ok`, `behavior_ok` for `tasks`), with the error or judge verdict as the comment.
- Run the same dataset with another profile (or after a change) and compare the experiments
  side by side in Langfuse. The local `report.json` gains a `langfuse` summary.

## LLM-as-a-judge in Langfuse (set up in its UI)

Langfuse runs these evaluators itself; the harness only has to send observations with input and
output, which the root span does.

1. **LLM connection** (Settings → LLM Connections): any OpenAI-compatible endpoint works, for
   example the Modal shared endpoint (base URL `https://<workspace>--ep-...modal.direct/v1`, key =
   your `MODAL_SHARED_TOKEN`), or an OpenAI/Anthropic key. The judge model is billed to that key.
2. **Evaluator** (Evaluation → LLM-as-a-Judge → new): pick a template (helpfulness, correctness,
   hallucination, toxicity) or write a prompt with `{{input}}` and `{{output}}`; map `input` to
   the observation's input and `output` to its output.
3. **Rule for live traffic**: filter **Is Root Observation** = true (our `agent run` / `agent turn`
   spans), optionally environment = `local` and a tag such as a profile name; set a sampling rate.
   Scores land on the root observation, next to `user_feedback`.
4. **For benchmark experiments**: Langfuse offers its Expected Output variable only for experiments
   started in its UI, not for ones sent over OpenTelemetry like ours. The harness therefore also puts
   each item's expected output in the root observation's metadata. Make a second evaluator with
   `{{ground_truth}}` mapped to **Metadata → `expected_output`** (plus `{{input}}` → Input,
   `{{output}}` → Output), and a rule filtered on `isRootObservation:true environment:experiment`.
   Benchmark runs from before this change lack the metadata; run the benchmark again with a new
   `--langfuse-run` name.

Trace-level evaluators are deprecated in v4; use observation-level rules as above.

## Reading data back

Langfuse v4 reads go through `GET /api/public/v2/observations` (with `fromStartTime` and
`toStartTime`; add `fields=core,io,metadata` for input, output, and metadata). The legacy
`GET /api/public/traces/{id}` answers 410 for new organizations.

## Troubleshooting

See the Langfuse rows in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
