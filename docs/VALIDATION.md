# Validating the Agent Harness

Validation runs in six stages. The first three run offline on every change. The last
three need your model servers (Ollama, llama.cpp, Modal) and public datasets that you
download yourself.

| # | Stage | What it proves | Needs a model? | Where it runs |
|---|-------|----------------|----------------|---------------|
| 1 | **static** | Code is well-formed: lint (`ruff`), types (`mypy`), compile | no | CI + pipeline |
| 2 | **unit** | Every component and failure path behaves (fakes only) | no | CI + pipeline |
| 3 | **scenarios** | The real orchestrator handles scripted situations: tool use, unknown tools, transient errors | no | CI + pipeline |
| 4 | **smoke** | Each profile's server is reachable, serves the model, completes, and returns structured tool calls (`doctor`) | yes | pipeline |
| 5 | **benchmarks** | Capability on public datasets, per profile | yes | pipeline / manual |
| 6 | **gates** | No regression against saved baselines; absolute floors | reports only | pipeline |

Stages 1–3 run in GitHub Actions on every push and PR (`.github/workflows/ci.yml`).
Stages 4–6 run on a machine that can reach your models:

```bash
pip install -e .            # provides agent-harness and agent-harness-bench
agent-harness-bench pipeline examples/validation.toml                          # all stages
agent-harness-bench pipeline examples/validation.toml --stages smoke,benchmarks,gates
```

## Benchmark suites

Each suite exercises a different layer of the harness:

| Suite | Harness layer validated | Public dataset | Record format |
|-------|-------------------------|----------------|---------------|
| `gsm8k` | Full agent loop: model ↔ calculator tool, argument validation and recovery, step budgets, retries | GSM8K test split (1,319 problems) | `{"question", "answer": "... #### 42"}` |
| `bfcl` | Model gateway and adapters: does each provider return correct **structured** tool calls (simple, multiple, parallel, irrelevance)? | Berkeley Function Calling Leaderboard | question file + possible-answer file, joined on `id` |
| `humaneval` | Coding path: generation, then execution in the Docker sandbox (no network) | HumanEval (164 problems) | `{"task_id", "prompt", "test", "entry_point"}` |
| `qa` | Plain answering, with or without tools; field names are configurable | NQ-open, TriviaQA, or any Q&A JSONL | `{"question", "answer" \| "answers"}` |

List the suites with `agent-harness-bench suites`.

### Getting the datasets

The harness never downloads data. Fetch the datasets yourself and point `--data` at
the files. JSON Lines, JSON arrays, and `.gz` files are supported. Upstream file names
and locations change between releases, so check each project's README.

- **GSM8K:** the `openai/grade-school-math` GitHub repository, file
  `grade_school_math/data/test.jsonl`. The Hugging Face `openai/gsm8k` dataset has
  the same records as Parquet.
- **BFCL:** the `ShishirPatil/gorilla` repository, directory
  `berkeley-function-call-leaderboard/`.
  - Use the category files (e.g. `BFCL_v3_simple.json`, `..._multiple.json`,
    `..._parallel.json`, `..._irrelevance.json`).
  - Pass the matching `possible_answer/...` file as `--answers`. Irrelevance
    categories need no answers.
  - Only the first turn is evaluated; multi-turn categories are out of scope.
- **HumanEval:** the `openai/human-eval` repository, file `data/HumanEval.jsonl.gz`.
- **Q&A:** for example NQ-open (`{"question", "answer": [...]}`), or TriviaQA
  converted to `{"question", "answers": [...]}`.
  - Other layouts work with `--option question_field=... --option answer_field=...`.

Convert Parquet to JSONL first:

```bash
python -c "import pandas as p; p.read_parquet('test.parquet').to_json('test.jsonl', orient='records', lines=True)"
```

### Running one benchmark

```bash
agent-harness-bench run --suite gsm8k --data datasets/gsm8k/test.jsonl \
  --config models.toml --model-profile ollama --limit 200 --concurrency 2 \
  --out results/gsm8k-ollama

agent-harness-bench run --suite bfcl --data datasets/bfcl/BFCL_v3_simple.json \
  --answers datasets/bfcl/possible_answer/BFCL_v3_simple.json \
  --config models.toml --model-profile modal --out results/bfcl-simple-modal

agent-harness-bench run --suite humaneval --data datasets/HumanEval.jsonl.gz \
  --config models.toml --model-profile llamacpp --out results/humaneval-llamacpp
```

Each run writes two files to `--out`:
- `results.jsonl`: one line per item with the expected and predicted answers, status,
  steps, tokens, cost, latency, and error.
- `report.json`: accuracy with a 95% Wilson interval, error rate, status counts,
  per-category accuracy, latency (mean and p95), steps, tokens, cost, a SHA-256 of
  the dataset, and the selected items.

**Runs are resumable.** Re-running with the same `--out` skips items already in
`results.jsonl`.

**HumanEval runs model-written code.** It executes in Docker by default: no network,
all capabilities dropped, read-only root, and CPU, memory, and PID limits.
`--unsafe-local-execution` runs it on your host; use that only for trusted,
throwaway experiments.

## Baselines and regression gates

1. Run the pipeline once and record baselines:

   ```bash
   agent-harness-bench pipeline validation.toml --update-baselines
   ```

2. After any change (code, prompts, model, quantization, server flags), run it again.
   A gate fails if any of these hold:
   - accuracy drops more than `max_accuracy_drop`;
   - accuracy is below `min_accuracy`;
   - the error rate exceeds `max_error_rate`;
   - latency or cost grew beyond `max_latency_increase` / `max_cost_increase`
     (optional);
   - the dataset file or item selection differs from the baseline, so the comparison
     would be meaningless.

3. Gate a single report by hand:

   ```bash
   agent-harness-bench compare results/gsm8k-ollama/report.json --baseline baselines/gsm8k-ollama.json
   ```

`--update-baselines` replaces a baseline only when its gate passes, or when no
baseline exists yet. The command exits with 1 on any failed stage or gate, so it can
be used in scripts and scheduled jobs.

## Recommended validation matrix

Run each suite against each model source. Each suite catches different problems:

| Profile | smoke | bfcl | gsm8k | humaneval | Typical findings |
|---------|-------|------|-------|-----------|------------------|
| `ollama` | ✔ | simple, parallel, irrelevance | 200 | optional | Missing tool-call IDs; small default context (set `OLLAMA_CONTEXT_LENGTH` or `api_mode = "native"`) |
| `llamacpp` | ✔ | simple, parallel, irrelevance | 200 | ✔ | Server started without `--jinja`: tool calls come back as text (BFCL near 0) |
| `modal` (vLLM/SGLang) | ✔ | all categories | full | full | Missing `--tool-call-parser`; cold starts (latency p95); prefix-cache hits (`doctor` without `--skip-cache`) |

## Reading the numbers

- **Small samples are noisy.** With n = 3 the 95% interval can be 0.44–1.0. Use at
  least 200 GSM8K items, full BFCL categories, and all 164 HumanEval problems before
  comparing models or accepting a regression.
- **Scores are the harness's own approximations.** The BFCL checker is a simplified AST
  match. HumanEval is scored pass@1 with one sample. For leaderboard-comparable
  numbers, feed the saved predictions in `results.jsonl` to each benchmark's official
  evaluator.
- **Errors are not wrong answers.** `error_rate` counts runs that failed (timeouts,
  exhausted retries, provider errors). A high error rate points at infrastructure, not
  model quality. Check `status_counts` and the `error` field.
- Add `--telemetry-file` to CLI runs, or read the run events, to see retries,
  fallbacks, and context trimming per item.

## Example: first live check on a local Ollama

```text
$ AGENT_HARNESS_MODEL__OLLAMA__MODEL=gpt-oss:20b agent-harness doctor --config examples/models.toml --model-profile ollama
[  OK] health, model, completion (6.0s), tool_call: structured tool call parsed
$ agent-harness-bench run --suite gsm8k --data examples/bench/gsm8k_sample.jsonl ...
gsm8k on ollama (gpt-oss:20b): accuracy 1.000 [95% CI 0.439-1.000] n=3 errors=0.0%
$ agent-harness-bench run --suite bfcl --data examples/bench/bfcl_sample.json ...
bfcl on ollama (gpt-oss:20b): accuracy 0.667 — parallel: 0.000 (one call instead of two)
```

In that run, one GSM8K item also shows the recovery path working. The model first
called the calculator with `"operation": "-"`, the harness rejected it with a schema
error, and the model retried with `"subtract"`. The count is in `details.tool_errors`.

The files in `examples/bench/` are small **synthetic** samples in each public format.
Use them for smoke tests and for CI, not for measuring quality.
