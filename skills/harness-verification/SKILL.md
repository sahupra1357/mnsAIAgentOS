---
name: harness-verification
description: Verify this agent harness with deterministic tests, scenario evaluations, and release checks. Use when adding tests, evaluating behavior, reviewing integration risk, or preparing a runnable handoff.
---

# Harness Verification

Verify observable behavior at contracts and state transitions. Tests must be deterministic, network-free, and safe to rerun.

## Test Layers

- Unit tests cover state transitions, configuration parsing, adapters with fake transports, schema validation, and policy decisions.
- Integration tests run a complete agent loop with an in-memory or temporary SQLite store, scripted model responses, and fake tools.
- Scenario evaluations cover success, tool chaining, malformed output, transient failure, permanent failure, budget exhaustion, cancellation, and recovery from a saved checkpoint.

Assert durable outcomes and emitted events rather than internal call order unless ordering is itself a contract.

## Release Checks

- No network or real credentials are required by the test suite.
- Examples use placeholder credentials and endpoints.
- Public interfaces are typed and importable.
- A fresh local run has one documented command.
- Logs and errors do not expose secrets or unsanitized tool output.

When a test exposes a cross-component contract mismatch, report the smallest failing scenario and notify the coordinating agent before changing another agent's owned component.

## Validation pipeline (implemented)

See `docs/VALIDATION.md`. Stages: static → unit → scenarios (offline, in CI via
`.github/workflows/ci.yml`) → smoke (`doctor`) → benchmarks → gates (live, with
`agent-harness-bench pipeline <plan.toml>`).

- Code lives in `src/agent_harness/bench/`:
  - `suites.py`: gsm8k, qa, bfcl, humaneval.
  - `runner.py`: concurrency, resumable `results.jsonl`, metrics with a Wilson CI.
  - `compare.py`: gates.
  - `cli.py`: `run`, `compare`, `pipeline`, `suites`.
- A new suite implements `load(path, options) -> list[BenchItem]` and
  `async run(item, BenchContext) -> ItemResult`, and is registered in `SUITES`. Use
  `BenchContext.run_agent` to exercise the real orchestrator. Raise `BenchConfigError`
  only for misconfiguration, never for a single bad item.
- Tests (`tests/test_bench.py`) use scripted "oracle" models and the synthetic samples
  in `examples/bench/`. Never commit public dataset files or run live models in tests.
- Model-generated code runs only through an `ExecutionRuntime`: Docker by default,
  local only with an explicit unsafe flag.
- Observability:
  - `run_pipeline(on_result=, on_progress=, telemetry_path=)` reports each `StageResult`
    as it is added.
  - `BenchContext.transcripts` plus `CURRENT_ITEM` make `run_agent` save each item's
    full run, and the runner deletes passing items' transcripts unless `transcripts = "all"`.
  - `html_report.write_html_report` writes a static, escaped `index.html` at the end of
    every pipeline run.
  - The user-facing guide is `docs/TROUBLESHOOTING.md`.
- Local model residency (`bench/residency.py`, plan `[residency]`, off by default):
  - Before each smoke or benchmark entry of an `ollama`/`llama_cpp` profile,
    `ResidencyManager.prepare` unloads models above `min_params_b` (or `min_size_gb`)
    on the *other* local servers and polls until they are gone.
  - Remote profiles and the end of the pipeline call `restore`. It restores only
    models present in the snapshot taken at the first `prepare`.
  - HTTP goes through the injectable `ResidencyHttp` (tests: `tests/test_residency.py`
    fakes Ollama and a llama.cpp router).
  - The unit and scenario stages run hermetically (no `AGENT_HARNESS_*` overrides,
    no settings layers); smoke and benchmarks keep the caller's environment.

## Phase 14: scenario options and judged benchmarks

- **Scenario options:** `[orchestrator]`, `[interaction]`, `ask_user`, `wait_seconds` (the
  scenario clock moves forward while paused, then `expire_pending_input` runs),
  `error = "context_overflow"`, and `expect.event_payloads`.
- **Coverage:** keep one example scenario per orchestrator safety behaviour, so the pipeline's
  scenarios stage checks them on every run.
- **Judged `qa`:** with `match = "judge"`, the judge reply is parsed leniently
  (`parse_verdict`); an unreadable reply counts as incorrect. Prefer a judge profile
  different from the profile under test.

## Phase 15c: per-feature validation (implemented)

- **Features in the plan:** `[[feature]]` entries in `examples/validation.toml`, each with
  `name`, `tests` (unit modules, e.g. `tests.test_guardrails`), `scenarios` (files under
  `examples/evals/`), and `smoke` / `benchmarks` (names of the existing `[[smoke]]` /
  `[[benchmark]]` entries). Planned features: `core`, `providers`, `context`, `guardrails`,
  `hooks-permissions`, `subagents-planning`, `interaction`, `tools`, `memory`, `coding`,
  `api-workers`, `production`, `extensibility`, `observability`.
- **One command, any subset:** `agent-harness-bench pipeline examples/validation.toml`
  runs everything (the master run). `--feature NAME` (repeatable) limits every stage to that
  feature's entries. `--offline` drops smoke and benchmarks. `scripts/validate.sh [feature ...]`
  wraps it. The report format is unchanged (`pipeline-summary.json`, `index.html`), with a
  feature column added.
- **Coverage guard:** a unit test lists `tests/test_*.py` and fails if a module is in no
  feature, so a new test cannot drop out of the per-feature runs. Never remove a module
  from the list to make that test pass.
- **Scenarios with features on:** a scenario's `config = "../../models.toml"` or an inline
  `[harness]` table is merged into the scenario's config before the app is built. The model
  is still scripted, so these scenarios stay offline and deterministic.
- **Live entries for a feature:** they reuse the existing smoke and benchmark machinery
  (residency, gates, baselines). A feature with no live check simply has none; do not
  invent a live check that needs a paid service.
- **Implementation notes:**
  - `bench/features.py`: `parse_features`, `select_features`, `uncovered`.
  - In `run_pipeline(features=...)`:
    - The unit stage runs `python -m unittest MODULE ...` with `tests/` on `PYTHONPATH`.
    - The scenarios stage passes the selected files to `agent-harness eval`.
  - Feature `scenarios` are file stems; `calculator.toml` holds `calculator_multiplication`.
  - App-backed scenarios use `evals.scenario_application`. Its `_ScenarioModel` adds
    deterministic `embed()`, so hybrid memory works offline. Do not give `ScriptedModelClient`
    itself an `embed()`; other tests rely on it lacking one.
  - When adding a test module or scenario, add it to a `[[feature]]` in the same change.

## Phase 16b: honest benchmark failures

- **Sandbox failures:** `bench/suites.py` has `docker_problem()` (a bounded `docker info`) and
  `SandboxError`.
  - `_raise_for_sandbox_failure` checks only results from a `DockerExecutionRuntime`, so local
    execution and test doubles are never classified.
  - Exit 125 or daemon text is an error; a timeout is an error only when Docker is unavailable.
- **Preflight:** `execute_benchmark` runs the same preflight for humaneval and raises
  `BenchConfigError`, which stops the benchmark.
- **Infrastructure errors:** never let one count as a wrong answer. Gates have `min_accuracy`
  floors so that a benchmark at 0% fails even without a baseline.
- **Tests:** `tests/test_sandbox_failures.py` uses a fake `docker` on `PATH`.

## Phase 16c: progress and time left

- **`PipelineProgress`:** in a terminal it redraws one line (`\r\x1b[2K`); otherwise it logs a
  plain line every `log_every` seconds.
- **Run timing:** each `StageResult` gets `duration_seconds`, and `timings.json` records stage
  totals and the mean seconds per item for the next run's estimates.
- **Benchmarks:** `execute_benchmark(live=...)` reports items to it.
- **Tests:** `tests/test_validation_progress.py` drives everything with a fake clock and a
  `StringIO` stream.
