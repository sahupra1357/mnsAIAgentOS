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
