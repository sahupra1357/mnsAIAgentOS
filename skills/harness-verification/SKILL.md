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
