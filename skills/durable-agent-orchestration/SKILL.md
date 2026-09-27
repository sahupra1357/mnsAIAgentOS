---
name: durable-agent-orchestration
description: Design or modify the agent run state machine, checkpoints, events, and persistence boundaries for this harness. Use for orchestration, run lifecycle, recovery, retries, budgets, or storage work.
---

# Durable Agent Orchestration

Build the orchestration loop as a resumable state machine over persisted state.

## Required Invariants

- Persist the run before its first model call and after each accepted model or tool result.
- A worker process is disposable; a run is recoverable from stored state.
- Use stable step IDs and idempotency keys so redelivery cannot duplicate side effects.
- Make terminal states explicit: completed, failed, cancelled, budget-exhausted, and awaiting-input.
- Treat `awaiting-input` as a durable pause. Store an approval request event, expose the pending call, and resume the same call only after an explicit allow/deny decision.
- Reject transitions that are invalid for the current state.
- Enforce step and wall-clock limits inside the loop rather than relying on callers.
- Record an append-only event for each transition and external request/result boundary.

## Boundary Design

Keep domain contracts independent of concrete persistence. The run store should expose intent-level operations such as creating a run, loading a run, appending an event, and atomically saving a checkpoint. Do not expose SQL details to the orchestrator.

Model and tool requests may be retried only when the error is classified as retryable. Persist retry attempts. Never automatically retry an ambiguous side-effecting result.

## Handoff

State any atomicity guarantees required from the store and provide network-free tests using fakes. If a contract change affects model or tool components, notify the coordinating agent instead of editing their owned paths.
