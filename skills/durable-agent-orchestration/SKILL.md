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
- Refuse a reply with more than `max_tool_calls_per_turn` tool calls as a whole, before
  prefetch: every call gets a refusal result, so the transcript stays valid, and the count
  of such replies (derived from messages, so resume is deterministic) fails the run after
  `max_refused_repeats`.
- Treat a provider context overflow as recoverable: record a retryable `model_responded`
  with `context_overflow` and a fixed `max_input_tokens`, derive the cap from events (never
  from process memory), and bound the retries (`context_overflow_retries`).
- Paused runs may expire: `expire_pending_input` resolves overdue approvals and questions
  through the normal `resolve_approval` path (`actor = "timeout"`), `cancel`, or `_fail`,
  so the event log looks the same as a human decision.
- Every loop limit is configurable through `[orchestrator]`; keep `OrchestratorSettings`
  and `OrchestratorConfig` in step (0 = off, `max_refused_repeats = -1` = never stop).
- Record an append-only event for each transition and external request/result boundary.

## Boundary Design

Keep domain contracts independent of concrete persistence. The run store should expose intent-level operations such as creating a run, loading a run, appending an event, and atomically saving a checkpoint. Do not expose SQL details to the orchestrator.

Model and tool requests may be retried only when the error is classified as retryable. Persist retry attempts. Never automatically retry an ambiguous side-effecting result.

## Handoff

State any atomicity guarantees required from the store and provide network-free tests using fakes. If a contract change affects model or tool components, notify the coordinating agent instead of editing their owned paths.

## Plans and parallel work (Phase 8)

- **Plan state:** it lives on `RunRecord.plan` (with `planning` as the switch). Every
  change is checkpointed together with a `plan_*` annotation event. The step-level
  state machine is in `_plan_next` and `_plan_on_final`, and the design is in
  `skills/planning-delegation/SKILL.md`.
- **One message per boundary:** the orchestrator adds at most one user message per
  boundary. Notes that are not yet delivered, such as delegated results, wait in
  `plan.outbox`, so they survive pauses and crashes.
- **Idempotent children:** plan children use deterministic ids
  (`child_run_id(run, step, "plan-r<revision>-<step>")`), so a resume finds them instead
  of creating new ones.
- **Parallel prefetch:** it reuses idempotency keys, so the sequential loop remains the
  only place that records events.

## Orchestrator seams (Phase 23)

- **Tool handlers:** orchestrator-owned tools are `ToolHandler`s (`orchestration/handlers.py`):
  `applies(run, call)`, `before_approval` (runs before the permission `ask` pause, like
  `ask_user` and `exit_plan`), `parallel_safe`, and `offer` (a spec appended after the built-in
  ones). Add a control primitive with `register_tool_handler()`, not another `if` in
  `_execute_tool`. `_ORCHESTRATOR_TOOLS` stays, and a test checks the registry covers it.
- **Compaction:** `CompactionPolicy` (`orchestration/compaction.py`) decides whether and what to
  summarize; `DefaultCompactionPolicy` is the old trigger (the builder's omitted range). A custom
  range must start where the stored summary ends (or after the pinned head) and end at a unit
  boundary, or it is ignored; a range ending at the stored `covers` reuses the summary.
- **Message skeleton:** events hold no message text, but the history's shape follows from them
  (`orchestration/replay.py`): `run_created.message_roles`, `model_responded.tool_call_ids`,
  `tool_responded.retrying`. Any new code that changes `run.messages` must checkpoint an event
  that `derive_message_skeleton` understands; every scenario eval checks it, and
  `tests/test_orchestrator_seams.py` checks it at every checkpoint. Full event sourcing is
  deliberately deferred.


## Repeated calls within one reply (Phase 16f)

Identical read-only calls within one model reply execute once; later duplicates get a tool
message reusing the first result (`_reuse_same_reply_result`, `_same_reply_twin`), and
parallel prefetch skips them. `_identical_prior_calls` counts each reply at most once, so
refusals are per reply. Side-effect tools are never merged.
