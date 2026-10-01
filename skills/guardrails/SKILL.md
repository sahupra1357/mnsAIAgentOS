---
name: guardrails
description: Input, output, and tool-output guardrails against prompt injection, jailbreaks, system-prompt leaks, secrets, and PII; taint mode; the optional Llama Guard model guard; and the guardrails red-team benchmark. Use when changing guardrails/, the [guardrails] config, refusal behavior, or anything that lets untrusted text reach the model.
---

# Guardrails (Phase 12a)

Read `docs/GUARDRAILS.md` first: it is the user-facing contract.

## Layout
- `src/agent_harness/guardrails/detectors.py`: deterministic detectors returning `Finding`s with spans.
  The source must stay ASCII: write invisible characters as escapes, and in tests build them with `chr()`.
- `config.py`: `GuardrailsConfig.from_mapping` (strict: unknown keys, bad actions, and bad regexes raise).
- `engine.py`: `Guardrails.check_input/check_output/check_tool_output` returning `GuardDecision`
  (strongest action wins: block > redact > flag > allow); the canary helpers; `classify` (Llama Guard),
  and `parse_llama_guard`.
- Orchestrator hooks:
  - `_guard_input`: top of `_perform_model_turn`, over `messages[guarded_through:]` user text.
  - `_guard_output`: before `_accept_model_response`; its events are passed in, so they share the checkpoint.
  - `_append_tool_result`: untrusted tools; events are recorded with `TOOL_RESPONDED`.
  - `_permission(..., run)`: taint mode.
  - `_add_canary`: in `create_run`.

## Invariants
- Off by default: with `guardrails=None` nothing changes (no canary, no streaming change).
- A refusal is a `GUARDRAIL_REFUSED` *state* event plus an assistant message; `_has_accepted_final_response`
  accepts it, and `_advance` completes the run without output-schema checks or planning.
- `GUARDRAIL_DECIDED` is an annotation event and **never contains matched text**: only detector, category,
  action, count, and location.
- `guarded_through` makes input checks (and guard-model calls) happen once per message across resumes.
- Output checks disable streaming unless `allow_streaming = true`: never stream text a guard may block.
- Taint mode asks for approval for any non-read-only tool after untrusted output, until the next user
  message, even in bypass mode. Deny rules still win.
- The canary is derived from the system prompt text (`canary_for`), so it is stable across runs (cache).
- Guard-model tokens count against the run budget. The guard uses its own profile, so the main model's
  prefix cache is untouched.

## Testing
- `tests/test_guardrails.py`: detectors (attack, extraction, and benign lists), config, engine, orchestrator
  integration (refusal, continue, redaction, ask_user answers, canary, reasoning leaks, streaming off,
  tool-output scanning, taint and bypass), the model guard (sequential, parallel cancel, errors), durability,
  and application wiring.
- Network-free: a scripted model answers `metadata["purpose"] == "guard"` requests separately.
- Measure detection rates with `agent-harness-bench run --suite guardrails`. When adding a detector, add
  benign look-alikes to the sample and keep the benign pass rate at 100%.
