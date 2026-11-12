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
- **A refused input message is withheld (Phase 42).** `_refuse` replaces what the user sent in the
  refused turn with `WITHHELD_NOTE` (a `[Guard]` harness message) and moves the original to
  `archived_messages`; before, the next message that passed carried the refused text to the model.
  `explain_refusal` (off in code, on in the root config) names the detector in the reply.
- **Flooding is `redact` by default (Phase 42).** Only a fragment with a letter or digit counts as a
  repeat; a repeat is collapsed to one copy and a count, and a message over `max_chars` loses its
  middle. Eleven pasted logs were refused in one live chat for their rule lines and padding.
- **The guard pause doubles** (60 s, 120 s, … 15 min) while the guard keeps missing its deadline.
- A refusal is a `GUARDRAIL_REFUSED` *state* event plus an assistant message; the phase becomes `answer`
  (`phase_after`), and `_advance_answer` completes the run without output-schema checks or planning.
- Stored text the harness adds as a message (recalled `[Memory]` notes, an `[Earlier run` handoff) is
  scanned with `check_tool_output` when it is added (`guardrail_decided` with `stored: true`) and is not
  input: `_guarded_input` skips it, so a quoted e-mail address in a note cannot get the user's turn refused.
- Taint for prompts is per turn (`_tainted`); taint for what a run *stores* (memory notes, its handoff) is
  the whole run (`_holds_untrusted`). `<stored-data>` wraps notes and handoffs a tool returns without tainting.
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
