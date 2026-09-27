---
name: context-management
description: Build a bounded prompt for each model call from durable run state — token estimation, context-window budgeting, truncation, tool-output trimming, and summarization. Use when changing what messages are sent to the model, adding context_window limits, or handling long runs on small-context local models.
---

# Context Management

The database is the source of truth; the prompt is a bounded view of it. Never delete or
rewrite `RunRecord.messages` to fit a context window. Build a separate, per-call view.

## Contract

- Add a small protocol, e.g. `ContextBuilder.build(run, tools, window) -> tuple[Message, ...]`,
  and inject it into `AgentOrchestrator` with a default that sends everything (current
  behavior) so existing tests stay unchanged.
- The window comes from the model profile (`context_window`, with `max_output_tokens`
  reserved). If it is unknown, send everything.
- Estimate tokens without a tokenizer dependency (e.g. `ceil(chars / 3.5)` plus
  per-message overhead) and keep a safety margin. Correct the estimate with reported
  `usage.input_tokens` when available.

## Rules

1. Always keep system messages and the first user message.
2. Never split an assistant `tool_calls` message from its matching `tool` result messages.
   Providers reject orphaned tool results.
3. Trim large tool outputs in the view (head + tail + "[truncated N chars]"). Keep the
   full output in events.
4. Drop or summarize the oldest middle turns first. If summarizing, the summary is a model
   call: record it as an event, count it against the run budget, and make it
   deterministic on resume by persisting the summary instead of regenerating it.
5. If the minimum view (system + first user + latest turn) still exceeds the window, fail
   with a clear, non-retryable error rather than sending a truncated request.

## Prefix-cache friendliness

Self-hosted servers (vLLM, SGLang, llama.cpp) reuse KV cache only for an identical token
prefix, so the context view must keep its prefix stable:

- Keep the system prompt fixed: no timestamps, run IDs, or random content.
- Keep tool specs in a fixed order, serialized as canonical JSON.
- History is append-only between trims.
- Trim in large chunks only when a high-water mark is crossed (e.g. 85% of the window),
  cutting down to a low-water mark (e.g. 60%). Never use a per-step sliding window, which
  changes the prefix every call and gives near-zero cache hits.
- After a trim, the new view is itself append-only until the next high-water mark.
- Test that consecutive views between trims are strict prefixes of each other.

## Implementation (Phase 2)

- `domain.py`: `ContextView` (messages + stats), the `ContextBuilder` protocol, and
  `ContextWindowExceededError` (a `PermanentModelError`, so the run fails with an event
  and no retry).
- `context.py`:
  - `FullHistoryContextBuilder` (the orchestrator default).
  - `BudgetedContextBuilder` (limits per profile, plus a `ContextPolicy`).
  - `limits_from_profiles` (`context_window`, with reserve = `parameters.max_tokens`
    or 1024).
- `application.py` wires `BudgetedContextBuilder` from config. The orchestrator
  builds the view inside the model-call `try` and adds a `context` stats block to
  `MODEL_RESPONDED` only when the view differs from the history.
- **Hysteresis without stored state:** the builder replays units from the start on every
  call (O(n)). The trim point is therefore a pure function of the messages: resume is
  exact, and views between trims are strict extensions of the previous view.
- **Units:** tool results always attach to the preceding unit, so a cut can never orphan
  one. The pinned head is the leading system messages plus the first user message.
- **Omission notice:** appended to the first user message only while messages are
  omitted. It avoids consecutive same-role messages that strict chat templates reject.
- **Deliberately not done:**
  - No usage-based calibration of the estimate: it would change past trim decisions.
  - No summarization yet: it needs a persisted summary event and budget accounting,
    and is deferred.

## Testing

Use scripted models and assert what was sent (`ScriptedModelClient.requests`): pairing
is preserved, the system prompt is kept, oversized tool output is trimmed, the too-small
window fails cleanly, and resume produces the same view.
