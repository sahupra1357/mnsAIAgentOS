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
5. If the minimum view (system + first user + latest turn) still exceeds the window, first
   shorten the latest turn's tool outputs together, in proportion to their size (never below
   200 characters each, with the usual truncation marker). Only if that is not enough (the
   pinned head or a user message is too large), fail with a clear, non-retryable error
   rather than sending a truncated request.

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
- `orchestration/context.py`:
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

## Summarization (Phase 2, step 3)

- **When and what (Phase 23c):** a `CompactionPolicy` (`orchestration/compaction.py`, the
  orchestrator's `compaction=` argument) decides. The default summarizes exactly the range the
  builder omitted, as before. Custom policies may summarize earlier or less, but the range must
  continue the stored summary (`compaction_start`) and end at a unit boundary; otherwise it is
  ignored. The hook, usage, and `CONTEXT_SUMMARIZED` event are unchanged.

- **Domain:** `ContextSummary(covers, content)` sits on `RunRecord.context_summary`, with
  a default of `None` (SQLite round-trips it). `covers` is the index in `run.messages`
  where the kept history resumes.
  - `ContextView` adds `summarized_messages`, plus `omitted_from`/`omitted_through`, the
    half-open range of stored indices that was left out.
  - The event is `CONTEXT_SUMMARIZED`.
- **Builder:** `BudgetedContextBuilder.build(..., summary=...)` replaces
  `messages[head_end:covers]` with a `[Summary of N earlier messages]` block appended to
  the first user message. It then runs the same hysteresis over the rest.
  - A summary that no longer applies is ignored: past the end, starting on a tool result,
    or inside the head.
  - **The newest request stays verbatim.** When the summary (or an omission) covers the
    newest user message the user wrote (`latest_request_index`, which skips
    `ORCHESTRATOR_NOTE_PREFIXES` `[Plan]`/`[Budget]` notes), its text is appended after the
    summary or omission notice as `[Latest user message - answer this one]`. Only the first
    user message is pinned as a message, so on 2026-10-10 a /ui chat's second question
    ("agent-harness approve RUN_ID --allow") survived only as a summary line, and the model
    answered the first question again. The orchestrator's `_summary_task` uses the same
    prefixes.
  - `FullHistoryContextBuilder` and profiles without limits ignore it.
- **Protocol:** `ContextBuilder` is unchanged. The orchestrator passes `summary=` only
  when the run has one, so custom builders keep working until summarization is used.
- **`ContextSummarizer` (orchestration/context.py):** builds the summary request. It uses a system
  prompt, a rendered transcript with the previous summary folded in, and no tools.
  - Each message is capped, and the whole transcript is bounded by
    `summary_input_chars(limits)` (75% of the summarizing profile's usable window) or
    `max_input_chars`.
- **Orchestrator:** `_summarize` runs inside the model turn, after `MODEL_REQUESTED`,
  whenever the view reports an omitted range.
  1. It runs the `pre_compact` hook; a deny skips summarization.
  2. It calls the summarizer profile, or the run's profile if none is set.
  3. It adds the call's usage to the run's tokens and cost, and checkpoints the summary
     together with the event.
  4. On failure, it records `ok: false` and keeps plain omission. It does not retry
     while the trim point (`covers`) is unchanged.
- **Crash between the call and the checkpoint:** the summary is regenerated. Nothing
  inconsistent is persisted.
- **Rewind:** if `covers` is past the rewind cut, the summary is cleared
  (`summary_cleared` in `REWOUND`).
- **Config:** `[summarization] enabled, profile, max_output_tokens, max_input_chars`. `enabled` is
  **on in code** since Phase 41 (it stays an option: `enabled = false` for a benchmark).
  An invalid profile is a `ConfigError`, raised before any store is opened.
- **Size:** keep `max_output_tokens` well below the smallest window. The summary sits in
  the pinned head; if the head no longer fits, the run fails with
  `ContextWindowExceededError`.
- **Two sizes, kept apart (Phase 42).** The server counts a model's thinking against
  `max_tokens`, and some deployments cannot turn thinking off (GLM on the shared Modal
  endpoint ignored every switch). So:
  - **The summary's size** is `ContextSummarizer.summary_tokens`: at least
    `max_output_tokens`, more for a long range or a long previous summary, at most
    `max_share` (0.06) of the profile's input budget and a third of one summary request.
    The request states it ("Length: at most N characters"). A summary that comes back
    over it gets one rewrite request (`tighten_request`), then is cut at a line end
    (`shortened` in the event).
  - **The request's output room** is for thinking plus the summary. `complete_whole`
    (`orchestration/side_call.py`) starts at the summary's size and asks again with more
    room when the reply ends with `finish_reason = length` (four times when nothing
    fitted, twice otherwise), up to the profile's `parameters.max_tokens`. The room that
    worked is remembered for the profile (`output_room_tokens` in the event).
  - **A cut-off reply is never stored.** Still cut off at the ceiling, the range is
    summarized in two halves; if that fails too, the event says `cut off` and the view
    omits.
  - **Failures in a row** wait 3 steps, then 6, 12, … before the next attempt.
  - Memory extraction (`save_from_turn`) uses the same call; `memory_saved` has `cut_off`
    when no reply came back whole.
  A scripted summary model that answers with `finish_reason = length` is now asked again.

## Testing

Use scripted models and assert what was sent (`ScriptedModelClient.requests`): pairing
is preserved, the system prompt is kept, oversized tool output is trimmed, the too-small
window fails cleanly, and resume produces the same view.

## Provider overflow recovery (Phase 13)

- The builder's estimate (3.5 characters per token) can still undershoot, and profiles
  without `context_window` are never trimmed. When the server answers "context too
  long", adapters raise `ContextOverflowError` with `limit_tokens` / `requested_tokens`
  when the message states them (`model_gateway/errors.py`).
- The orchestrator records a retryable `model_responded` event with `context_overflow`,
  `estimated_tokens`, and `max_input_tokens`, then retries the step. `_build_view` applies
  the smallest cap recorded for the run's current profile through `fit_to_tokens`, which
  replays the same unit-wise hysteresis against that fixed cap (and limits each tool
  output to a quarter of it). Because the cap is fixed and persisted, resume rebuilds the
  same view and later turns stay append-only between trims.
- Messages omitted by the cap are not summarized (the summarizer only sees the
  builder's own omitted range).

## Configuration and server counts (Phase 14)

- `[context]` (`ContextConfig`) maps to `ContextPolicy` through
  `context_policy_from_config`. The application, the orchestrator's overflow cap
  (`fit_to_tokens`, `estimate_tokens`), the summarizer's input size, and benchmarks all use
  it. An explicit `build_application(context_policy=...)` still wins.
- `token_counting = "server"` on a profile with `context_window`: before sending, if the
  estimate is at least half the limit, the orchestrator asks the client
  (`count_input_tokens`, `input_token_limit`). A count over the limit records a
  `CONTEXT_CAPPED` annotation event, whose cap `_overflow_cap` reads alongside overflow
  recoveries, and the view is rebuilt once.
- `_fitting_cap` raises any new cap until the minimum view fits; when no smaller cap fits,
  no cap is recorded.

## Phase 26: context as working memory (implemented, opt-in)

Since Phase 31 these are the only behaviour, with no switch: the anchored layout (the legacy layout was deleted),
recovery hints, the ledger, estimate calibration, summarizing at a threshold before anything is omitted, and
chunked summaries. The tuning values (`default_window`, `stale_tool_output_chars`, `style`, the ratios)
are set by the root `models.toml`. Under a tight budget the state message shrinks: first without the ledger,
then a short note without the header, so it never costs more than the turns it stands for.

- **Anchored layout** (the only one): the first user message is sent as stored.
  The summary, the omission note and the ledger are one user message after it
  (`STATE_HEADER`), and the user's latest request follows as its own message when the summary or
  a trim covers it. The legacy layout (glued onto the first message) is unchanged.
- **Staged trimming**: over the high-water mark, tool outputs of turns older than
  `keep_recent_units` are first shortened to `stale_tool_output_chars`; whole turns are dropped
  only if that is not enough. The newest turn's outputs keep at least `min_fitted_output_chars`.
- **Ledger** (`ledger = true`, `tool_ledger`): one line per tool call that is no longer shown,
  written without a model. A summary may forget a path; the ledger cannot.
- **Early compaction** (`ThresholdCompactionPolicy`):
  summarizes at `trigger_ratio` of the input budget and keeps about `keep_ratio`, so no request
  is sent with a hole. A long range is summarized in parts (`covers` advances per
  part); `style = "structured"` uses fixed sections. After a failed summary an early policy waits
  `_SUMMARY_RETRY_STEPS` steps.
- **Calibration**: `model_responded` records `estimate_ratio` when the
  provider counted clearly more input tokens than estimated; later views scale the estimate. It
  only rises, and is read from events, so resume builds the same views.
- **Caps**: a builder with `native_caps` (anchored) takes `max_input_tokens` and `scale` in
  `build`; `fit_to_tokens` is only used for the legacy layout (it would treat the state message
  as an ordinary turn).
- **Repeated reads**: past
  `max_identical_tool_calls`, a read-only call gets the earlier result with a note;
  `max_refused_repeats` still ends a loop.
- `ContextView.budget_tokens` is new (default `None`). Tests: `tests/test_working_context.py`.

## Phase 41: what a request really carries

- **Summaries without a config.** `AgentOrchestrator(context=<a builder that trims>)` with no
  `summarizer` builds a default `ContextSummarizer`, sized from the builder's windows
  (`BudgetedContextBuilder.summary_input_chars`). Pass `summarizer=NO_SUMMARIZER` for omission
  only; `build_application` does that for `[summarization] enabled = false`, and `BenchContext`
  does it by default (suites measure the loop, one model call per step).
- **Reserve.** Hook context and the status block are appended to a request after the view. They
  are now built first and passed to `build(..., reserve=...)`: the view leaves room for them and
  `estimated_tokens` includes them. They do **not** move the high and low water marks, so a
  status block that comes and goes does not change which turns are kept (the cached prefix).
- **A request larger than the window.** When the pinned head takes more than 75% of the input
  budget, the middle of the user's first message is left out of the view, with a marker
  (`ContextView.shortened_request_chars`, `context.shortened_request_chars` in `model_responded`).
  The stored message is unchanged. System messages are never shortened: if they do not fit, the
  run still fails with `ContextWindowExceededError`.
- **The pinned head** ends after the first message the user wrote. A harness message before it
  (`[Memory]` notes, an `[Earlier run` handoff) is pinned with it.
- **Summarizer input.** Only the transcript is shortened; the user's request and the previous
  summary are sent whole. What did not fit is recorded as `truncated_chars` in
  `context_summarized`. One turn too long for a request (many large parallel results) is
  summarized in several requests (`ContextSummarizer.pieces`, `requests` in the event); the stored
  summary and `covers` change only when the whole range is done.
- Tests: `tests/test_context.py` (`ReserveAndHeadTests`), `tests/test_summarization.py`.
