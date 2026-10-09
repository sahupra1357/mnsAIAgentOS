---
name: planning-delegation
description: Orchestrator-owned planning and delegation — the persisted plan (steps, dependencies, status), plan-and-execute mode, replanning, orchestrator-initiated subagents for plan steps, parallel subagents, and concurrent read-only tool calls. Use for any Phase 8 item or when changing how the orchestrator sequences work.
---

# Planning and Delegation

The model proposes; the **orchestrator owns** the plan. Plan state lives on the run and in
events, so resume continues exactly where it stopped.

## Plan model (`orchestration/planning.py`, `domain.py`)

- `PlanStep(id, title, instructions, depends_on, agent, status, result, child_run_id,
  attempts)`.
- `status`: `pending | in_progress | completed | failed | skipped`.
- `Plan(goal, steps, revision)` is stored on `RunRecord.plan`, which defaults to `None`.
- Validation: step ids are unique and match `[a-z0-9_-]{1,32}`; dependencies exist and
  have no cycles; each `agent` names a defined subagent; the count is at most
  `max_plan_steps` (default 20).
- Events: `PLAN_CREATED`, `PLAN_STEP_UPDATED`, and `PLAN_REVISED`.

## Plan-and-execute mode (`[planning]` or `run --plan`)

1. **Plan.** At creation, the system context gains the planner instructions: first
   explore with read-only tools, then call `create_plan`. Until a plan exists, only
   read-only tools and `create_plan` are offered.
2. **Execute, the orchestrator driving.** At each boundary the orchestrator picks the
   ready steps, meaning pending steps whose dependencies are all completed.
   - **Agent steps** (`agent` set): the orchestrator starts the child runs itself,
     **concurrently** for every ready agent step. When they finish, their results are
     injected as one user message: `[Plan] Step … (agent) completed: …`. No model turn
     is needed to delegate.
   - **Other steps:** the orchestrator injects `[Plan] Execute step <id>: <title> —
     <instructions>`, and the main model works with its normal tools.
     - The step ends when the model calls `update_plan(step_id, status, result)`, or
       gives a final text answer, which completes the step with that text as its result.
3. **Replan.** When a step fails, the orchestrator injects the failure and asks for
   `revise_plan`.
   - `revise_plan` replaces the steps that are not finished; completed steps are kept.
   - It is bounded by `max_replans` (default 2); past that, the run fails with a clear
     reason.
4. **Finish.** When every step is terminal, the orchestrator injects `[Plan] All steps are
   finished. Write the final answer.` The next final answer completes the run.

Injected messages are appended through checkpoints, so they are durable and replayed as
ordinary history. There is exactly one injection per boundary, which avoids consecutive
user messages.

## Parallelism

- **Read-only tool calls:** when a response has several pending calls, eligible calls are
  **prefetched concurrently** before the normal sequential loop. A call is eligible when:
  - its tool is read-only, is not an orchestrator tool, and has no recorded request yet;
  - its permission decision is not `deny` or `ask`;
  - no `pre_tool` hook is configured, since a hook could change the arguments.

  The prefetch uses the same idempotency key, so the sequential loop gets the cached
  result instantly. Events, hooks, and ordering stay exactly as before. Read-only tools
  have no side effects, so a prefetched result that is later refused changes nothing.
- **Subagents:** several `agent` calls in one response (or several ready agent steps)
  create their children one by one, then run them together with `asyncio.gather`.
  - **Budgets:** the parent's remaining tokens and cost are split evenly across the
    concurrent children.
  - **How many at once:** `[subagents] max_parallel` (0 = no limit; the root config uses 2)
    caps the children of one parent running together (`_run_children`); the rest wait for a
    slot. A child that waited gets at most the time its parent has left
    (`_cap_to_parent_deadline`). llama.cpp shares one KV cache (`ctx-size`) across concurrent
    requests: five 16K children overflowed a 32K cache on 2026-10-09.
  - **Approvals:** a child that pauses for approval pauses the parent
    (`SUBAGENT_WAITING`). The other children still finish and their results are kept.
  - **Safety:** children are separate runs, so the store's lock keeps the writes safe.
  - **Ending in an answer:** with `[orchestrator] final_answer_on_last_step = true`, a run's
    last allowed step (a child's `max_steps`, or the parent's) gets the `[Budget]` note
    (`FINAL_STEP_NOTE`) and `tool_choice = "none"`; the tools stay in the request, so the
    cached prefix holds. A child that still does not finish reports its context summary to
    the parent ("What it had found before it stopped"), not only its error. Measured on
    2026-10-08: a researcher spent its 12 steps on 31 reads and returned nothing.
    Checked live on llama.cpp (Qwen 3.6): `tool_choice = "none"` is honoured, but without the
    note Qwen can write a tool call as plain text, so keep both. An empty answer (a thinking
    model out of tokens) is reported as "returned no answer" plus the summary.
  - **Also a finished plan and the time limit:** the same setting sends the answer step of a
    finished plan (`finalizing`) with `tool_choice = "none"` (the note adds
    `PLAN_FINISHED_NO_TOOLS`). When less than `final_answer_reserve_seconds` (default 120, at most
    a quarter of the wall-clock budget; 0 = off) is left, the `[Budget]` note and
    `tool_choice = "none"` come early. An answer to that note ends a planned run even with steps
    left (`phase = "done"`, `final_step: true`). Measured on 2026-10-08 (tasks/run-lifecycle,
    Qwen 3.6 35B-A3B): after five researcher steps the parent re-read files for 15 steps,
    each one summarized, lost the "write the final answer" note, and hit 15 minutes with no answer.

## Nesting

`[subagents] max_depth` defaults to 2: top-level runs (depth 0) may start subagents, and
those children may start their own. Set `max_depth = 1` so only the top-level run
delegates. `examples/subagents.toml` does this.

## Testing

Use scripted models:
- **Plan lifecycle:** create, execute, update, and complete a plan.
- **Validation errors:** cycles, unknown agents, duplicate ids.
- **Replanning:** replan after a failed step, and the `max_replans` limit.
- **Agent steps:** they run without a model turn, and ready ones run concurrently.
  Measure with an event log of start and finish order, not wall-clock sleeps.
- **Parallel reads:** prefetch runs calls concurrently, and is skipped when a `pre_tool`
  hook exists.
- **Resume:** a plan interrupted mid-step resumes correctly.

## Live delegation benchmark (Phase 16h)

`agent-harness-bench run --suite agents --data examples/bench/agents_sample.jsonl` (also the
`agents` entry of the validation plan) measures how well a real model delegates:
- **Setup:** the top-level run may only call `agent`, and a `researcher` subagent reads the
  notes.
- **Scoring:** the answer must contain every expected fact. Delegation style is reported,
  not scored.
- **Details:** `subagents`, `max_parallel` (subagents started in the same step), and
  `facts_found`.
- **Tests:** `tests/test_agents_suite.py` uses a content-driven scripted model, because parallel
  children make reply order nondeterministic.
