---
name: interactive-extensibility
description: Interactive and extensibility features modeled on Claude Code-style harnesses — multi-turn continuation, mid-run cancel and steering, hooks, subagents, detailed permission rules and modes, plan mode, task tracking, instruction files, skills, prompt templates, rewind, layered settings, and plugins. Use for any Phase 6 item.
---

# Interactive and Extensibility Features (Phase 6)

Every feature here must keep the harness's core guarantees: durable state, append-only
events, deterministic resume, bounded budgets, and hard guardrails that nothing
overrides. The orchestrator gains new **states and events**, not provider or UI logic.
Anything user-supplied (hooks, rules, instruction files, plugins) is configuration loaded
at a boundary and validated there.

## Design rules for the whole phase

- A new capability is a protocol with a no-op default. With nothing configured, behavior
  and existing tests are unchanged.
- Every decision that changes a run (hook verdict, permission match, mode change, steer
  message, rewind) is an event, so resume and audit replay it rather than recomputing it.
- New `RunStatus`/`EventType` members are additive. Update both stores and the event
  decoding, and cover them with persistence tests.
- Nothing grants capability implicitly: installing a plugin or adding a skill never
  enables tools or bypasses approval by itself.

## Status

6a is implemented: items 1, 2, and 15.
- `AgentOrchestrator.continue_run`, `steer`, `usage`.
- `SteerStore` (a `steer_messages` table) plus `STEER_APPLIED` / `USER_MESSAGE_ADDED`
  events.
- The run loop refreshes at each step boundary and recovers from a mid-step
  `ConcurrencyError` by reloading (bounded).
- CLI: `continue`, `steer`, `cancel`, `status`.
- API: `POST /runs/{id}/messages`, `POST /runs/{id}/steer`, `GET /runs/{id}/usage`.

Deviations from the design below:
- A steer is durable in its own table instead of a `STEER_QUEUED` event. That event
  would bump the run version and race the worker.
- Cancel takes effect at the next step boundary. An in-flight model call is not
  interrupted, but its response is discarded.

6b is implemented: items 3, 5, 6, and 7.
- `hooks.py`: `HookRunner`, `CallableHook`, `CommandHook` (argv, JSON I/O, timeout,
  output limit), fail-closed for guarding events.
- `permissions.py`: rule parser, `PermissionPolicy`, and the `exit_plan` spec.
- `RunRecord.mode`, `set_mode`, the `MODE_CHANGED` / `HOOK_DECIDED` events, and
  `resolve_approval(feedback=, mode=)`.
- Pipeline in `_execute_tool`: pre_tool hook → permissions on the final arguments →
  record the decision in `TOOL_REQUESTED` (replayed on resume) → ask/approval →
  execute → post_tool hook.
- State checks skip annotation events (`HOOK_DECIDED`, `MODE_CHANGED`, `CONFIG_DRIFT`).
- Config: `[permissions]` and `[[hooks]]`. CLI: `run --mode`,
  `approve --feedback/--mode`, `mode`. API: `mode` on create, approval
  feedback/mode, and `POST /runs/{id}/mode`.

Not implemented:
- `pre_compact`: implemented with context summarization (Phase 2, step 3). It runs
  before each summary call; a deny skips the summary and keeps plain omission.
- `subagent_stop`: comes with subagents in 6c.

6c is implemented: items 4 and 8.
- `agents.py`: `SubagentDefinition`, `SubagentLimits`, the `agent` / `todo_write`
  specs, and deterministic `child_run_id`.
- The orchestrator owns the virtual tools `agent` and `todo_write`. They go through
  hooks, permissions, and approval like other tools, then dispatch after approval.
- A child is an ordinary run with `parent_run_id`, `parent_call_id`, `depth`, and
  `allowed_tools`, created idempotently and executed inline.
  - The child's usage rolls into the parent in the same checkpoint as the result.
  - A child approval pauses the parent (`SUBAGENT_WAITING`). `pending_approval` and
    `resolve_approval` on the parent forward to the child, and the parent resumes
    with `SUBAGENT_RESUMED`.
  - Cancel cascades to children, and children inherit the parent's permission mode.
- Hooks: `subagent_stop` adds context to the child's result.
- Config: `[[agents]]` (prompt or prompt_file), `[subagents]`, `[tasks]`.
- Todos: `RunRecord.todos` and `TODOS_UPDATED`, shown in `status`,
  `GET /runs/{id}`, and `/usage`.

6d is implemented: items 9, 10, 11, 13, and 14.
- `settings.py`: layers user → project → local, then enterprise last (it wins),
  with provenance and `config show`.
- `instructions.py`: instruction files, the skills index, `load_skill`, and
  `SystemContextBuilder`, applied once in `create_run` for top-level runs.
- `templates.py`: `/name` expansion with `$ARGUMENTS`, `$1`, and `${key}`.
- `plugins.py`: manifest validation, install (copy; symlinks rejected), and
  enable-by-name.
- Config: `[instructions]`, `[skills]`, `[templates]`, `[plugins]`.
- CLI: `templates`, `skills`, `config show`, `plugin install/list/remove`, and
  `run --arg`.
- API: `template_args` and `GET /templates`.

Deviation from the design below: enterprise settings are applied last and win on
scalars, instead of first. Rule lists still concatenate, so an enterprise deny can
never be removed.

6e is implemented: item 12.
- `ToolSpec.writes_paths` declares file-path arguments (`apply_patch` sets `path`).
- `snapshots.py` `WorkspaceSnapshotter` stores pre-images content-addressed.
- The orchestrator checkpoints `FILE_SNAPSHOT` *before* executing, so a replay never
  snapshots a post-image.
- `rewind(run_id, to_step, confirm=)` is a dry run unless confirmed. When applied:
  - files are restored to their earliest pre-image from that step on;
  - messages from that step on move to `RunRecord.archived_messages` (never
    deleted);
  - a `REWOUND` event is recorded, and approvals and subagent waits before it are
    closed;
  - the run becomes `completed`, so it can be continued;
  - side effects it cannot undo are listed.
- CLI `rewind --to-step N [--yes]`; API `POST /runs/{id}/rewind`.

Deviation from the design below: the design called for a view cursor over the
message history. Instead, messages from the rewind point on move to
`archived_messages`, which keeps all loop logic on `run.messages` unchanged. Nothing
is deleted.

## 1. Continuing a conversation
- `continue_run(run_id, message)`: allowed from `completed` (and `budget_exhausted`, if
  budget is also raised explicitly). Appends a user `Message`, emits `USER_MESSAGE_ADDED`,
  and sets the status to `running`. Terminal-state checks must treat this as the only
  sanctioned reopen.
- Budgets are cumulative across turns unless the caller passes a new budget (recorded
  in the event).

## 2. Cancel and steer mid-run
- Before each model or tool step, the loop re-reads a lightweight control record
  (status + pending steer messages). Cancellation takes effect at the next boundary.
  In-flight model calls may be abandoned with `asyncio` cancellation; in-flight tool calls
  finish or time out, because they may have side effects.
- Steer messages are queued durably (`STEER_QUEUED`) and injected as user messages at the
  next model-turn boundary (`STEER_APPLIED`), never between a tool call and its result.

## 3. Hooks
- Events: `session_start`, `user_prompt`, `pre_model`, `post_model`, `pre_tool`,
  `post_tool`, `pre_compact`, `stop`, `subagent_stop`.
- Handler kinds: in-process callables (the library API) and external commands (structured
  argv, JSON on stdin and stdout, timeout, output limit, no shell).
- Verdicts: `allow`, `deny(reason)`, `modify(arguments)` (pre_tool only; re-validate
  against the schema), and `add_context(text)`. `stop` can request continuation, bounded
  by budget.
- A hook can never override a hard guardrail or an always-denied command. A hook failure or
  timeout follows a configured policy (`fail_open`/`fail_closed`, default closed for
  pre_tool).

## 4. Subagents
- The `agent` tool creates a child `RunRecord` with `parent_run_id` and `parent_call_id`,
  its own profile, tool subset, system instructions, and budget ≤ the parent's
  remaining budget.
- The parent's tool call stays pending until the child is terminal. The result returned
  to the parent is only the child's final message (plus status). Child usage rolls up
  into the parent.
- Enforce a depth limit and a count limit. Resume handles both levels: resuming the parent
  resumes an unfinished child first. Idempotency key for the spawn = parent key.
- Subagent definitions live in files (name, description, allowed tools, profile, prompt).
- Nothing is offered until `[[agents]]` are defined; then the model gets an `agent` tool
  whose description lists the agents. `[subagents] max_depth` defaults to 2, so a child
  is offered `agent` too and can start its own subagents; use `max_depth = 1` to allow
  delegation from the top-level run only. `examples/subagents.toml` has three working
  definitions (researcher, reviewer, test-runner) with prompt files in
  `examples/agents/`.
- Orchestrator-initiated and parallel subagents are Phase 8 (`skills/planning-delegation`).

## 5–7. Permission rules, modes, and plan mode
- Rule syntax: `tool` or `tool(field=glob, ...)`, where field paths address
  arguments (e.g. `run_command(argv.0=pytest)`, `read_file(path=src/**)`).
- Evaluation order: hard guardrails → deny rules → mode → ask rules → allow rules → risk
  policy default. Deny wins over everything except guardrails, which always deny.
- Modes: `default`, `accept_edits` (auto-allow workspace file edits), `plan` (read-only
  tools plus `exit_plan`), and `bypass` (auto-allow except deny rules and guardrails).
  The mode is stored on the run. Changes emit `MODE_CHANGED`.
- Plan mode: `exit_plan(plan)` → `awaiting_input` with an approval request of kind
  `plan`. Approve → switch to the chosen mode and continue. Reject → the feedback becomes
  a tool result.

## 8. Task tracking
- `todo_write(items[])`: each item has `id`, `content`, and a status
  (`pending|in_progress|completed`). The list replaces the run's list, is persisted on the
  run, and emits `TODOS_UPDATED`. Surface it in the CLI status and API.

## 9–11. Instruction files, skills, prompt templates
- Instruction files: discovered from configured roots (user → project → subdirectories
  of the workspace), concatenated in precedence order with source headers, bounded in
  size, and hashed into the run's version record. They are loaded once at creation and
  never re-read mid-run, which keeps the prompt prefix stable.
  Implemented (6f): `[instructions] subdirectories = true` appends the `filenames`
  (default `AGENTS.md`) found in each directory between the project root (the config
  file's directory, excluded) and the working directory at run creation, outermost
  first, after `files`. Files already listed, directories outside the root, and
  symlinks that escape it are skipped; filenames must be plain names.
- Skills: index `skills/*/SKILL.md` frontmatter (`name`, `description`). Put only the
  index in the system prompt. `load_skill(name)` (read-only) returns the body. Validate
  names; no path input.
- Templates: `templates/<name>.md` with `$ARGUMENTS`/named placeholders, expanded at the
  CLI/API boundary into the user message. Record the template name and hash in the event.

## 12. Rewind
- Before a file-modifying tool runs, store pre-images of the target files in the artifact
  store (`FILE_SNAPSHOT`).
- `rewind(run_id, step)` restores the pre-images from that step onward, adds a
  `REWOUND` event, and sets a view cursor so the context builder omits later turns. It
  never deletes events or messages. It requires approval, because it writes files.

## 13–14. Layered settings and plugins
- Settings layers: enterprise (read-only) → user → project → local, merged with defined
  semantics (scalars override; rule lists concatenate; enterprise deny cannot be removed).
  Show the result with `config show --effective`, including where each value came from.
- Plugins: a manifest (`plugin.toml`) declaring templates, subagents, hooks, and skills.
  Install means copy plus validate. Enable is an explicit settings entry. Plugin hooks are
  subject to the same permission and guardrail rules.

## 15. Usage display
- `status RUN_ID` / `GET /runs/{id}/usage`: steps, tokens (input/output/cached), cost
  (marked estimated), wall time versus budget, and child-run rollup.

## `ask_user` (Phase 11d)
- `interaction.py` defines `ASK_USER_SPEC`. It is offered only with `[interaction] ask_user = true`
  (`AgentOrchestrator(ask_user=True)`), after `todo_write`, and is read-only, so plan mode
  and the planning phase keep it.
- The call is validated against its schema, then `_request_approval(kind="question")` pauses
  the run. There is no separate pause mechanism, so worker leases, subagent forwarding
  (`_resolve_child_approval`), and wall-clock compensation all come from the approval path.
- `resolve_approval(..., answer=...)` / `answer(...)` appends the tool result (`SUCCESS`, "The
  user answered: ...") and `TOOL_RESPONDED` in the same checkpoint as `APPROVAL_RESOLVED`.
  So the call is never pending again, and resume cannot ask twice. Approving a question
  without an answer, or answering a non-question, raises `ValueError`. Denial declines.
- Events carry `answer_chars` only; the text lives in the conversation.

## Testing

Use network-free tests with scripted models:
- **Continuation:** continue after completion; a disallowed state is rejected.
- **Cancel and steer:** cancel between tool calls; steer is injected at the correct
  boundary.
- **Hooks:** deny, modify with re-validation, and fail-closed.
- **Subagents:** the child's budget rolls up into the parent's; resume works mid-child.
- **Permissions:** rule precedence, and guardrails still deny under `bypass`.
- **Plan mode:** approve and reject.
- **Tasks:** the todo list is persisted.
- **Instructions:** files load in precedence order, and their hash is recorded.
- **Skills:** progressive loading works.
- **Rewind:** files are restored, and the event history stays intact.
- **Settings:** layers merge in the right precedence.
- **Plugins:** a plugin that isn't enabled has no effect.

## Phase 15a: CLI defaults (implemented)

- **Config discovery:** `--config` → `AGENT_HARNESS_CONFIG` → `./models.toml` →
  `~/.agent-harness/models.toml`. It is resolved once in the CLI. `load_config()` itself is
  unchanged, so library callers and tests see no difference. Layered settings still apply on
  top of the config that was found.
- **Implicit `run`:** in `main()`, if `argv[0]` is neither a known subcommand nor an option,
  `run` is inserted before argparse runs. A prompt that equals a subcommand name needs an
  explicit `agent-harness run "status"`.
- **REPL:** `agent-harness` with no arguments. The first line creates a run, later lines call
  `continue_run`, and streaming is on.
  - A pause for approval prompts `[y/N]`; a pause for `ask_user` prompts for the answer, inline.
  - `/new`, `/status`, `/exit` are REPL commands; any other `/name` expands a template.
  - Ctrl-C cancels the current run, not the REPL.
  - The run is persisted as usual, so `resume` and `runs` see it.
- **Full-feature config (15b, implemented):** the root `models.toml` turns features on through config only.
  Never flip a dataclass default to "on". Defaults are part of the version hash and the tests.
- **Profile switches:** `enabled = false` in `[models.X]`, popped in `load_config` before
  the profile is parsed. Routes to the profile are dropped and fallbacks skip it. A disabled
  default is a `ConfigError`. Tests: `tests/test_cli_defaults.py`.
