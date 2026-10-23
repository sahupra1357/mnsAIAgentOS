---
name: coding-agent
description: Build or operate a repository coding agent with workspace-safe file tools, guarded commands, approval pauses, Git workflows, and Docker-backed execution.
---

# Coding Agent

Use this skill when an agent must inspect or modify a repository, run tests, or perform Git operations.

## Workflow

The default loop is:

```text
inspect → plan → read/search → propose narrow patch → request approval
→ apply patch → request approval → run tests in sandbox → inspect diff
→ request approval for commit/PR → summarize
```

Read-only inspection tools can run automatically. File writes, test/code execution, commits, and pull requests must return an approval-required result before invoking their handler. The application must show the exact tool, risk, and arguments, then call the orchestrator approval API. A denial should be returned to the model as a tool result so it can explain or choose another path.

## Workspace Boundary

- Resolve every path beneath one configured workspace root.
- Reject absolute paths, parent traversal, protected directories, credential-like files, and writes through symlinks.
- Read bounded UTF-8 text only; skip binary and oversized files during search.
- Apply one exact text replacement atomically and reject ambiguous matches.
- Creating a file (`apply_patch` with `create`) makes missing parent folders, after the checks:
  `_resolve_new` resolves the deepest existing folder strictly (no symlink escape) and appends the
  missing names; protected folders and credential-like names stay refused, and a refused path makes
  no folders. Editing still needs the file to exist.
- Never let model input select an arbitrary working directory.
- Report file-system problems in words the model can act on, naming the path as it was
  given (never the host path): a missing path or parent is `WorkspacePathNotFound` ("no such
  file or directory: X; search_files can find the right path"), an unreadable file is "file
  could not be read: X (Permission denied)", and a binary file is "not a UTF-8 text file".
  `_safe_workspace_call` turns these into `ToolRejectedError`, including for the `cwd` and
  `path` arguments of `run_command`, `git_status`, `git_diff`, and `git_commit`. Before,
  they reached the runtime's catch-all ("Tool execution failed.", or "Tool returned an
  invalid result." for a bad `cwd`), and `search_files` failed outright on an unreadable file.

## Command Guardrails

- Execute structured `argv` with `shell=False`; never pass a model-built shell string.
- Allow only configured command names and bounded argument counts/sizes.
- Always deny shells, privilege escalation, Docker socket access, destructive filesystem commands, forced Git operations, and network escape flags.
- Approval never overrides an always-denied command.
- `find` (opt-in through `allowed_commands`) may only list files: `-delete`, `-exec`,
  `-execdir`, `-ok`, `-okdir`, `-fls`, and `-fprint*` are refused, because the workspace
  mount is read-write and those actions would change or run things on the host's files.
  Permission rules match only `argv[0]`, so `find -exec rm` would otherwise pass a
  `run_command(argv.0=rm)` deny rule.
- The root `models.toml` adds `ls`, `cat`, `head`, `wc`, and `find` to the default commands,
  with `allow` rules so they run without an approval pause, and `ask` rules for a `.env`
  argument (positions 1–4), matching `read_file(path=*.env)`. `allowed_commands` replaces
  the default list, so the defaults are repeated there.
- `allowed_commands` entries are globs: `"*.sh"` allows any script, and `"*"` allows any
  program, leaving the decision to `[permissions]` rules (Claude Code's model). The
  forbidden set still applies; `[coding] forbidden_commands` replaces it (for example
  without `sh`, so skill scripts run on a trusted local workspace). Argument checks (git,
  find, inline Python, privilege flags) always apply.
- Keep local host execution explicitly opt-in. Prefer the Docker runtime.

## Docker Runtime

The Docker backend should mount only the configured workspace at `/workspace`, use a configured image, disable networking by default, drop capabilities, set `no-new-privileges`, cap CPUs/memory/PIDs, use a read-only root filesystem with a bounded temporary filesystem, and run as the host UID when possible. Pull requests require an explicitly network-enabled configuration and credentials supplied outside the repository.

## Handoff

Test traversal and symlink rejection, exact patch behavior, dangerous-command rejection, approval pause/resume, Docker command construction, timeout, output limits, Git status/diff, and denial behavior. Never run a real destructive command or contact a real Git hosting service in tests.

## Warm per-run containers (Phase 10)

- **Modes:** `[coding.docker] mode = "per_call"` (the default: one `docker run --rm` per
  command) or `"per_run"`.
  - `per_run` starts one container on the run's first command (`docker run -d ... sleep
    max_lifetime_seconds`), uses `docker exec -w /workspace/...` for later commands, and
    removes the container when the run ends.
  - A timed-out command discards the container.
  - A container that expired or disappeared is restarted once.
- **Measured locally** (Docker Desktop, image cached): about 150 ms per command with a fresh
  container, about 55 ms per `exec` in a warm one.
- **Scope:** only commands run in the container. File tools (`read_file`, `apply_patch`, ...)
  still run in the harness with workspace path guards.
- **Mounts:** `--mount type=bind,...` must not carry `,rw`; Docker rejects it.

