# Run agent-harness from any folder

Like Claude Code, one installed `agent-harness` serves every folder: start it in a
project, and that project is the workspace. One config (this repository's
`models.toml`) holds the models, tools, agents, hooks, and skills; each project can
add its own settings, instructions, and skills.

## Set up once

```bash
# 1. Put the command on your PATH (editable: changes in the repository apply at once)
uv tool install --editable /path/to/mnsAIAgentOS      # or: pipx install -e /path/to/mnsAIAgentOS

# 2. Link ~/.agent-harness/models.toml to the repository's config
cd /path/to/mnsAIAgentOS
agent-harness init --global

# 3. Keys for every folder (optional): the repository's .env is read too
cp .env ~/.agent-harness/.env && chmod 600 ~/.agent-harness/.env
```

The link is followed, so the config's relative paths (`.agent-harness/agents/`, hooks,
skills, tools, plugins) still point into the repository: there is one copy of each.

## Use it

```bash
cd ~/dev_data/VideoPipeline
agent-harness                          # chat
agent-harness "summarize this project" # one run
agent-harness runs                     # this project's runs only
agent-harness config show              # merged settings, the layer of each, and the project root
```

## What changes in another folder

`[project] root = "cwd"` (on in the root `models.toml`) turns this on. Started in the
config's own folder, nothing changes.

| Concern | Where it comes from |
|---|---|
| Project root | nearest folder at or above the current one with `.git` or `.agent-harness/` (`[project] markers`); never your home folder; otherwise the current folder. `--workspace DIR` names it directly. |
| Coding workspace | the project root (`[coding] workspace`, if relative, is relative to it) |
| Run database, artifacts, snapshots, memory | `~/.agent-harness/projects/<path-slug>/` (`[project] state_dir`); `--database` still overrides |
| Settings | global config, then `<project>/.agent-harness/settings.toml`, then `settings.local.toml` (keep it out of git), then enterprise policy |
| Instructions | configured files, then the project's `AGENTS.md` and `CLAUDE.md` (`[project] instruction_files`), then subdirectory files down to the current folder |
| Skills | `<project>/.agent-harness/skills`, `<project>/.claude/skills`, the configured paths, `~/.claude/skills` (the first skill with a name wins) |
| Templates | `<project>/.agent-harness/templates`, then the configured paths |
| `.env` | `./.env`, the config's `.env`, `~/.agent-harness/.env` (earlier files and shell variables win) |

`[project] claude_code = false` leaves out `CLAUDE.md`, `.claude/skills`, and
`~/.claude/skills`. Each skill's description is in every system prompt, so narrow
`~/.claude/skills` by name with `[project] claude_skills = ["aisystems-video", "hyperframes*"]`
(globs; default `["*"]`) when a small context window is tight. Claude Code's
`description: >` block descriptions are read in full. Claude Code skills that rely on Claude Code's own tools (a browser
daemon, `Task`) work less well; the harness offers `read_file`, `search_files`,
`apply_patch`, and `run_command`.

## Project settings

`agent-harness init` in a project writes a commented `.agent-harness/settings.toml`.
Lists add to the global lists; values replace them. Relative paths (skills, templates,
instruction files, agent prompt files, Python tool paths, plugins) resolve against the
project. Hook commands run in the config's folder; give project hooks absolute paths.

A video project might use:

```toml
[coding]
allowed_commands = ["ffmpeg", "ffprobe", "npx", "node", "bash", "modal"]
command_timeout_seconds = 1800
execution_runtime = "local"        # use the Mac's own ffmpeg, Node, and modal login
allow_local_execution = true
```

or keep the Docker sandbox with an image that has the tools:

```toml
[coding.docker]
image = "video-tools:latest"
network_enabled = true
memory = "8g"
```

A project's settings can add hooks, permissions, and tools, so only run agent-harness
in folders whose `.agent-harness/` you trust, or set `[project] settings = false`.
Enterprise settings still apply last, and guardrails and deny rules still hold.
