"""The project a run works on, found from the launch directory (Phase 24).

With ``[project] root = "cwd"``, ``agent-harness`` behaves like Claude Code: one
config (usually ``~/.agent-harness/models.toml``) serves every folder, and the
project is the folder it is started from. The project root is the nearest
directory at or above the launch directory that holds a marker (``.git`` or
``.agent-harness/``); the home directory never counts, because
``~/.agent-harness`` is the global folder. Without a marker, the launch
directory is the root.

In a project:

- the coding workspace (``[coding] workspace``, relative) resolves against the root;
- run state (database, artifacts, snapshots, memory) lives in
  ``~/.agent-harness/projects/<slug>/``, so project folders stay clean;
- ``<root>/.agent-harness/settings.toml`` and ``settings.local.toml`` are merged
  over the config (layers ``workspace`` and ``workspace-local``); relative paths in
  them resolve against the root;
- ``AGENTS.md`` (and ``CLAUDE.md``) at the root are instruction files, and
  ``<root>/.agent-harness/skills`` (plus ``.claude/skills`` and ``~/.claude/skills``)
  are skill directories.

When the root is the config file's own directory, nothing changes: the run is
not a project run and keeps the config-relative behavior.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT_MODES = ("config", "cwd")
SET_ROOT_ENV = "AGENT_HARNESS_SET__PROJECT__ROOT"
PROJECT_DIR = ".agent-harness"


def home_dir(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("HOME", str(Path.home())))


def project_mode(raw: Mapping[str, Any], environ: Mapping[str, str]) -> str:
    """``[project] root`` after the config layers, or its ``AGENT_HARNESS_SET__`` override."""

    section = raw.get("project")
    value = section.get("root", "config") if isinstance(section, Mapping) else "config"
    return str(environ.get(SET_ROOT_ENV, value)).strip().strip('"').lower()


def find_project_root(
    start: Path, markers: tuple[str, ...] = (".git", PROJECT_DIR), *, home: Path | None = None
) -> Path:
    """The nearest directory at or above ``start`` with a marker; ``start`` if none."""

    start = start.resolve()
    stop = home.resolve() if home is not None else None
    for directory in (start, *start.parents):
        if directory == stop:
            break  # ~/.agent-harness is the global folder, not a project marker
        if any((directory / marker).exists() for marker in markers):
            return directory
    return start


def project_slug(root: Path) -> str:
    """A directory name for a project path, like Claude Code's: ``/a/b c`` -> ``-a-b-c``."""

    return re.sub(r"[^A-Za-z0-9._-]", "-", str(root.resolve()))


def project_state_dir(root: Path, state_dir: str, environ: Mapping[str, str]) -> Path:
    base = Path(state_dir).expanduser()
    if not base.is_absolute():
        base = home_dir(environ) / base
    return base / project_slug(root)


def resolve_project_root(
    raw: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    config_dir: Path | None,
    workspace: Path | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """The project root for this launch, or ``None`` for config-relative behavior.

    ``workspace`` (the CLI's ``--workspace``) names the root directly and turns
    project mode on. The root of the config file's own directory is not a project.
    """

    if workspace is not None:
        root = workspace.expanduser().resolve()
    else:
        mode = project_mode(raw, environ)
        if mode not in PROJECT_ROOT_MODES:
            from agent_harness.config.loader import ConfigError

            raise ConfigError(f"project.root must be one of {', '.join(PROJECT_ROOT_MODES)}")
        if mode == "config":
            return None
        section = raw.get("project")
        markers: tuple[str, ...] = (".git", PROJECT_DIR)
        if isinstance(section, Mapping) and isinstance(section.get("markers"), list):
            markers = tuple(str(item) for item in section["markers"])
        root = find_project_root(cwd or Path.cwd(), markers, home=home_dir(environ))
    if config_dir is not None and root == config_dir.resolve():
        return None
    return root


def project_layer_paths(root: Path) -> list[tuple[str, Path]]:
    folder = root / PROJECT_DIR
    return [("workspace", folder / "settings.toml"), ("workspace-local", folder / "settings.local.toml")]


# Keys whose values are paths; in a project layer they resolve against the project root.
_PATH_LISTS = {("skills", "paths"), ("templates", "paths"), ("instructions", "files")}
_PATH_VALUES = {("plugins", "directory"), ("coding", "workspace"), ("memory", "vectors")}
_ENTRY_PATHS = {"agents": "prompt_file", "tools": "path"}


def absolutize_layer(data: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """A copy of a project settings layer with its relative paths made absolute."""

    def absolute(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else root / path)

    result: dict[str, Any] = {key: (dict(value) if isinstance(value, Mapping) else value)
                              for key, value in data.items()}
    for section, key in _PATH_LISTS:
        table = result.get(section)
        if isinstance(table, dict) and isinstance(table.get(key), list):
            table[key] = [absolute(item) for item in table[key]]
    for section, key in _PATH_VALUES:
        table = result.get(section)
        if isinstance(table, dict) and key in table:
            table[key] = absolute(table[key])
    for section, key in _ENTRY_PATHS.items():
        entries = result.get(section)
        if isinstance(entries, list):
            result[section] = [
                {**entry, key: absolute(entry[key])} if isinstance(entry, Mapping) and key in entry else entry
                for entry in entries
            ]
    return result
