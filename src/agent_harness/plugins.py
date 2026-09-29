"""Plugins: installable bundles of templates, skills, subagents, and hooks.

A plugin is a directory with a ``plugin.toml`` manifest::

    name = "code-review"
    version = "1.0.0"
    description = "Review diffs before committing."
    templates = "templates"          # optional directory of <name>.md templates
    skills = "skills"                # optional directory of <name>/SKILL.md skills

    [[agents]]                       # same fields as [[agents]] in settings;
    name = "reviewer"                # prompt_file is relative to the plugin
    description = "Reviews diffs."
    prompt_file = "agents/reviewer.md"

    [[hooks]]                        # same fields as [[hooks]]; commands run
    event = "pre_tool"               # with the plugin directory as cwd
    command = ["python3", "hooks/check.py"]

``install`` validates the manifest and copies the directory into the plugins
directory; nothing takes effect until the plugin's name is listed in
``[plugins] enabled``. Plugin hooks and agents are subject to the same
permission rules, approvals, and guardrails as everything else.
"""

from __future__ import annotations

import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.config import AgentConfig, ConfigError, HookConfig

MANIFEST = "plugin.toml"
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_FIELDS = {"name", "version", "description", "templates", "skills", "agents", "hooks"}


@dataclass(frozen=True, slots=True)
class Plugin:
    name: str
    version: str
    description: str
    root: Path
    templates_dir: Path | None
    skills_dir: Path | None
    agents: tuple[AgentConfig, ...]
    hooks: tuple[HookConfig, ...]


def load_plugin(root: Path) -> Plugin:
    manifest = root / MANIFEST
    try:
        raw = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"invalid plugin at {root}: {exc}") from exc
    unknown = set(raw) - _FIELDS
    if unknown:
        raise ConfigError(f"plugin {root.name}: unknown manifest fields {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ConfigError(f"plugin {root.name}: name must be lowercase letters, digits, _ or -")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ConfigError(f"plugin {name}: version is required")

    def subdirectory(key: str) -> Path | None:
        value = raw.get(key)
        if value is None:
            return None
        path = (root / str(value)).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_dir():
            raise ConfigError(f"plugin {name}: {key} must be a directory inside the plugin")
        return path

    try:
        agents = tuple(AgentConfig(**_checked(entry, AgentConfig)) for entry in raw.get("agents", []))
        hooks = tuple(
            HookConfig(**{**_checked(entry, HookConfig), "name": entry.get("name") or f"{name}:{entry.get('event')}"})
            for entry in raw.get("hooks", [])
        )
    except TypeError as exc:
        raise ConfigError(f"plugin {name}: {exc}") from exc
    for agent in agents:
        if agent.prompt_file:
            prompt = (root / agent.prompt_file).resolve()
            if not prompt.is_relative_to(root.resolve()) or not prompt.is_file():
                raise ConfigError(f"plugin {name}: agent prompt_file must be a file inside the plugin")
    return Plugin(
        name=name,
        version=version,
        description=str(raw.get("description", "")),
        root=root,
        templates_dir=subdirectory("templates"),
        skills_dir=subdirectory("skills"),
        agents=agents,
        hooks=hooks,
    )


def _checked(entry: Any, kind: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ConfigError("agents and hooks must be tables")
    unknown = set(entry) - set(kind.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"unknown fields {sorted(unknown)}")
    if "timeout_seconds" in entry:
        entry = {**entry, "timeout_seconds": float(entry["timeout_seconds"])}
    return dict(entry)


def install_plugin(source: Path, plugins_dir: Path, *, replace: bool = False) -> Plugin:
    """Validate ``source`` and copy it to ``plugins_dir/<name>`` (not enabled)."""

    plugin = load_plugin(source)
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ConfigError(f"plugin {plugin.name}: symbolic links are not allowed ({path})")
    target = plugins_dir / plugin.name
    if target.exists():
        if not replace:
            raise ConfigError(f"plugin {plugin.name} is already installed (use --replace)")
        shutil.rmtree(target)
    plugins_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    return load_plugin(target)


def installed_plugins(plugins_dir: Path) -> list[Plugin]:
    if not plugins_dir.is_dir():
        return []
    return [load_plugin(path) for path in sorted(plugins_dir.iterdir()) if (path / MANIFEST).is_file()]


def enabled_plugins(plugins_dir: Path, enabled: tuple[str, ...]) -> list[Plugin]:
    plugins = []
    for name in enabled:
        root = plugins_dir / name
        if not (root / MANIFEST).is_file():
            raise ConfigError(f"enabled plugin {name!r} is not installed in {plugins_dir}")
        plugins.append(load_plugin(root))
    return plugins
