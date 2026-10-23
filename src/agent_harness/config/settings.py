"""Layered settings: user -> project -> local, with enterprise enforced last.

Layers (missing files are skipped):

==========  ================================================  ==================
layer       default location (override with env variable)     typical use
==========  ================================================  ==================
user        ~/.agent-harness/settings.toml                    personal defaults
            (AGENT_HARNESS_USER_SETTINGS)
project     the --config file (e.g. models.toml)              shared, committed
local       <config>.local.toml next to it                    personal, gitignored
workspace   <project>/.agent-harness/settings.toml            per project (Phase 24)
workspace-  <project>/.agent-harness/settings.local.toml      personal, per project
local
enterprise  /etc/agent-harness/settings.toml                  managed policy
            (AGENT_HARNESS_ENTERPRISE_SETTINGS)
==========  ================================================  ==================

The two workspace layers apply only to project runs (``[project] root = "cwd"``
or ``--workspace``, see ``config/project.py``) with ``[project] settings`` on;
relative paths in them resolve against the project root.

Merge rules: tables merge key by key; scalars from a later layer replace
earlier ones; lists concatenate (duplicates dropped), so permission rules,
hooks, and agents accumulate. Enterprise is applied *last*: its scalars win
and its deny rules can never be removed by other layers.

Relative paths inside settings (hook commands, prompt files, template and
skill directories) resolve against the project config's directory; use
absolute paths in user and enterprise layers.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAYER_ORDER = ("user", "project", "local", "workspace", "workspace-local", "enterprise")


class SettingsError(ValueError):
    """A settings layer could not be read."""


@dataclass(slots=True)
class LayeredSettings:
    raw: dict[str, Any]
    layers: list[tuple[str, Path]] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)  # dotted key -> layer(s)
    project_root: Path | None = None  # set for project runs (Phase 24)


def layer_paths(config_path: Path | None, environ: Mapping[str, str]) -> list[tuple[str, Path]]:
    home = Path(environ.get("HOME", str(Path.home())))
    user = Path(environ.get("AGENT_HARNESS_USER_SETTINGS", home / ".agent-harness" / "settings.toml"))
    enterprise = Path(
        environ.get("AGENT_HARNESS_ENTERPRISE_SETTINGS", "/etc/agent-harness/settings.toml")
    )
    paths: list[tuple[str, Path]] = [("user", user.expanduser())]
    if config_path is not None:
        paths.append(("project", config_path))
        paths.append(("local", config_path.with_name(config_path.stem + ".local.toml")))
    paths.append(("enterprise", enterprise.expanduser()))
    return paths


def load_layers(
    config_path: Path | None,
    environ: Mapping[str, str],
    *,
    workspace: Path | None = None,
    cwd: Path | None = None,
) -> LayeredSettings:
    """Merge the layers. Whether this is a project run is decided after the user,
    project, and local layers; the workspace layers then go before enterprise."""

    from .project import absolutize_layer, project_layer_paths, resolve_project_root

    settings = LayeredSettings(raw={})
    paths = layer_paths(config_path, environ)
    enterprise = paths.pop()
    for name, path in paths:
        _merge_layer(settings, name, path, required=name == "project" and config_path is not None)
    root = resolve_project_root(
        settings.raw, environ, config_dir=config_path.parent if config_path is not None else None,
        workspace=workspace, cwd=cwd,
    )
    settings.project_root = root
    project = settings.raw.get("project")
    if root is not None and not (isinstance(project, Mapping) and project.get("settings") is False):
        for name, path in project_layer_paths(root):
            _merge_layer(settings, name, path, transform=lambda data: absolutize_layer(data, root))
    _merge_layer(settings, *enterprise)
    return settings


def _merge_layer(
    settings: LayeredSettings, name: str, path: Path, *, required: bool = False, transform: Any = None
) -> None:
    if not path.is_file():
        if required:
            raise SettingsError(f"could not load config {path}: file not found")
        return
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SettingsError(f"could not load {name} settings {path}: {exc}") from exc
    merge_into(settings.raw, transform(data) if transform else data, layer=name,
               provenance=settings.provenance)
    settings.layers.append((name, path))


def merge_into(
    target: dict[str, Any],
    source: Mapping[str, Any],
    *,
    layer: str,
    provenance: dict[str, str],
    prefix: str = "",
) -> None:
    for key, value in source.items():
        dotted = f"{prefix}{key}"
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merge_into(current, value, layer=layer, provenance=provenance, prefix=dotted + ".")
        elif isinstance(current, list) and isinstance(value, list):
            for item in value:
                if item not in current:
                    current.append(item)
            provenance[dotted] = f"{provenance.get(dotted, '?')}+{layer}"
        elif isinstance(value, Mapping):
            target[key] = {}
            merge_into(target[key], value, layer=layer, provenance=provenance, prefix=dotted + ".")
        else:
            target[key] = list(value) if isinstance(value, list) else value
            provenance[dotted] = layer


def render_effective(settings: LayeredSettings) -> str:
    """Human-readable effective settings with the layer each value came from."""

    lines = ["# layers: " + ", ".join(f"{name}={path}" for name, path in settings.layers)]
    if settings.project_root is not None:
        lines.append(f"# project: {settings.project_root}")

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{prefix}{key}" if not prefix else f"{prefix}.{key}")
            return
        source = settings.provenance.get(prefix, "?")
        shown = _mask(prefix, value)
        lines.append(f"{prefix} = {shown!r}    # from {source}")

    walk(settings.raw, "")
    return "\n".join(lines)


def _mask(key: str, value: Any) -> Any:
    # Settings hold env-variable *names*, not secrets, but be defensive.
    lowered = key.lower()
    if any(word in lowered for word in ("secret", "password", "token")) and not lowered.endswith("_env"):
        return "***"
    return value


def default_user_dir(environ: Mapping[str, str]) -> Path:
    return Path(environ.get("HOME", os.path.expanduser("~"))) / ".agent-harness"
