"""Layered settings: user -> project -> local, with enterprise enforced last.

Layers (missing files are skipped):

==========  ================================================  ==================
layer       default location (override with env variable)     typical use
==========  ================================================  ==================
user        ~/.agent-harness/settings.toml                    personal defaults
            (AGENT_HARNESS_USER_SETTINGS)
project     the --config file (e.g. models.toml)              shared, committed
local       <config>.local.toml next to it                    personal, gitignored
enterprise  /etc/agent-harness/settings.toml                  managed policy
            (AGENT_HARNESS_ENTERPRISE_SETTINGS)
==========  ================================================  ==================

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

LAYER_ORDER = ("user", "project", "local", "enterprise")


class SettingsError(ValueError):
    """A settings layer could not be read."""


@dataclass(slots=True)
class LayeredSettings:
    raw: dict[str, Any]
    layers: list[tuple[str, Path]] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)  # dotted key -> layer(s)


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


def load_layers(config_path: Path | None, environ: Mapping[str, str]) -> LayeredSettings:
    settings = LayeredSettings(raw={})
    for name, path in layer_paths(config_path, environ):
        if not path.is_file():
            if name == "project" and config_path is not None:
                raise SettingsError(f"could not load config {path}: file not found")
            continue
        try:
            with path.open("rb") as file:
                data = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise SettingsError(f"could not load {name} settings {path}: {exc}") from exc
        merge_into(settings.raw, data, layer=name, provenance=settings.provenance)
        settings.layers.append((name, path))
    return settings


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
