"""``[guardrails]`` configuration (off unless ``enabled = true``).

    [guardrails]
    enabled = true
    refusal_message = "I can't help with that request."

    [guardrails.input]            # user text: prompts, follow-ups, steers, ask_user answers
    injection = "block"           # instruction overrides, role-play jailbreaks, role tags
    prompt_extraction = "block"   # "reveal your system prompt"
    hidden_characters = "redact"  # zero-width, bidi, Unicode tag characters (removed)
    encoded_payloads = "block"    # base64/hex that decodes to an injection
    flooding = "block"            # longer than max_chars, or long repetitions
    max_chars = 20000
    pii = "allow"                 # email, phone, card (Luhn), SSN, IBAN
    denylist = []                 # regexes; action set by denylist_action
    denylist_action = "block"

    [guardrails.output]           # the model's replies
    canary = "block"              # system prompt leak (a marker added to the system prompt)
    secrets = "redact"
    pii = "redact"
    denylist = []
    denylist_action = "block"
    allow_streaming = false       # stream previews only after the guard (off: no streaming)

    [guardrails.tools]            # untrusted tool output (MCP, web search, HTTP tools)
    injection = "redact"
    hidden_characters = "redact"
    taint = true                  # after untrusted content, side-effect tools need approval

    [guardrails.model]            # optional safety classifier, e.g. Llama Guard 3 on Ollama
    profile = ""                  # empty: off
    check = ["input", "output"]
    on_error = "allow"            # "allow" (flagged) or "block" when the guard model fails
    parallel = false              # check input alongside the main model call

Actions: ``allow`` (ignore), ``flag`` (record an event), ``redact`` (remove or mask
the matched text), ``block`` (refuse: the user gets ``refusal_message``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

ACTIONS = ("allow", "flag", "redact", "block")


class GuardrailsConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class InputGuardConfig:
    injection: str = "block"
    prompt_extraction: str = "block"
    hidden_characters: str = "redact"
    encoded_payloads: str = "block"
    flooding: str = "block"
    max_chars: int = 20_000
    pii: str = "allow"
    denylist: tuple[str, ...] = ()
    denylist_action: str = "block"


@dataclass(frozen=True, slots=True)
class OutputGuardConfig:
    canary: str = "block"
    secrets: str = "redact"
    pii: str = "redact"
    denylist: tuple[str, ...] = ()
    denylist_action: str = "block"
    allow_streaming: bool = False


@dataclass(frozen=True, slots=True)
class ToolGuardConfig:
    injection: str = "redact"
    hidden_characters: str = "redact"
    taint: bool = True


@dataclass(frozen=True, slots=True)
class ModelGuardConfig:
    profile: str = ""
    check: tuple[str, ...] = ("input", "output")
    on_error: str = "allow"
    parallel: bool = False


@dataclass(frozen=True, slots=True)
class GuardrailsConfig:
    enabled: bool = False
    refusal_message: str = "I can't help with that request."
    input: InputGuardConfig = field(default_factory=InputGuardConfig)
    output: OutputGuardConfig = field(default_factory=OutputGuardConfig)
    tools: ToolGuardConfig = field(default_factory=ToolGuardConfig)
    model: ModelGuardConfig = field(default_factory=ModelGuardConfig)

    @classmethod
    def from_mapping(cls, raw: Any) -> "GuardrailsConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise GuardrailsConfigError("guardrails must be a TOML table")
        sections = {"input": InputGuardConfig, "output": OutputGuardConfig,
                    "tools": ToolGuardConfig, "model": ModelGuardConfig}
        unknown = set(raw) - {"enabled", "refusal_message", *sections}
        if unknown:
            raise GuardrailsConfigError(f"unknown fields in guardrails: {', '.join(sorted(unknown))}")
        values: dict[str, Any] = {}
        if "enabled" in raw:
            values["enabled"] = _bool(raw["enabled"], "guardrails.enabled")
        if "refusal_message" in raw:
            message = raw["refusal_message"]
            if not isinstance(message, str) or not message.strip():
                raise GuardrailsConfigError("guardrails.refusal_message must be a non-empty string")
            values["refusal_message"] = message
        for name, section_type in sections.items():
            values[name] = _section(raw.get(name), f"guardrails.{name}", section_type)
        config = cls(**values)
        for check in config.model.check:
            if check not in {"input", "output"}:
                raise GuardrailsConfigError('guardrails.model.check entries must be "input" or "output"')
        if config.model.on_error not in {"allow", "block"}:
            raise GuardrailsConfigError('guardrails.model.on_error must be "allow" or "block"')
        return config


def _section(raw: Any, name: str, section_type: Any) -> Any:
    if raw is None:
        return section_type()
    if not isinstance(raw, Mapping):
        raise GuardrailsConfigError(f"{name} must be a TOML table")
    defaults = section_type()
    known = {f.name for f in fields(section_type)}
    unknown = set(raw) - known
    if unknown:
        raise GuardrailsConfigError(f"unknown fields in {name}: {', '.join(sorted(unknown))}")
    values: dict[str, Any] = {}
    for key, value in raw.items():
        default = getattr(defaults, key)
        where = f"{name}.{key}"
        if isinstance(default, bool):
            values[key] = _bool(value, where)
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GuardrailsConfigError(f"{where} must be a non-negative integer")
            values[key] = value
        elif isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise GuardrailsConfigError(f"{where} must be a list of strings")
            if key == "denylist":
                for pattern in value:
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        raise GuardrailsConfigError(f"{where}: invalid pattern {pattern!r}: {exc}") from exc
            values[key] = tuple(value)
        elif key in {"profile", "on_error"}:
            if not isinstance(value, str):
                raise GuardrailsConfigError(f"{where} must be a string")
            values[key] = value
        else:  # an action
            if value not in ACTIONS:
                raise GuardrailsConfigError(f"{where} must be one of: {', '.join(ACTIONS)}")
            values[key] = value
    return section_type(**{**{f.name: getattr(defaults, f.name) for f in fields(section_type)}, **values})


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise GuardrailsConfigError(f"{where} must be true or false")
    return value
