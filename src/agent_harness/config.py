"""Configuration loading for named model profiles.

Secrets are deliberately not resolved here.  Profiles hold the name of the
environment variable containing a credential, and provider adapters resolve it
only when they make a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tomllib
from types import MappingProxyType
from typing import Any, Mapping

from agent_harness.coding.config import CodingAgentConfig, parse_coding_config


_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HEADER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
# Headers the adapter owns; credentials for Authorization come from api_key_env.
_RESERVED_HEADERS = frozenset({"authorization", "content-type", "accept", "content-length", "host"})
SUPPORTED_SERVERS = frozenset({"vllm", "sglang"})
SUPPORTED_API_MODES = frozenset({"openai", "native"})
_ENV_PREFIX = "AGENT_HARNESS_MODEL__"
_PROFILE_FIELDS = {
    "ADAPTER": "adapter",
    "MODEL": "model",
    "ENDPOINT": "endpoint",
    "API_KEY_ENV": "api_key_env",
    "TIMEOUT_SECONDS": "timeout_seconds",
    "MAX_RETRIES": "max_retries",
    "RETRY_BACKOFF_SECONDS": "retry_backoff_seconds",
    "PARAMETERS": "parameters",
    "HEADERS_ENV": "headers_env",
    "CONTEXT_WINDOW": "context_window",
    "COLD_START_TIMEOUT_SECONDS": "cold_start_timeout_seconds",
    "SERVER": "server",
    "API_MODE": "api_mode",
    "PRICING": "pricing",
    "FALLBACK": "fallback",
}
_PRICING_FIELDS = (
    "input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
    "gpu_usd_per_second",
)


class ConfigError(ValueError):
    """Configuration is absent, malformed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Optional price inputs for cost estimation (enforced from Phase 3)."""

    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None
    gpu_usd_per_second: float | None = None

    def __post_init__(self) -> None:
        for name in _PRICING_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise ConfigError(f"pricing.{name} must be a non-negative number")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Configuration for one named model endpoint.

    ``headers_env`` maps extra request header names to the *names* of environment
    variables holding their values; values are resolved only at request time.
    ``cold_start_timeout_seconds`` extends the timeout until the first successful
    request, for scale-to-zero deployments. ``server`` identifies the inference
    server behind a hosted endpoint, and ``api_mode`` selects a provider's native
    API where an adapter supports one. ``pricing`` and ``fallback`` are validated
    here and consumed by cost estimation and fallback routing.
    """

    adapter: str
    model: str
    endpoint: str
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25
    parameters: Mapping[str, Any] = field(default_factory=dict)
    headers_env: Mapping[str, str] = field(default_factory=dict)
    context_window: int | None = None
    cold_start_timeout_seconds: float | None = None
    server: str | None = None
    api_mode: str = "openai"
    pricing: ModelPricing | None = None
    fallback: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter.strip():
            raise ConfigError("model profile adapter must not be empty")
        if not self.model.strip():
            raise ConfigError("model profile model must not be empty")
        if not self.endpoint.strip():
            raise ConfigError("model profile endpoint must not be empty")
        if self.timeout_seconds <= 0:
            raise ConfigError("model profile timeout_seconds must be greater than zero")
        if not 0 <= self.max_retries <= 10:
            raise ConfigError("model profile max_retries must be between 0 and 10")
        if self.retry_backoff_seconds < 0:
            raise ConfigError("model profile retry_backoff_seconds must not be negative")
        if self.api_key_env is not None and not self.api_key_env.strip():
            object.__setattr__(self, "api_key_env", None)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "headers_env", MappingProxyType(_validated_headers_env(self.headers_env)))
        if self.context_window is not None and (
            isinstance(self.context_window, bool)
            or not isinstance(self.context_window, int)
            or self.context_window <= 0
        ):
            raise ConfigError("model profile context_window must be a positive integer")
        if self.cold_start_timeout_seconds is not None and self.cold_start_timeout_seconds <= 0:
            raise ConfigError("model profile cold_start_timeout_seconds must be greater than zero")
        if self.server is not None and self.server not in SUPPORTED_SERVERS:
            raise ConfigError(
                f"model profile server must be one of: {', '.join(sorted(SUPPORTED_SERVERS))}"
            )
        if self.api_mode not in SUPPORTED_API_MODES:
            raise ConfigError(
                f"model profile api_mode must be one of: {', '.join(sorted(SUPPORTED_API_MODES))}"
            )
        fallback = tuple(self.fallback)
        for name in fallback:
            if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
                raise ConfigError(f"invalid fallback profile name: {name!r}")
        if len(set(fallback)) != len(fallback):
            raise ConfigError("model profile fallback must not repeat a profile")
        object.__setattr__(self, "fallback", fallback)


def _validated_headers_env(raw: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    seen: set[str] = set()
    for header, env_name in dict(raw).items():
        if not isinstance(header, str) or not _HEADER_NAME.fullmatch(header):
            raise ConfigError(f"invalid header name in headers_env: {header!r}")
        folded = header.casefold()
        if folded in _RESERVED_HEADERS:
            raise ConfigError(
                f"headers_env must not set {header!r}; use api_key_env for bearer credentials"
            )
        if folded in seen:
            raise ConfigError(f"duplicate header in headers_env: {header!r}")
        if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
            raise ConfigError(f"headers_env[{header!r}] must name an environment variable")
        seen.add(folded)
        validated[header] = env_name
    return validated


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Validated application configuration."""

    models: Mapping[str, ModelProfile]
    default_model_profile: str
    coding: CodingAgentConfig = field(default_factory=CodingAgentConfig)

    def __post_init__(self) -> None:
        copied = dict(self.models)
        if not copied:
            raise ConfigError("at least one model profile is required")
        for name in copied:
            if not _PROFILE_NAME.fullmatch(name):
                raise ConfigError(f"invalid model profile name: {name!r}")
        if self.default_model_profile not in copied:
            raise ConfigError(
                f"default model profile {self.default_model_profile!r} is not configured"
            )
        for name, profile in copied.items():
            for target in profile.fallback:
                if target == name:
                    raise ConfigError(f"model profile {name!r} cannot fall back to itself")
                if target not in copied:
                    raise ConfigError(
                        f"model profile {name!r} falls back to unknown profile {target!r}"
                    )
        object.__setattr__(self, "models", MappingProxyType(copied))

    def model_profile(self, name: str | None = None) -> ModelProfile:
        """Return a named profile, or the configured default."""

        selected = name or self.default_model_profile
        try:
            return self.models[selected]
        except KeyError as exc:
            raise ConfigError(f"unknown model profile: {selected!r}") from exc


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> HarnessConfig:
    """Load TOML configuration and apply environment overrides.

    If *path* is omitted, ``AGENT_HARNESS_CONFIG`` may name the TOML file.
    Profiles can also be defined entirely through variables of the form
    ``AGENT_HARNESS_MODEL__PROFILE__FIELD``. Profile names in environment
    variables are normalized to lower case.
    """

    env = os.environ if environ is None else environ
    configured_path = path or env.get("AGENT_HARNESS_CONFIG")
    raw: dict[str, Any] = {}
    if configured_path:
        config_path = Path(configured_path)
        try:
            with config_path.open("rb") as config_file:
                parsed = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"could not load config {config_path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError("configuration root must be a table")
        raw = parsed

    raw_models = raw.get("models", {})
    if not isinstance(raw_models, dict):
        raise ConfigError("models must be a TOML table")
    models: dict[str, dict[str, Any]] = {}
    for name, values in raw_models.items():
        if not isinstance(values, dict):
            raise ConfigError(f"models.{name} must be a TOML table")
        models[name] = dict(values)

    _apply_environment_overrides(models, env)
    profiles = {name: _parse_profile(name, values) for name, values in models.items()}

    default = env.get("AGENT_HARNESS_DEFAULT_MODEL_PROFILE") or raw.get(
        "default_model_profile"
    )
    if default is None:
        if len(profiles) == 1:
            default = next(iter(profiles))
        else:
            raise ConfigError("default_model_profile is required when multiple profiles exist")
    if not isinstance(default, str) or not default:
        raise ConfigError("default_model_profile must be a non-empty string")
    coding_raw = raw.get("coding")
    if coding_raw is not None and not isinstance(coding_raw, dict):
        raise ConfigError("coding must be a TOML table")
    try:
        coding = parse_coding_config(coding_raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return HarnessConfig(
        models=profiles,
        default_model_profile=default,
        coding=coding,
    )


def _apply_environment_overrides(
    models: dict[str, dict[str, Any]], env: Mapping[str, str]
) -> None:
    for key, value in env.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        remainder = key[len(_ENV_PREFIX) :]
        parts = remainder.split("__", 1)
        if len(parts) != 2 or not all(parts):
            raise ConfigError(f"invalid model environment variable: {key}")
        profile_token, field_token = parts
        field_name = _PROFILE_FIELDS.get(field_token)
        if field_name is None:
            raise ConfigError(f"unknown model configuration field in {key}")
        profile_name = profile_token.lower()
        profile = models.setdefault(profile_name, {})
        profile[field_name] = _parse_environment_value(key, field_name, value)


def _parse_environment_value(key: str, field_name: str, value: str) -> Any:
    try:
        if field_name in {"timeout_seconds", "retry_backoff_seconds", "cold_start_timeout_seconds"}:
            return float(value)
        if field_name in {"max_retries", "context_window"}:
            return int(value)
        if field_name in {"parameters", "headers_env", "pricing"}:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("must decode to an object")
            return parsed
        if field_name == "fallback":
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("must decode to an array")
            return parsed
        return value
    except (ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid value for {key}: {exc}") from exc


def _parse_profile(name: str, raw: Mapping[str, Any]) -> ModelProfile:
    allowed = {
        "adapter",
        "model",
        "endpoint",
        "api_key_env",
        "timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
        "parameters",
        "headers_env",
        "context_window",
        "cold_start_timeout_seconds",
        "server",
        "api_mode",
        "pricing",
        "fallback",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown fields in models.{name}: {', '.join(sorted(unknown))}")
    missing = [field for field in ("adapter", "model", "endpoint") if field not in raw]
    if missing:
        raise ConfigError(f"models.{name} is missing: {', '.join(missing)}")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ConfigError(f"models.{name}.parameters must be a table/object")
    headers_env = raw.get("headers_env", {})
    if not isinstance(headers_env, dict):
        raise ConfigError(f"models.{name}.headers_env must be a table/object")
    fallback = raw.get("fallback", [])
    if not isinstance(fallback, list):
        raise ConfigError(f"models.{name}.fallback must be an array")
    context_window = raw.get("context_window")
    if context_window is not None and (
        isinstance(context_window, bool) or not isinstance(context_window, int)
    ):
        raise ConfigError(f"models.{name}.context_window must be an integer")
    cold_start = raw.get("cold_start_timeout_seconds")
    try:
        return ModelProfile(
            adapter=_required_string(raw["adapter"], f"models.{name}.adapter"),
            model=_required_string(raw["model"], f"models.{name}.model"),
            endpoint=_required_string(raw["endpoint"], f"models.{name}.endpoint"),
            api_key_env=_optional_string(raw.get("api_key_env"), f"models.{name}.api_key_env"),
            timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
            max_retries=int(raw.get("max_retries", 2)),
            retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 0.25)),
            parameters=parameters,
            headers_env=headers_env,
            context_window=context_window,
            cold_start_timeout_seconds=None if cold_start is None else float(cold_start),
            server=_optional_string(raw.get("server"), f"models.{name}.server"),
            api_mode=_required_string(raw.get("api_mode", "openai"), f"models.{name}.api_mode"),
            pricing=_parse_pricing(name, raw.get("pricing")),
            fallback=tuple(fallback),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid models.{name}: {exc}") from exc


def _parse_pricing(name: str, raw: Any) -> ModelPricing | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"models.{name}.pricing must be a table/object")
    unknown = set(raw) - set(_PRICING_FIELDS)
    if unknown:
        raise ConfigError(
            f"unknown fields in models.{name}.pricing: {', '.join(sorted(unknown))}"
        )
    return ModelPricing(**raw)


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty string")
    return value


def _optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{location} must be a string")
    return value or None
