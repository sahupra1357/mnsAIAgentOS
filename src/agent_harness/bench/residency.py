"""Local model residency for the validation pipeline.

Two local servers (Ollama and a llama.cpp router) share one machine's GPU memory.
Two large models loaded at once can run the GPU out of memory, so before a local
profile runs, large models loaded on the *other* local servers are unloaded and
the pipeline waits until they are gone. Before a remote profile (Modal, hosted)
and at the end of the pipeline, unloaded models are restored, but only models
that were already loaded when the pipeline started: models the pipeline itself
loaded are not brought back, so the machine ends as it began.

- Ollama: ``GET /api/ps`` lists loaded models with their parameter counts;
  ``POST /api/generate {"model", "keep_alive": 0}`` unloads one, and
  ``POST /api/generate {"model"}`` loads it again.
- llama.cpp router (``llama serve``): ``GET /models`` lists models with a status;
  ``POST /models/unload`` unloads one. Routers load models on their next request,
  so a restore is reported as "reloads on demand".
- Anything else (a single-model ``llama-server``, remote profiles) is skipped.

Off unless the plan enables it with ``[residency] enabled = true``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .core import BenchConfigError

LOCAL_ADAPTERS = {"ollama": "ollama", "llama_cpp": "llamacpp"}
_PARAMS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([BbMm])(?![A-Za-z])")


@dataclass(frozen=True, slots=True)
class ResidencyPolicy:
    enabled: bool = False
    min_params_b: float = 20.0  # "large" means more than this many billion parameters
    min_size_gb: float = 10.0  # used when the parameter count is unknown
    restore: bool = True
    unload_timeout_seconds: float = 60.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ResidencyPolicy":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise BenchConfigError(f"unknown [residency] fields: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for key, value in raw.items():
            if key in {"enabled", "restore"}:
                if not isinstance(value, bool):
                    raise BenchConfigError(f"[residency] {key} must be true or false")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise BenchConfigError(f"[residency] {key} must be a positive number")
            values[key] = float(value) if key not in {"enabled", "restore"} else value
        return cls(**values)


class ResidencyHttp(Protocol):
    """Blocking JSON HTTP calls; fakeable in tests."""

    def get_json(self, url: str, timeout: float) -> Any: ...

    def post_json(self, url: str, body: Mapping[str, Any], timeout: float) -> Any: ...


class UrllibResidencyHttp:
    def get_json(self, url: str, timeout: float) -> Any:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - configured local URL
            return json.loads(response.read() or b"null")

    def post_json(self, url: str, body: Mapping[str, Any], timeout: float) -> Any:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured local URL
            return json.loads(response.read() or b"null")


@dataclass(frozen=True, slots=True)
class LoadedModel:
    model: str
    params_b: float | None = None
    size_gb: float | None = None

    def describe(self) -> str:
        if self.params_b is not None:
            return f"{self.model} ({self.params_b:g}B)"
        if self.size_gb is not None:
            return f"{self.model} ({self.size_gb:.1f} GB)"
        return self.model


@dataclass(frozen=True, slots=True)
class LocalServer:
    kind: str  # "ollama" | "llamacpp"
    root: str  # base URL without the OpenAI /v1 suffix
    profiles: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.kind} ({self.root})"


@dataclass(frozen=True, slots=True)
class ResidencyNote:
    status: str  # "pass" | "skip" | "fail"
    detail: str


class ResidencyManager:
    """Unloads large models on other local servers before a local profile runs."""

    def __init__(
        self,
        profiles: Mapping[str, Any],
        policy: ResidencyPolicy,
        *,
        http: ResidencyHttp | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        file_size: Callable[[str], int | None] | None = None,
    ) -> None:
        self._profiles = profiles
        self._policy = policy
        self._http = http or UrllibResidencyHttp()
        self._sleep = sleep
        self._monotonic = monotonic
        self._file_size = file_size or _file_size
        self._servers = _local_servers(profiles)
        self._unloaded: list[tuple[LocalServer, LoadedModel]] = []
        # Models loaded before the pipeline touched anything: the only ones restored.
        self._initial: set[tuple[str, str]] | None = None

    @property
    def unloaded(self) -> tuple[tuple[str, str], ...]:
        return tuple((server.kind, model.model) for server, model in self._unloaded)

    def prepare(self, profile_name: str) -> list[ResidencyNote]:
        """Make room for ``profile_name``: unload (local) or restore (remote)."""

        self._snapshot()
        profile = self._profiles.get(profile_name)
        if profile is None:
            return [ResidencyNote("skip", f"unknown profile {profile_name!r}; nothing changed")]
        kind = LOCAL_ADAPTERS.get(getattr(profile, "adapter", ""))
        if kind is None:
            if self._policy.restore and self._unloaded:
                return self.restore(f"before remote profile {profile_name}")
            return []
        target = _root(profile.endpoint)
        notes: list[ResidencyNote] = []
        for server in self._servers:
            if server.root == target:
                continue  # never touch the server under test
            notes.extend(self._unload_large(server, profile_name))
        return notes

    def restore(self, reason: str) -> list[ResidencyNote]:
        notes: list[ResidencyNote] = []
        pending, self._unloaded = self._unloaded, []
        for server, model in pending:
            if server.kind == "llamacpp":
                notes.append(ResidencyNote(
                    "pass", f"{model.model} on {server.label} reloads on demand ({reason})"))
                continue
            try:
                self._http.post_json(f"{server.root}/api/generate", {"model": model.model}, timeout=600.0)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                notes.append(ResidencyNote("fail", f"could not restore {model.model} on {server.label}: {exc}"))
                continue
            notes.append(ResidencyNote("pass", f"restored {model.model} on {server.label} ({reason})"))
        return notes

    # -- internals ------------------------------------------------------------------

    def _unload_large(self, server: LocalServer, before: str) -> list[ResidencyNote]:
        try:
            loaded = self._loaded(server)
        except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
            return [ResidencyNote("skip", f"cannot list loaded models on {server.label}: {_brief(exc)}")]
        notes: list[ResidencyNote] = []
        for model in loaded:
            if not self._large(model):
                continue
            try:
                self._unload(server, model)
                self._wait_until_unloaded(server, model)
            except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError, TimeoutError) as exc:
                notes.append(ResidencyNote("fail", f"could not unload {model.model} on {server.label}: {_brief(exc)}"))
                continue
            assert self._initial is not None
            if (server.root, model.model) not in self._initial:
                notes.append(ResidencyNote("pass", f"unloaded {model.describe()} on {server.label} before "
                                                   f"{before} (loaded during this pipeline; not restored)"))
                continue
            if all(m.model != model.model or s != server for s, m in self._unloaded):
                self._unloaded.append((server, model))
            notes.append(ResidencyNote("pass", f"unloaded {model.describe()} on {server.label} before {before}"))
        return notes

    def _snapshot(self) -> None:
        if self._initial is not None:
            return
        self._initial = set()
        for server in self._servers:
            try:
                self._initial.update((server.root, m.model) for m in self._loaded(server))
            except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
                continue  # reported when the server is actually used

    def _large(self, model: LoadedModel) -> bool:
        if model.params_b is not None:
            return model.params_b > self._policy.min_params_b
        return model.size_gb is not None and model.size_gb >= self._policy.min_size_gb

    def _loaded(self, server: LocalServer) -> list[LoadedModel]:
        if server.kind == "ollama":
            data = self._http.get_json(f"{server.root}/api/ps", timeout=10.0)
            return [
                LoadedModel(
                    model=str(item["name"]),
                    params_b=_params_b(str((item.get("details") or {}).get("parameter_size", ""))),
                    size_gb=_gb(item.get("size_vram") or item.get("size")),
                )
                for item in (data or {}).get("models") or []
            ]
        data = self._http.get_json(f"{server.root}/models", timeout=10.0)
        models = []
        for item in (data or {}).get("data") or []:
            status = item.get("status")
            state = status.get("value") if isinstance(status, Mapping) else None
            if state is None:
                raise ValueError("server does not report model status (not a llama.cpp router)")
            if state not in {"loaded", "loading"}:
                continue
            args = status.get("args") or []
            path = _arg_value(args, "--model") or _arg_value(args, "-m")
            size = self._file_size(path) if path else None
            models.append(LoadedModel(
                model=str(item["id"]),
                params_b=_params_b(str(item["id"])) or (_params_b(os.path.basename(path)) if path else None),
                size_gb=_gb(size),
            ))
        return models

    def _unload(self, server: LocalServer, model: LoadedModel) -> None:
        if server.kind == "ollama":
            self._http.post_json(f"{server.root}/api/generate",
                                 {"model": model.model, "keep_alive": 0}, timeout=60.0)
        else:
            self._http.post_json(f"{server.root}/models/unload", {"model": model.model}, timeout=60.0)

    def _wait_until_unloaded(self, server: LocalServer, model: LoadedModel) -> None:
        deadline = self._monotonic() + self._policy.unload_timeout_seconds
        while any(m.model == model.model for m in self._loaded(server)):
            if self._monotonic() >= deadline:
                raise TimeoutError(f"still loaded after {self._policy.unload_timeout_seconds:g}s")
            self._sleep(1.0)


def _local_servers(profiles: Mapping[str, Any]) -> tuple[LocalServer, ...]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for name, profile in profiles.items():
        kind = LOCAL_ADAPTERS.get(getattr(profile, "adapter", ""))
        endpoint = getattr(profile, "endpoint", "")
        if kind is None or not endpoint:
            continue
        grouped.setdefault((kind, _root(endpoint)), []).append(name)
    return tuple(LocalServer(kind, root, tuple(names)) for (kind, root), names in grouped.items())


def _root(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _params_b(text: str) -> float | None:
    """Largest parameter count named in ``text`` ("20.9B", "Qwen3.8-27B" -> 27.0)."""

    values = [float(n) / (1000.0 if unit in "Mm" else 1.0) for n, unit in _PARAMS.findall(text)]
    return max(values) if values else None


def _gb(size: Any) -> float | None:
    return size / 2**30 if isinstance(size, (int, float)) and not isinstance(size, bool) and size > 0 else None


def _arg_value(args: Any, flag: str) -> str | None:
    if not isinstance(args, list):
        return None
    for index, arg in enumerate(args[:-1]):
        if arg == flag and isinstance(args[index + 1], str):
            return args[index + 1]
    return None


def _file_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)  # follows symlinks (Hugging Face cache layout)
    except OSError:
        return None


def _brief(error: BaseException) -> str:
    return (str(error) or type(error).__name__)[:200]
