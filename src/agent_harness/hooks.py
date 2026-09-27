"""Lifecycle hooks: user code that observes or steers runs at defined points.

Events and what a hook may return:

==============  ===========================================================
session_start   deny (run fails), context (added as a user message)
user_prompt     deny (prompt rejected), modify (``content``), context
pre_model       deny (run fails), context (appended to this call's request
                only - never persisted into history, so prefixes stay cached)
post_model      observe only
pre_tool        deny (result returned to the model), modify (``arguments``,
                re-validated by the tool runtime), context (added to result)
post_tool       context (added to the tool result)
stop            continue (``reason`` becomes a user message; bounded)
subagent_stop   context (added to the subagent's result for the parent)
==============  ===========================================================

Hooks never override hard guardrails, deny rules, or approval requirements:
"allow" means "no objection". Every hook decision is recorded as a
``hook_decided`` event, and a resumed run replays recorded tool decisions
instead of re-running hooks.

Two implementations: :class:`CallableHook` (in-process function) and
:class:`CommandHook` (external program, structured argv, JSON on stdin and
stdout, timeout, output limit, never a shell).
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

MAX_HOOK_OUTPUT_BYTES = 65_536


class HookEvent(StrEnum):
    SESSION_START = "session_start"
    USER_PROMPT = "user_prompt"
    PRE_MODEL = "pre_model"
    POST_MODEL = "post_model"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    STOP = "stop"
    SUBAGENT_STOP = "subagent_stop"


# Hooks that guard an action fail closed; observers fail open.
_FAIL_CLOSED = {HookEvent.SESSION_START, HookEvent.USER_PROMPT, HookEvent.PRE_TOOL}
_DECISIONS = {"allow", "deny", "modify", "continue"}


@dataclass(frozen=True, slots=True)
class HookInput:
    event: HookEvent
    run_id: str
    step: int = 0
    model_profile: str | None = None
    tool: str | None = None
    call_id: str | None = None
    arguments: Mapping[str, Any] | None = None
    result_status: str | None = None
    result: str | None = None
    prompt: str | None = None
    final_output: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {key: value for key, value in _fields(self).items() if value is not None}


@dataclass(frozen=True, slots=True)
class HookOutcome:
    decision: str = "allow"
    reason: str = ""
    arguments: Mapping[str, Any] | None = None
    content: str | None = None
    context: str = ""

    def __post_init__(self) -> None:
        if self.decision not in _DECISIONS:
            raise ValueError(f"unknown hook decision {self.decision!r}")


class Hook(Protocol):
    name: str
    event: HookEvent

    def applies_to(self, tool: str | None) -> bool: ...

    async def run(self, hook_input: HookInput) -> HookOutcome: ...


HookFunction = Callable[[HookInput], "HookOutcome | None | Awaitable[HookOutcome | None]"]


@dataclass(slots=True)
class CallableHook:
    name: str
    event: HookEvent
    function: HookFunction
    matcher: str = "*"
    on_error: str | None = None

    def applies_to(self, tool: str | None) -> bool:
        return tool is None or fnmatch.fnmatchcase(tool, self.matcher)

    async def run(self, hook_input: HookInput) -> HookOutcome:
        value = self.function(hook_input)
        if inspect.isawaitable(value):
            value = await value
        return value if isinstance(value, HookOutcome) else HookOutcome()


@dataclass(slots=True)
class CommandHook:
    """Run ``argv`` with the hook input as JSON on stdin; read a JSON outcome.

    Exit code 0 with empty output means "allow". A non-zero exit, a timeout,
    oversized output, or invalid JSON is a hook error, handled per ``on_error``.
    """

    name: str
    event: HookEvent
    argv: Sequence[str]
    matcher: str = "*"
    timeout_seconds: float = 10.0
    on_error: str | None = None
    cwd: Path | None = None
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.argv or not all(isinstance(part, str) and part for part in self.argv):
            raise ValueError(f"hook {self.name!r} needs a non-empty argv list")
        if self.timeout_seconds <= 0:
            raise ValueError("hook timeout_seconds must be greater than zero")

    def applies_to(self, tool: str | None) -> bool:
        return tool is None or fnmatch.fnmatchcase(tool, self.matcher)

    async def run(self, hook_input: HookInput) -> HookOutcome:
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.cwd) if self.cwd else None,
            env=dict(self.env) if self.env is not None else dict(os.environ),
        )
        payload = json.dumps(hook_input.as_json(), default=str).encode()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                stdout, stderr = await process.communicate(payload)
        except TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise HookError(f"hook {self.name!r} timed out after {self.timeout_seconds}s") from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip()[:300]
            raise HookError(f"hook {self.name!r} exited with {process.returncode}: {detail}")
        if len(stdout) > MAX_HOOK_OUTPUT_BYTES:
            raise HookError(f"hook {self.name!r} output exceeds {MAX_HOOK_OUTPUT_BYTES} bytes")
        text = stdout.decode("utf-8", "replace").strip()
        if not text:
            return HookOutcome()
        try:
            raw = json.loads(text)
            if not isinstance(raw, dict):
                raise ValueError("output must be a JSON object")
            unknown = set(raw) - {"decision", "reason", "arguments", "content", "context"}
            if unknown:
                raise ValueError(f"unknown fields {sorted(unknown)}")
            return HookOutcome(
                decision=str(raw.get("decision", "allow")),
                reason=str(raw.get("reason", "")),
                arguments=raw.get("arguments") if isinstance(raw.get("arguments"), dict) else None,
                content=raw["content"] if isinstance(raw.get("content"), str) else None,
                context=str(raw.get("context", "")),
            )
        except ValueError as exc:
            raise HookError(f"hook {self.name!r} returned invalid output: {exc}") from exc


class HookError(Exception):
    """A hook failed to produce a usable outcome."""


@dataclass(frozen=True, slots=True)
class HookResult:
    """The combined outcome of every hook that applied to one invocation."""

    decision: str = "allow"
    reason: str = ""
    arguments: Mapping[str, Any] | None = None
    content: str | None = None
    context: str = ""
    hooks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ran(self) -> bool:
        return bool(self.hooks)

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision": self.decision,
            "hooks": list(self.hooks),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.arguments is not None:
            payload["arguments"] = dict(self.arguments)
        if self.content is not None:
            payload["content_modified"] = True
        if self.context:
            payload["context"] = self.context
        if self.errors:
            payload["errors"] = list(self.errors)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HookResult":
        arguments = payload.get("arguments")
        return cls(
            decision=str(payload.get("decision", "allow")),
            reason=str(payload.get("reason", "")),
            arguments=dict(arguments) if isinstance(arguments, Mapping) else None,
            context=str(payload.get("context", "")),
            hooks=tuple(payload.get("hooks", ())),
            errors=tuple(payload.get("errors", ())),
        )


@dataclass(slots=True)
class HookRunner:
    hooks: list[Hook] = field(default_factory=list)

    def add(self, hook: Hook) -> None:
        self.hooks.append(hook)

    def has(self, event: HookEvent) -> bool:
        return any(hook.event is event for hook in self.hooks)

    async def run(self, hook_input: HookInput) -> HookResult:
        """Run matching hooks in order. The first deny stops the chain;
        modifications feed forward; contexts accumulate."""

        current = hook_input
        names: list[str] = []
        errors: list[str] = []
        contexts: list[str] = []
        arguments: Mapping[str, Any] | None = None
        content: str | None = None
        continue_reason = ""
        for hook in self.hooks:
            if hook.event is not hook_input.event or not hook.applies_to(hook_input.tool):
                continue
            names.append(hook.name)
            try:
                outcome = await hook.run(current)
            except Exception as exc:  # noqa: BLE001 - hook failures follow on_error
                errors.append(f"{hook.name}: {exc}"[:300])
                policy = getattr(hook, "on_error", None) or (
                    "deny" if hook_input.event in _FAIL_CLOSED else "allow"
                )
                if policy == "deny":
                    return HookResult("deny", f"hook {hook.name!r} failed", hooks=tuple(names), errors=tuple(errors))
                continue
            if outcome.context:
                contexts.append(outcome.context)
            if outcome.decision == "deny":
                return HookResult(
                    "deny",
                    outcome.reason or f"blocked by hook {hook.name!r}",
                    hooks=tuple(names),
                    errors=tuple(errors),
                    context="\n".join(contexts),
                )
            if outcome.decision == "modify":
                if outcome.arguments is not None and hook_input.event is HookEvent.PRE_TOOL:
                    arguments = dict(outcome.arguments)
                    current = _replace(current, arguments=arguments)
                if outcome.content is not None and hook_input.event is HookEvent.USER_PROMPT:
                    content = outcome.content
                    current = _replace(current, prompt=content)
            if outcome.decision == "continue" and hook_input.event is HookEvent.STOP:
                continue_reason = outcome.reason or "Please continue."
        decision = "continue" if continue_reason else ("modify" if arguments is not None or content is not None else "allow")
        return HookResult(
            decision,
            continue_reason,
            arguments=arguments,
            content=content,
            context="\n".join(contexts),
            hooks=tuple(names),
            errors=tuple(errors),
        )


def _fields(value: Any) -> dict[str, Any]:
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def _replace(hook_input: HookInput, **changes: Any) -> HookInput:
    values = _fields(hook_input)
    values.update(changes)
    return HookInput(**values)
