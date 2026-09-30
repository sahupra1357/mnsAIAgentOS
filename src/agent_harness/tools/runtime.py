"""Validated, policy-aware, bounded tool execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from agent_harness.domain import (
    IdempotencyStore,
    JsonObject,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
)

from .policy import DefaultToolPolicy, ToolPolicy
from .registry import ToolRegistry
from .validation import ArgumentValidationError, validate_arguments


class RetryableToolError(Exception):
    """A tool failed in a way that may succeed when called again."""


class PermanentToolError(Exception):
    """A tool rejected execution and should not be retried unchanged."""


class ToolRejectedError(Exception):
    """A policy-safe validation message that may be returned to the model."""


class ToolRuntime:
    """Execute registered tools behind validation, policy, and resource limits."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: ToolPolicy | None = None,
        max_output_bytes: int = 65_536,
        idempotency_store: IdempotencyStore | None = None,
    ) -> None:
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero")
        self._registry = registry
        self._policy = policy or DefaultToolPolicy()
        self._max_output_bytes = max_output_bytes
        self._idempotency_results: dict[tuple[str, str], tuple[str, ToolResult]] = {}
        self._idempotency_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._idempotency_lock = asyncio.Lock()
        self._idempotency_store = idempotency_store
        self._run_finalizers: list[Any] = []
        self._closers: list[Any] = []

    def add_run_finalizer(self, finalizer: Any) -> None:
        """``async finalizer(run_id)``: release per-run resources (e.g. a warm container)."""

        self._run_finalizers.append(finalizer)

    def add_closer(self, closer: Any) -> None:
        """``async closer()``: release everything when the application closes."""

        self._closers.append(closer)

    async def release_run(self, run_id: str) -> None:
        for finalizer in self._run_finalizers:
            try:
                await finalizer(run_id)
            except Exception:  # cleanup must never fail a finished run
                continue

    async def close(self) -> None:
        for closer in self._closers:
            try:
                await closer()
            except Exception:
                continue

    def attach_idempotency_store(self, store: IdempotencyStore) -> None:
        """Back the in-process result cache with durable storage."""

        self._idempotency_store = store

    def register(
        self,
        spec: ToolSpec,
        handler: Any,
        *,
        source: str = "builtin",
        source_kind: str = "builtin",
        replace: bool = False,
    ) -> None:
        """Explicitly add a capability to this runtime's registry."""

        self._registry.register(
            spec, handler, source=source, source_kind=source_kind, replace=replace
        )

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._registry.specs()

    async def execute(
        self,
        call: ToolCall,
        *,
        idempotency_key: str,
        approved: bool = False,
    ) -> ToolResult:
        invalid_call = _validate_call_envelope(call, idempotency_key)
        if invalid_call is not None:
            return invalid_call

        registered = self._registry.resolve(call.name)
        if registered is None:
            return _result(
                call,
                ToolResultStatus.PERMANENT_ERROR,
                "Unknown tool.",
                error="unknown_tool",
            )

        try:
            _assert_json_value(call.arguments)
            validate_arguments(call.arguments, registered.spec.input_schema)
        except (ArgumentValidationError, TypeError, ValueError) as exc:
            return _result(
                call,
                ToolResultStatus.VALIDATION_ERROR,
                f"Invalid tool arguments: {exc}",
                error="invalid_arguments",
            )

        decision = self._policy.evaluate(registered.spec, approved=approved)
        if not decision.allowed:
            status = (
                ToolResultStatus.APPROVAL_REQUIRED
                if decision.requires_approval
                else ToolResultStatus.POLICY_DENIED
            )
            return _result(
                call,
                status,
                decision.reason or "Tool execution was denied by policy.",
                error=(
                    "approval_required"
                    if decision.requires_approval
                    else "policy_denied"
                ),
            )

        if registered.spec.risk is not ToolRisk.READ_ONLY and not idempotency_key.strip():
            return _result(
                call,
                ToolResultStatus.VALIDATION_ERROR,
                "A non-empty idempotency key is required for side-effecting tools.",
                error="idempotency_key_required",
            )

        cache_key = (call.name, idempotency_key) if idempotency_key.strip() else None
        if cache_key is None:
            return await self._invoke(call, registered.spec.timeout_seconds, registered.handler)

        fingerprint = json.dumps(
            call.arguments,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key_lock = await self._lock_for(cache_key)
        async with key_lock:
            cached = self._idempotency_results.get(cache_key)
            if cached is None and self._idempotency_store is not None:
                cached = await self._idempotency_store.get_tool_result(*cache_key)
                if cached is not None:
                    self._idempotency_results[cache_key] = cached
            if cached is not None:
                cached_fingerprint, cached_result = cached
                if cached_fingerprint != fingerprint:
                    return _result(
                        call,
                        ToolResultStatus.VALIDATION_ERROR,
                        "The idempotency key was already used with different arguments.",
                        error="idempotency_key_conflict",
                    )
                return _for_call(cached_result, call.id)

            result = await self._invoke(call, registered.spec.timeout_seconds, registered.handler)
            if result.status is ToolResultStatus.SUCCESS:
                self._idempotency_results[cache_key] = (fingerprint, result)
                if self._idempotency_store is not None:
                    await self._idempotency_store.put_tool_result(
                        cache_key[0], cache_key[1], fingerprint, result
                    )
            return result

    async def _lock_for(self, cache_key: tuple[str, str]) -> asyncio.Lock:
        async with self._idempotency_lock:
            return self._idempotency_locks.setdefault(cache_key, asyncio.Lock())

    async def _invoke(self, call: ToolCall, timeout: float, handler: Any) -> ToolResult:
        try:
            raw_output = await asyncio.wait_for(handler(dict(call.arguments)), timeout=timeout)
            output = _normalize_output(raw_output)
            encoded = json.dumps(
                output,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(encoded) > self._max_output_bytes:
                return _result(
                    call,
                    ToolResultStatus.PERMANENT_ERROR,
                    "Tool output exceeded the configured size limit.",
                    error="output_too_large",
                    limit_bytes=self._max_output_bytes,
                )
            return ToolResult(
                call_id=call.id,
                name=call.name,
                status=ToolResultStatus.SUCCESS,
                output=output,
            )
        except TimeoutError:
            return _result(
                call,
                ToolResultStatus.TIMEOUT,
                "Tool execution timed out.",
                retryable=True,
                error="timeout",
            )
        except RetryableToolError:
            return _result(
                call,
                ToolResultStatus.RETRYABLE_ERROR,
                "Tool execution failed temporarily.",
                retryable=True,
                error="retryable_failure",
            )
        except PermanentToolError:
            return _result(
                call,
                ToolResultStatus.PERMANENT_ERROR,
                "Tool execution failed permanently.",
                error="permanent_failure",
            )
        except ToolRejectedError as exc:
            message = str(exc).strip()[:500] or "Tool request was rejected."
            return _result(
                call,
                ToolResultStatus.PERMANENT_ERROR,
                message,
                error="request_rejected",
            )
        except (TypeError, ValueError, OverflowError):
            return _result(
                call,
                ToolResultStatus.PERMANENT_ERROR,
                "Tool returned an invalid result.",
                error="invalid_output",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _result(
                call,
                ToolResultStatus.PERMANENT_ERROR,
                "Tool execution failed.",
                error="tool_failure",
            )


def _normalize_output(value: Any) -> JsonObject:
    if isinstance(value, Mapping):
        output = dict(value)
    else:
        output = {"result": value}
    _assert_json_value(output)
    return output


def _validate_call_envelope(call: ToolCall, idempotency_key: str) -> ToolResult | None:
    call_id = call.id if isinstance(call.id, str) else ""
    name = call.name if isinstance(call.name, str) else ""
    if not call_id.strip():
        return ToolResult(
            call_id=call_id,
            name=name,
            status=ToolResultStatus.VALIDATION_ERROR,
            output={"error": "invalid_call_id"},
            model_message="Tool call ID must be a non-empty string.",
        )
    if not name.strip():
        return ToolResult(
            call_id=call_id,
            name=name,
            status=ToolResultStatus.VALIDATION_ERROR,
            output={"error": "invalid_tool_name"},
            model_message="Tool name must be a non-empty string.",
        )
    if not isinstance(idempotency_key, str):
        return ToolResult(
            call_id=call_id,
            name=name,
            status=ToolResultStatus.VALIDATION_ERROR,
            output={"error": "invalid_idempotency_key"},
            model_message="Idempotency key must be a string.",
        )
    return None


def _assert_json_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _assert_json_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("object keys must be strings")
        for item in value.values():
            _assert_json_value(item)
        return
    raise TypeError("value is not JSON-compatible")


def _result(
    call: ToolCall,
    status: ToolResultStatus,
    message: str,
    *,
    retryable: bool = False,
    **output: Any,
) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        name=call.name,
        status=status,
        output=output,
        model_message=message,
        retryable=retryable,
    )


def _for_call(result: ToolResult, call_id: str) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        name=result.name,
        status=result.status,
        output=dict(result.output),
        model_message=result.model_message,
        retryable=result.retryable,
    )
