"""HTTP API as a dependency-free ASGI application.

The API validates input at the boundary, creates and inspects durable runs,
records approval decisions, and enqueues work; queue workers execute runs.
Serve it with any ASGI server (``agent-harness serve`` uses uvicorn from the
optional ``[api]`` extra), which can also host in-process workers through the
ASGI lifespan protocol.

Endpoints (all JSON; every route except ``/healthz`` and ``/readyz`` requires
``Authorization: Bearer <key>`` unless auth was explicitly disabled). Keys map to
a principal with a tenant and a role (viewer < operator < admin; see
``agent_harness.access``); a principal sees only its tenant's runs:

    GET  /healthz                   liveness
    GET  /readyz                    store reachable, schema version
    GET  /profiles
    GET  /templates
    POST /runs                      {"prompt", "system"?, "model_profile"?, "budget"?, "mode"?,
                                     "planning"?}
    GET  /runs/{id}
    GET  /runs/{id}/events?after=N
    GET  /runs/{id}/events/stream?after=N   (server-sent events)
    POST /runs/{id}/cancel
    POST /runs/{id}/messages       {"content", "budget"?}  (continue a completed run)
    POST /runs/{id}/steer          {"content"}  (message for a run in progress)
    GET  /runs/{id}/usage
    POST /runs/{id}/resume
    GET  /runs/{id}/approval
    POST /runs/{id}/approval        {"call_id", "decision", "feedback"?, "mode"?}
    POST /runs/{id}/mode            {"mode": "default" | "accept_edits" | "plan" | "bypass"}
    POST /runs/{id}/rewind          {"to_step", "confirm"?}  (dry run unless confirm)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qs

from agent_harness.access import Principal, PrincipalDirectory, Quota, QuotaExceeded, check_quota
from agent_harness.access import Role as AccessRole
from agent_harness.config import ConfigError, HarnessConfig
from agent_harness.domain import (
    ArtifactStore,
    EventType,
    HarnessError,
    Message,
    PermissionMode,
    Role,
    RunBudget,
    RunQueue,
    RunRecord,
    RunStatus,
    RunStore,
)
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.planning import plan_payload
from agent_harness.templates import Template, TemplateError, expand_prompt
from agent_harness.worker import Worker, run_is_settled


Scope = Mapping[str, Any]
Receive = Callable[[], Awaitable[Mapping[str, Any]]]
Send = Callable[[Mapping[str, Any]], Awaitable[None]]

MAX_TEXT_CHARS = 100_000
_ID = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
_RUN_ROUTE = re.compile(rf"^/runs/(?P<run_id>{_ID})(?P<rest>/.*)?$")
_ARTIFACT_ROUTE = re.compile(r"^/artifacts/(?P<artifact_id>[0-9a-f]{64})$")
_BUDGET_FIELDS = {"max_steps", "max_wall_seconds", "max_tokens", "max_cost_usd"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class HarnessAPI:
    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        store: RunStore,
        queue: RunQueue,
        config: HarnessConfig,
        auth_token: str | None,
        run_versions: Callable[[str], Mapping[str, str]] | None = None,
        default_system_prompt: str | None = None,
        workers: Sequence[Worker] = (),
        stream_poll_seconds: float = 0.5,
        max_body_bytes: int = 1_000_000,
        shutdown_grace_seconds: float = 10.0,
        artifacts: ArtifactStore | None = None,
        templates: Mapping[str, Template] | None = None,
        principals: PrincipalDirectory | None = None,
        quotas: Mapping[str, Quota] | None = None,
    ) -> None:
        if auth_token is not None and len(auth_token) < 16:
            raise ValueError("API auth token must be at least 16 characters")
        self._principals = principals or PrincipalDirectory((), legacy_token=auth_token)
        if principals is not None and auth_token is not None:
            self._principals = PrincipalDirectory(principals._keys, legacy_token=auth_token)
        self._quotas = dict(quotas or {})
        self._orchestrator = orchestrator
        self._store = store
        self._queue = queue
        self._config = config
        self._token = auth_token
        self._run_versions = run_versions
        self._default_system = default_system_prompt
        self._workers = tuple(workers)
        self._stream_poll = stream_poll_seconds
        self._max_body = max_body_bytes
        self._grace = shutdown_grace_seconds
        self._artifacts = artifacts
        self._templates = dict(templates or {})
        self._stop = asyncio.Event()
        self._worker_tasks: list[asyncio.Task[None]] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        try:
            await self._dispatch(scope, receive, send)
        except ApiError as exc:
            await _json(send, exc.status, {"error": exc.message})
        except QuotaExceeded as exc:  # admission is reserved atomically by the orchestrator
            await _json(send, 429, {"error": str(exc)})
        except KeyError as exc:
            await _json(send, 404, {"error": str(exc.args[0]) if exc.args else "not found"})
        except (ValueError, ConfigError) as exc:
            await _json(send, 400, {"error": str(exc)})
        except HarnessError as exc:
            await _json(send, 409, {"error": str(exc)})

    # -- lifespan: in-process workers ------------------------------------------------

    async def start_workers(self) -> None:
        self._stop.clear()
        self._worker_tasks = [
            asyncio.create_task(worker.run_forever(self._stop)) for worker in self._workers
        ]

    async def stop_workers(self) -> None:
        self._stop.set()
        if not self._worker_tasks:
            return
        _, pending = await asyncio.wait(self._worker_tasks, timeout=self._grace)
        for task in pending:  # cancelled runs are put back on the queue
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks = []

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await self.start_workers()
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.stop_workers()
                await send({"type": "lifespan.shutdown.complete"})
                return

    # -- routing ---------------------------------------------------------------------

    async def _dispatch(self, scope: Scope, receive: Receive, send: Send) -> None:
        method = scope["method"]
        path = scope["path"]
        if path == "/healthz" and method == "GET":
            await _json(send, 200, {"ok": True})
            return
        if path == "/readyz" and method == "GET":
            await self._ready(send)
            return
        principal = self._authorize(scope)
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))

        if path == "/profiles" and method == "GET":
            await _json(send, 200, self._profiles())
            return
        if path == "/templates" and method == "GET":
            await _json(send, 200, {
                name: {"description": t.description, "sha256": t.sha256}
                for name, t in sorted(self._templates.items())
            })
            return
        if path == "/runs" and method == "POST":
            _require(principal, AccessRole.OPERATOR)
            await _json(send, 202, await self._create_run(await self._body(receive), principal))
            return
        artifact = _ARTIFACT_ROUTE.match(path)
        if artifact is not None and method == "GET":
            await self._send_artifact(send, artifact["artifact_id"])
            return
        match = _RUN_ROUTE.match(path)
        if match is None:
            raise ApiError(404, "not found")
        run_id, rest = match["run_id"], match["rest"] or ""
        _require(principal, AccessRole.VIEWER if method == "GET"
                 else AccessRole.ADMIN if rest in {"/mode", "/rewind"} else AccessRole.OPERATOR)
        owned = await self._store.get_run(run_id)
        if owned is None or not principal.can_access(owned.tenant):
            raise ApiError(404, f"unknown run: {run_id}")  # other tenants' runs do not exist
        routes: dict[tuple[str, str], Callable[[], Awaitable[None]]] = {
            ("GET", ""): lambda: self._send_run(send, run_id),
            ("GET", "/events"): lambda: self._send_events(send, run_id, query),
            ("GET", "/events/stream"): lambda: self._stream_events(send, run_id, query),
            ("POST", "/cancel"): lambda: self._cancel(send, run_id),
            ("POST", "/messages"): lambda: self._continue(send, run_id, receive, owned.tenant),
            ("POST", "/steer"): lambda: self._steer(send, run_id, receive),
            ("GET", "/usage"): lambda: self._send_usage(send, run_id),
            ("POST", "/mode"): lambda: self._set_mode(send, run_id, receive, principal),
            ("POST", "/rewind"): lambda: self._rewind(send, run_id, receive),
            ("POST", "/resume"): lambda: self._resume(send, run_id),
            ("GET", "/approval"): lambda: self._send_approval(send, run_id),
            ("POST", "/approval"): lambda: self._decide(send, run_id, receive, principal),
        }
        handler = routes.get((method, rest))
        if handler is None:
            raise ApiError(404 if not any(r == rest for _, r in routes) else 405, "not found")
        await handler()

    def _authorize(self, scope: Scope) -> Principal:
        if not self._principals.enabled:
            return Principal("anonymous", "*", AccessRole.ADMIN)  # auth explicitly disabled
        header = next(
            (value for name, value in scope.get("headers", []) if name.lower() == b"authorization"),
            b"",
        ).decode("latin-1")
        scheme, _, supplied = header.partition(" ")
        principal = self._principals.resolve(supplied.strip()) if scheme.lower() == "bearer" else None
        if principal is None:
            raise ApiError(401, "missing or invalid bearer token")
        return principal

    async def _ready(self, send: Send) -> None:
        try:
            await self._store.get_run("readyz-probe")
            version = getattr(self._store, "schema_version", None)
            payload: dict[str, Any] = {"ok": True}
            if callable(version):
                payload["schema_version"] = version()
        except Exception as exc:  # any store failure means not ready
            await _json(send, 503, {"ok": False, "error": type(exc).__name__})
            return
        await _json(send, 200, payload)

    async def _body(self, receive: Receive) -> dict[str, Any]:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self._max_body:
                raise ApiError(413, "request body too large")
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        try:
            parsed = json.loads(b"".join(chunks) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"invalid JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ApiError(400, "JSON body must be an object")
        return parsed

    # -- handlers --------------------------------------------------------------------

    def _profiles(self) -> dict[str, Any]:
        return {
            "default": self._config.default_model_profile,
            "profiles": {
                name: {
                    "adapter": profile.adapter,
                    "model": profile.model,
                    "context_window": profile.context_window,
                    "fallback": list(profile.fallback),
                }
                for name, profile in self._config.models.items()
            },
            "routes": dict(self._config.routes),
        }

    async def _create_run(self, body: dict[str, Any], principal: Principal) -> dict[str, Any]:
        unknown = set(body) - {"prompt", "system", "model_profile", "budget", "mode", "template_args",
                               "planning"}
        if unknown:
            raise ApiError(400, f"unknown fields: {', '.join(sorted(unknown))}")
        prompt = _required_text(body.get("prompt"), "prompt")
        named = body.get("template_args") or {}
        if not isinstance(named, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in named.items()
        ):
            raise ApiError(400, "template_args must map strings to strings")
        try:
            prompt, template = expand_prompt(prompt, self._templates, named)
        except TemplateError as exc:
            raise ApiError(400, str(exc)) from exc
        system = _text(body.get("system"), "system", required=False) or self._default_system
        requested = body.get("model_profile")
        if requested is not None and not isinstance(requested, str):
            raise ApiError(400, "model_profile must be a string")
        profile = self._config.resolve_model_name(requested)
        route = requested if requested and requested != profile else None
        tenant = principal.creation_tenant
        try:
            await check_quota(self._store, self._quotas, tenant)
        except QuotaExceeded as exc:
            raise ApiError(429, str(exc)) from exc
        messages = [Message(role=Role.USER, content=prompt)]
        if system:
            messages.insert(0, Message(role=Role.SYSTEM, content=system))
        run = await self._orchestrator.create_run(
            model_profile=profile,
            route=route,
            messages=messages,
            budget=_budget(body.get("budget")),
            versions=self._run_versions(profile) if self._run_versions else None,
            mode=_mode(body.get("mode")),
            template={"name": template.name, "sha256": template.sha256} if template else None,
            planning=_optional_bool(body.get("planning"), "planning"),
            tenant=tenant,
            created_by=principal.name,
        )
        await self._queue.enqueue(run.id)
        return {"id": run.id, "status": run.status.value, "model_profile": profile}

    async def _load(self, run_id: str) -> RunRecord:
        run = await self._store.get_run(run_id)
        if run is None:
            raise ApiError(404, f"unknown run: {run_id}")
        return run

    async def _send_run(self, send: Send, run_id: str) -> None:
        run = await self._load(run_id)
        payload: dict[str, Any] = {
            "id": run.id,
            "status": run.status.value,
            "model_profile": run.model_profile,
            "step_count": run.step_count,
            "consumed_tokens": run.consumed_tokens,
            "consumed_cost_usd": run.consumed_cost_usd,
            "created_at": run.created_at.isoformat(),
            "updated_at": run.updated_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error": run.error,
            "output": _final_output(run),
            "mode": run.mode.value,
            "parent_run_id": run.parent_run_id,
            "depth": run.depth,
            "todos": [
                {"id": t.id, "content": t.content, "status": t.status.value} for t in run.todos
            ],
            "subagents": [
                {"run_id": e.payload["child_run_id"], "agent": e.payload.get("agent")}
                for e in await self._store.list_events(run_id)
                if e.type is EventType.SUBAGENT_STARTED
            ],
            "plan": plan_payload(run.plan) if run.plan is not None else None,
        }
        if run.status is RunStatus.AWAITING_INPUT:
            payload["pending_approval"] = await self._approval_payload(run_id)
        await _json(send, 200, payload)

    async def _send_events(self, send: Send, run_id: str, query: Mapping[str, list[str]]) -> None:
        await self._load(run_id)
        after = _after(query)
        events = list(await self._store.list_events(run_id))
        await _json(
            send,
            200,
            {
                "events": [_event_payload(i, e) for i, e in enumerate(events) if i >= after],
                "next": len(events),
            },
        )

    async def _stream_events(self, send: Send, run_id: str, query: Mapping[str, list[str]]) -> None:
        await self._load(run_id)
        cursor = _after(query)
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        while True:
            events = list(await self._store.list_events(run_id))
            for index in range(cursor, len(events)):
                event = events[index]
                data = json.dumps(_event_payload(index, event), sort_keys=True, default=str)
                chunk = f"id: {index}\nevent: {event.type.value}\ndata: {data}\n\n"
                await send({"type": "http.response.body", "body": chunk.encode(), "more_body": True})
            cursor = len(events)
            run = await self._load(run_id)
            if run_is_settled(run):
                end = json.dumps({"status": run.status.value})
                await send({"type": "http.response.body", "body": f"event: end\ndata: {end}\n\n".encode()})
                return
            await asyncio.sleep(self._stream_poll)

    async def _send_artifact(self, send: Send, artifact_id: str) -> None:
        data = await self._artifacts.get(artifact_id) if self._artifacts else None
        if data is None:
            raise ApiError(404, "unknown artifact")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"cache-control", b"no-store"),
                    (b"content-length", str(len(data)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": data})

    async def _continue(self, send: Send, run_id: str, receive: Receive, tenant: str) -> None:
        try:
            await check_quota(self._store, self._quotas, tenant, new_run=False)
        except QuotaExceeded as exc:
            raise ApiError(429, str(exc)) from exc
        body = await self._body(receive)
        unknown = set(body) - {"content", "budget"}
        if unknown:
            raise ApiError(400, f"unknown fields: {', '.join(sorted(unknown))}")
        content = _required_text(body.get("content"), "content")
        current = await self._load(run_id)
        raised = _budget(body.get("budget"))
        budget = None
        if raised is not None:
            # Only the supplied limits change; the rest keep the run's values.
            supplied = body["budget"]
            budget = dataclasses.replace(
                current.budget, **{name: getattr(raised, name) for name in supplied}
            )
        try:
            run = await self._orchestrator.continue_run(run_id, content, budget=budget)
        except ValueError as exc:
            raise ApiError(409, str(exc)) from exc
        await self._queue.enqueue(run_id)
        await _json(send, 202, {"id": run.id, "status": run.status.value})

    async def _steer(self, send: Send, run_id: str, receive: Receive) -> None:
        body = await self._body(receive)
        unknown = set(body) - {"content"}
        if unknown:
            raise ApiError(400, f"unknown fields: {', '.join(sorted(unknown))}")
        content = _required_text(body.get("content"), "content")
        run = await self._load(run_id)
        try:
            steer = await self._orchestrator.steer(run_id, content)
        except ValueError as exc:
            raise ApiError(409, str(exc)) from exc
        if run.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
            await self._queue.enqueue(run_id)  # no-op if already queued or leased
        await _json(send, 202, {"id": run_id, "steer_id": steer.id, "status": run.status.value})

    async def _send_usage(self, send: Send, run_id: str) -> None:
        await self._load(run_id)
        usage = await self._orchestrator.usage(run_id)
        payload = {
            field: getattr(usage, field) for field in usage.__dataclass_fields__
        }
        payload["status"] = usage.status.value
        payload["todos"] = [
            {"id": t.id, "content": t.content, "status": t.status.value} for t in usage.todos
        ]
        await _json(send, 200, payload)

    async def _set_mode(self, send: Send, run_id: str, receive: Receive, principal: Principal) -> None:
        body = await self._body(receive)
        if set(body) != {"mode"}:
            raise ApiError(400, 'body must be {"mode": ...}')
        await self._load(run_id)
        target = _mode(body["mode"])
        assert target is not None
        try:
            run = await self._orchestrator.set_mode(run_id, target, actor=principal.name)
        except ValueError as exc:
            raise ApiError(409, str(exc)) from exc
        await _json(send, 200, {"id": run.id, "mode": run.mode.value})

    async def _rewind(self, send: Send, run_id: str, receive: Receive) -> None:
        body = await self._body(receive)
        unknown = set(body) - {"to_step", "confirm"}
        if unknown:
            raise ApiError(400, f"unknown fields: {', '.join(sorted(unknown))}")
        to_step = body.get("to_step")
        if isinstance(to_step, bool) or not isinstance(to_step, int):
            raise ApiError(400, "to_step must be an integer")
        confirm = body.get("confirm", False)
        if not isinstance(confirm, bool):
            raise ApiError(400, "confirm must be a boolean")
        await self._load(run_id)
        try:
            plan = await self._orchestrator.rewind(run_id, to_step, confirm=confirm)
        except ValueError as exc:
            raise ApiError(409, str(exc)) from exc
        await _json(send, 200, dataclasses.asdict(plan))

    async def _cancel(self, send: Send, run_id: str) -> None:
        await self._load(run_id)
        run = await self._orchestrator.cancel(run_id)
        await _json(send, 200, {"id": run.id, "status": run.status.value})

    async def _resume(self, send: Send, run_id: str) -> None:
        run = await self._load(run_id)
        if run.status.is_terminal:
            raise ApiError(409, f"run is {run.status.value}")
        if run.status is RunStatus.AWAITING_INPUT:
            raise ApiError(409, "run is awaiting an approval decision")
        await self._queue.enqueue(run_id)
        await _json(send, 202, {"id": run.id, "status": run.status.value})

    async def _approval_payload(self, run_id: str) -> dict[str, Any] | None:
        pending = await self._orchestrator.pending_approval(run_id)
        if pending is None:
            return None
        return {
            "call_id": pending.call_id,
            "kind": pending.kind,
            "tool": pending.tool_name,
            "risk": pending.risk.value,
            "arguments": dict(pending.arguments),
            "requested_at": pending.requested_at.isoformat(),
        }

    async def _send_approval(self, send: Send, run_id: str) -> None:
        await self._load(run_id)
        approval = await self._approval_payload(run_id)
        if approval is None:
            raise ApiError(404, "run has no pending approval")
        await _json(send, 200, approval)

    async def _decide(self, send: Send, run_id: str, receive: Receive, principal: Principal) -> None:
        body = await self._body(receive)
        unknown = set(body) - {"call_id", "decision", "feedback", "mode"}
        if unknown:
            raise ApiError(400, f"unknown fields: {', '.join(sorted(unknown))}")
        decision = body.get("decision")
        if decision not in {"allow", "deny"}:
            raise ApiError(400, 'decision must be "allow" or "deny"')
        call_id = body.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ApiError(400, "call_id is required")
        await self._load(run_id)
        pending = await self._orchestrator.pending_approval(run_id)
        if pending is None:
            raise ApiError(409, "run has no pending approval")
        if pending.call_id != call_id:
            raise ApiError(409, f"pending call ID is {pending.call_id!r}")
        feedback = _text(body.get("feedback"), "feedback", required=False)
        try:
            run = await self._orchestrator.resolve_approval(
                run_id,
                call_id=call_id,
                approved=decision == "allow",
                feedback=feedback,
                mode=_mode(body.get("mode")),
                actor=principal.name,
            )
        except ValueError as exc:
            raise ApiError(409, str(exc)) from exc
        await self._queue.enqueue(run_id)
        await _json(send, 202, {"id": run.id, "status": run.status.value, "decision": decision})


def _require(principal: Principal, role: AccessRole) -> None:
    if principal.role < role:
        raise ApiError(403, f"this key's role ({principal.role.name.lower()}) cannot do that; "
                            f"{role.name.lower()} is required")


def _mode(value: Any) -> PermissionMode | None:
    if value is None:
        return None
    try:
        return PermissionMode(value)
    except ValueError as exc:
        modes = ", ".join(mode.value for mode in PermissionMode)
        raise ApiError(400, f"mode must be one of: {modes}") from exc


def _required_text(value: Any, field: str) -> str:
    text = _text(value, field, required=True)
    assert text is not None
    return text


def _text(value: Any, field: str, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ApiError(400, f"{field} is required")
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise ApiError(400, f"{field} must be a non-empty string")
    if len(value) > MAX_TEXT_CHARS:
        raise ApiError(400, f"{field} exceeds {MAX_TEXT_CHARS} characters")
    return value


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ApiError(400, f"{name} must be a boolean")


def _budget(raw: Any) -> RunBudget | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ApiError(400, "budget must be an object")
    unknown = set(raw) - _BUDGET_FIELDS
    if unknown:
        raise ApiError(400, f"unknown budget fields: {', '.join(sorted(unknown))}")
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ApiError(400, f"budget.{name} must be a positive number")
    if "max_steps" in raw and (not isinstance(raw["max_steps"], int) or raw["max_steps"] > 1000):
        raise ApiError(400, "budget.max_steps must be an integer up to 1000")
    if "max_tokens" in raw and not isinstance(raw["max_tokens"], int):
        raise ApiError(400, "budget.max_tokens must be an integer")
    if raw.get("max_wall_seconds", 0) > 86_400:
        raise ApiError(400, "budget.max_wall_seconds must be at most 86400")
    return RunBudget(**raw)


def _after(query: Mapping[str, list[str]]) -> int:
    values = query.get("after", ["0"])
    try:
        after = int(values[-1])
    except ValueError as exc:
        raise ApiError(400, "after must be an integer") from exc
    if after < 0:
        raise ApiError(400, "after must not be negative")
    return after


def _final_output(run: RunRecord) -> str | None:
    if run.status is not RunStatus.COMPLETED:
        return None
    for message in reversed(run.messages):
        if message.role is Role.ASSISTANT and not message.tool_calls:
            return message.content
    return None


def _event_payload(index: int, event: Any) -> dict[str, Any]:
    return {
        "index": index,
        "id": event.id,
        "type": event.type.value,
        "step": event.step,
        "created_at": event.created_at.isoformat(),
        "payload": dict(event.payload),
    }


async def _json(send: Send, status: int, payload: Any) -> None:
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"cache-control", b"no-store"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_api(
    app: Any,
    *,
    auth_token: str | None,
    worker_count: int = 1,
    lease_seconds: float = 60.0,
    poll_interval_seconds: float = 1.0,
) -> HarnessAPI:
    """Build the API for a :class:`~agent_harness.application.HarnessApplication`."""

    from agent_harness.access import api_keys_from_config
    from agent_harness.coding import CODING_AGENT_SYSTEM_PROMPT

    if worker_count < 0:
        raise ValueError("worker_count must not be negative")
    workers = [
        Worker(
            app.orchestrator,
            app.store,
            lease_seconds=lease_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        for _ in range(worker_count)
    ]
    return HarnessAPI(
        orchestrator=app.orchestrator,
        store=app.store,
        queue=app.store,
        config=app.config,
        auth_token=auth_token,
        run_versions=app.run_versions,
        default_system_prompt=CODING_AGENT_SYSTEM_PROMPT if app.config.coding.enabled else None,
        workers=workers,
        artifacts=app.artifacts,
        templates=app.templates,
        principals=PrincipalDirectory(api_keys_from_config(app.config.api_keys), legacy_token=auth_token),
    )
