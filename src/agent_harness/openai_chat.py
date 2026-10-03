"""OpenAI-compatible chat endpoint (Phase 15d): use the harness as a "model".

``GET /v1/models`` lists the configured profiles and routes, and
``POST /v1/chat/completions`` runs the agent. Chat clients such as Open WebUI,
LibreChat, Chatbox, AnythingLLM, Continue, or any OpenAI SDK can then talk to
the harness. Tools, hooks, guardrails, and approvals all run server-side; the
client only sees text.

**Conversations.** OpenAI clients send the whole history on every request. The
harness keeps one durable run per conversation:

* an ``X-Agent-Harness-Run: RUN_ID`` header names the run explicitly;
* otherwise a digest of the history before the last message is looked up in
  the store (``ConversationStore``). A match continues that run; a miss starts
  a new run, seeded with the client's earlier turns. After each reply the
  digest of the history *including* that reply is saved for the next request.
  An edited or regenerated history no longer matches, so it starts a new run.

**Pauses.** A run waiting for an approval or an ``ask_user`` answer replies
with the question as plain text. The next user message resolves it:
``approve`` / ``yes`` allows a tool call or plan; ``deny`` / ``no`` (optionally
followed by a reason) denies it; any other text denies a tool call or plan
with that text as feedback, and is the answer to an ``ask_user`` question.

Each reply is the run's newest assistant answer. With ``"stream": true`` text
is sent as ``chat.completion.chunk`` server-sent events from in-process token
previews; the final answer is sent at the end when no preview carried it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from agent_harness.config import ConfigError, HarnessConfig
from agent_harness.domain import ContentPart, Message, Role, RunRecord, RunStatus
from agent_harness.orchestrator import AgentOrchestrator, validate_image_url
from agent_harness.streaming import StreamHub
from agent_harness.worker import run_is_settled

Send = Callable[[Mapping[str, Any]], Awaitable[None]]

# A model name that always means the default profile.
DEFAULT_MODEL_ALIAS = "agent-harness"
RUN_HEADER = b"x-agent-harness-run"
MAX_MESSAGES = 500
_APPROVE = {"approve", "approved", "allow", "yes", "y", "ok", "okay", "go", "proceed"}
_DENY = {"deny", "denied", "no", "n", "reject", "stop", "cancel"}
# Each chat turn gets this much more budget than the run has used so far.
TURN_STEPS = 20
TURN_WALL_SECONDS = 300.0


class ChatRequestError(ValueError):
    """The request is not a valid chat completion request (HTTP 400)."""


@dataclasses.dataclass(frozen=True, slots=True)
class ChatTurn:
    role: str  # system | user | assistant
    text: str
    images: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    turns: tuple[ChatTurn, ...]
    stream: bool = False
    user: str | None = None

    @property
    def system(self) -> str | None:
        texts = [turn.text for turn in self.turns if turn.role == "system" and turn.text]
        return "\n\n".join(texts) if texts else None

    @property
    def last(self) -> ChatTurn:
        return self.turns[-1]

    @property
    def history(self) -> tuple[ChatTurn, ...]:
        """Turns before the newest user message."""

        return self.turns[:-1]


def parse_chat_request(body: Mapping[str, Any]) -> ChatRequest:
    """Validate an OpenAI chat completion body. Sampling options are ignored: the
    profile's parameters apply. Client ``tools`` are not supported."""

    if body.get("tools") or body.get("functions"):
        raise ChatRequestError("client-side tools are not supported; the harness runs its own tools")
    model = body.get("model") or DEFAULT_MODEL_ALIAS
    if not isinstance(model, str):
        raise ChatRequestError("model must be a string")
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw or len(raw) > MAX_MESSAGES:
        raise ChatRequestError(f"messages must be a non-empty list of at most {MAX_MESSAGES} messages")
    turns = []
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise ChatRequestError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            continue  # tool / function messages belong to client-side tools; not used here
        text, images = _content(message.get("content"), index)
        turns.append(ChatTurn(role=role, text=text, images=images))
    if not turns or turns[-1].role != "user" or not (turns[-1].text.strip() or turns[-1].images):
        raise ChatRequestError("the last message must be a non-empty user message")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise ChatRequestError("stream must be true or false")
    user = body.get("user")
    if user is not None and (not isinstance(user, str) or not user.strip()):
        raise ChatRequestError("user must be a non-empty string")
    return ChatRequest(model=model, turns=tuple(turns), stream=stream, user=user)


def _content(value: Any, index: int) -> tuple[str, tuple[str, ...]]:
    if value is None:
        return "", ()
    if isinstance(value, str):
        return value, ()
    if not isinstance(value, list):
        raise ChatRequestError(f"messages[{index}].content must be a string or a list of parts")
    texts: list[str] = []
    images: list[str] = []
    for part in value:
        if not isinstance(part, dict):
            raise ChatRequestError(f"messages[{index}].content parts must be objects")
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str):
                raise ChatRequestError(f"messages[{index}]: image_url needs a url")
            try:
                validate_image_url(url)
            except ValueError as exc:
                raise ChatRequestError(f"messages[{index}]: {exc}") from exc
            images.append(url)
    return "\n".join(texts), tuple(images)


def conversation_key(turns: Sequence[ChatTurn]) -> str:
    """Digest of a conversation as the client sees it (whitespace-trimmed text)."""

    canonical = [[turn.role, turn.text.strip()] for turn in turns]
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False).encode("utf-8")).hexdigest()


def pause_text(pending: Any) -> str:
    """What a paused run says in a plain chat window."""

    if pending.kind == "question":
        lines = [str(pending.arguments.get("question", "")).strip()]
        options = pending.arguments.get("options") or ()
        if options:
            lines.append("Options: " + ", ".join(str(option) for option in options))
        lines.append("(Reply with your answer.)")
        return "\n\n".join(lines)
    if pending.kind == "plan":
        plan = str(pending.arguments.get("plan", "")).strip()
        return (f"Plan ready for approval:\n\n{plan}\n\n"
                "(Reply **approve** to carry it out, or reply with what to change.)")
    arguments = json.dumps(dict(pending.arguments), indent=2, ensure_ascii=False)
    return (f"Approval needed: `{pending.tool_name}` ({pending.risk.value}).\n\n```json\n{arguments}\n```\n\n"
            "(Reply **approve** or **deny**, optionally with a reason.)")


def decision_from_text(text: str) -> tuple[bool, str | None]:
    """``(approved, feedback)`` from a chat reply to an approval request."""

    words = text.strip().split(maxsplit=1)
    first = words[0].lower().strip(".!,:") if words else ""
    rest = words[1].strip() if len(words) > 1 else None
    if first in _APPROVE:
        return True, None
    if first in _DENY:
        return False, rest or None
    return False, text.strip() or None


def answer_text(run: RunRecord) -> str:
    """The answer to the newest user message (``output`` as JSON for structured runs).

    Only messages after the newest user message count, so an empty reply is never
    replaced by an earlier turn's answer.
    """

    if run.output is not None:
        return json.dumps(run.output, indent=2, ensure_ascii=False)
    for message in reversed(run.messages):
        if message.role is Role.USER:
            break
        if message.role is Role.ASSISTANT and not message.tool_calls and message.content:
            return message.content
    return ""


class OpenAIChat:
    """The ``/v1`` routes, built by ``HarnessAPI`` with its own dependencies."""

    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        store: Any,
        queue: Any,
        config: HarnessConfig,
        run_versions: Callable[[str], Mapping[str, str]] | None,
        default_system: str | None,
        hub: StreamHub | None,
        poll_seconds: float,
    ) -> None:
        self._orchestrator = orchestrator
        self._store = store
        self._queue = queue
        self._config = config
        self._run_versions = run_versions
        self._default_system = default_system
        self._hub = hub
        self._poll = poll_seconds

    def models(self) -> dict[str, Any]:
        names = [DEFAULT_MODEL_ALIAS, *self._config.chat_models(), *self._config.routes]
        return {"object": "list", "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "agent-harness"} for name in names
        ]}

    def _profile(self, model: str) -> tuple[str, str | None]:
        if model == DEFAULT_MODEL_ALIAS:
            return self._config.resolve_model_name(None), None
        try:
            profile = self._config.resolve_model_name(model)
        except ConfigError as exc:
            raise ChatRequestError(f"unknown model {model!r}; GET /v1/models lists them") from exc
        return profile, model if model != profile else None

    async def complete(
        self, request: ChatRequest, send: Send, *, tenant: str, actor: str, run_header: str | None,
        can_access: Callable[[str], bool],
    ) -> None:
        profile, route = self._profile(request.model)
        run_id = await self._start_turn(request, profile, route, tenant=tenant, actor=actor,
                                        run_header=run_header, can_access=can_access)
        created = int(time.time())
        reply_id = f"chatcmpl-{run_id}-{created}"
        if request.stream:
            # Subscribed to previews before the run is queued, so no early text is missed.
            sent = await self._stream(send, run_id, reply_id, created, request.model)
        else:
            await self._queue.enqueue(run_id)
            run = await self._settle(run_id)
            sent = await self._reply_text(run)
            payload = {
                "id": reply_id, "object": "chat.completion", "created": created, "model": request.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": sent},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": run.consumed_tokens},
                "agent_harness": {"run_id": run.id, "status": run.status.value},
            }
            await _send_json(send, 200, payload, run_id)
        seen = [*request.turns, ChatTurn("assistant", sent)]
        save = getattr(self._store, "save_conversation", None)
        if save is not None:
            with contextlib.suppress(KeyError):  # a run removed meanwhile simply is not remembered
                await save(tenant, conversation_key(seen), run_id)

    async def _start_turn(
        self, request: ChatRequest, profile: str, route: str | None, *, tenant: str, actor: str,
        run_header: str | None, can_access: Callable[[str], bool],
    ) -> str:
        """Continue, resolve, or create the run for this request; returns its ID.

        The caller enqueues it (after subscribing to its stream). A created run is
        already marked for queue workers (``dispatch = "queue"``).
        """

        find = getattr(self._store, "find_conversation", None)  # optional store capability
        run_id = run_header or (await find(tenant, conversation_key(request.history)) if find else None)
        run = await self._store.get_run(run_id) if run_id else None
        if run_header and (run is None or not can_access(run.tenant)):
            raise KeyError(f"unknown run: {run_header}")
        if run is not None and not can_access(run.tenant):
            run = None
        last = request.last
        if run is not None and run.status is RunStatus.AWAITING_INPUT:
            pending = await self._orchestrator.pending_approval(run.id)
            if pending is not None:
                if pending.kind == "question":
                    await self._orchestrator.answer(run.id, call_id=pending.call_id, answer=last.text.strip())
                else:
                    approved, feedback = decision_from_text(last.text)
                    await self._orchestrator.resolve_approval(
                        run.id, call_id=pending.call_id, approved=approved, feedback=feedback, actor=actor)
                return run.id
        if run is not None and run.status in {RunStatus.COMPLETED, RunStatus.BUDGET_EXHAUSTED}:
            usage = await self._orchestrator.usage(run.id)
            budget = dataclasses.replace(
                run.budget, max_steps=usage.steps + TURN_STEPS,
                max_wall_seconds=usage.active_seconds + TURN_WALL_SECONDS)
            await self._orchestrator.continue_run(run.id, last.text, budget=budget, images=list(last.images))
            return run.id
        if run is not None and not run_is_settled(run):
            raise ChatRequestError(f"run {run.id} is still working; wait for its reply")
        return await self._create(request, profile, route, tenant=tenant, actor=actor)

    async def _create(self, request: ChatRequest, profile: str, route: str | None, *, tenant: str,
                      actor: str) -> str:
        messages: list[Message] = []
        system = request.system or self._default_system
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))
        # A history the harness does not know (new client, edited conversation) seeds the run.
        for turn in request.turns:
            if turn.role == "system" or not (turn.text or turn.images):
                continue
            messages.append(Message(
                role=Role.USER if turn.role == "user" else Role.ASSISTANT,
                content=turn.text,
                parts=tuple(ContentPart(image_url=url) for url in turn.images) if turn.role == "user" else (),
            ))
        run = await self._orchestrator.create_run(
            model_profile=profile, route=route, messages=messages,
            versions=self._run_versions(profile) if self._run_versions else None,
            tenant=tenant, created_by=actor, end_user=request.user, enqueue=True,
        )
        return run.id

    async def _settle(self, run_id: str) -> RunRecord:
        while True:
            run = await self._store.get_run(run_id)
            if run is None:
                raise KeyError(f"unknown run: {run_id}")
            if run_is_settled(run):
                return run
            await asyncio.sleep(self._poll)

    async def _reply_text(self, run: RunRecord) -> str:
        if run.status is RunStatus.AWAITING_INPUT:
            pending = await self._orchestrator.pending_approval(run.id)
            if pending is not None:
                return pause_text(pending)
        text = answer_text(run)
        if run.status is RunStatus.COMPLETED:
            return text
        detail = f": {run.error}" if run.error else ""
        note = f"(The run {run.id} ended as {run.status.value}{detail}.)"
        return f"{text}\n\n{note}" if text else note

    async def _stream(self, send: Send, run_id: str, reply_id: str, created: int, model: str) -> str:
        await send({"type": "http.response.start", "status": 200, "headers": [
            (b"content-type", b"text/event-stream"), (b"cache-control", b"no-store"),
            (b"x-accel-buffering", b"no"), (RUN_HEADER, run_id.encode()),
        ]})

        async def chunk(delta: dict[str, Any], finish: str | None = None) -> None:
            body = {"id": reply_id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            await send({"type": "http.response.body", "body": f"data: {json.dumps(body)}\n\n".encode(),
                        "more_body": True})

        sent: list[str] = []
        await chunk({"role": "assistant", "content": ""})
        with (self._hub.subscribe(run_id) if self._hub is not None else contextlib.nullcontext()) as deltas:
            await self._queue.enqueue(run_id)
            while True:
                run = await self._store.get_run(run_id)
                if run is None:
                    raise KeyError(f"unknown run: {run_id}")
                if deltas is not None:
                    while not deltas.empty():
                        await self._forward(deltas.get_nowait(), chunk, sent)
                if run_is_settled(run):
                    break
                if deltas is None:
                    await asyncio.sleep(self._poll)
                    continue
                with contextlib.suppress(asyncio.TimeoutError):
                    await self._forward(await asyncio.wait_for(deltas.get(), timeout=self._poll), chunk, sent)
        final = await self._reply_text(run)
        streamed = "".join(sent)
        if final.strip() and not streamed.rstrip().endswith(final.strip()):
            tail = f"\n\n{final}" if streamed.strip() else final
            await chunk({"content": tail})
            sent.append(tail)
        await chunk({}, finish="stop")
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})
        return "".join(sent)

    @staticmethod
    async def _forward(item: Mapping[str, Any], chunk: Callable[..., Awaitable[None]], sent: list[str]) -> None:
        text = str(item.get("text", ""))
        if not text:
            return
        if item.get("type") == "reasoning":
            await chunk({"reasoning_content": text})
            return
        await chunk({"content": text})
        sent.append(text)


async def _send_json(send: Send, status: int, payload: Any, run_id: str) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send({"type": "http.response.start", "status": status, "headers": [
        (b"content-type", b"application/json"), (b"cache-control", b"no-store"),
        (b"content-length", str(len(body)).encode()), (RUN_HEADER, run_id.encode()),
    ]})
    await send({"type": "http.response.body", "body": body})
