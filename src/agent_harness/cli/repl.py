"""Interactive chat in the terminal: ``agent-harness`` with no arguments.

The first message creates a run; later messages continue it with ``continue_run``,
so the whole conversation is one durable run that ``resume``, ``status`` and
``runs`` can see. Approvals, plan approvals, and ``ask_user`` questions are asked
inline. Input and output are injected so tests can drive a session without a
terminal.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from ..domain import ContentPart, Message, Role, RunBudget, RunQuery, RunRecord, RunStatus
from ..extensions.templates import expand_prompt
from ..orchestration.handoff import linked_note
from ..orchestration.orchestrator import resumable_steps, steps_left_note
from .images import split_images

HELP = """commands:
  /new      start a new conversation
  /resume   list recent conversations; /resume N or /resume RUN_ID continues one
  /status   show the current run's usage against its budget
  /help     show this help
  /exit     leave (also Ctrl-D)
  /NAME ... expand the prompt template NAME (see `agent-harness templates`)"""


class ChatIO(Protocol):
    async def ask(self, prompt: str) -> str | None:
        """Read one line; ``None`` at end of input."""

    def say(self, text: str) -> None:
        """Print answer text (stdout)."""

    def note(self, text: str) -> None:
        """Print status text (stderr)."""


class StreamState(Protocol):
    printed: bool


@dataclass(frozen=True, slots=True)
class ChatOptions:
    profile: str
    route: str | None = None
    system: str | None = None
    mode: str | None = None
    planning: bool | None = None
    # Per-turn budget: each turn gets this many more steps and seconds.
    budget: RunBudget = RunBudget()
    # A new conversation whose first message matches an earlier run starts with its handoff
    # ([project] link_runs; --no-link turns it off).
    link: bool = False


class ChatSession:
    """One terminal conversation over an application's orchestrator."""

    def __init__(
        self,
        app: Any,
        options: ChatOptions,
        io: ChatIO,
        *,
        stream: StreamState | None = None,
    ) -> None:
        self._app = app
        self._options = options
        self._io = io
        self._stream = stream
        self.run_id: str | None = None
        self._listed: list[str] = []

    async def loop(self, *, resume: str | None = None, latest: bool = False) -> int:
        """``latest`` (``--continue``) or ``resume`` (``--resume RUN_ID``) reopens an earlier run."""

        self._io.note(
            f"agent-harness chat — profile {self._options.profile}. /help for commands, /exit to leave."
        )
        if latest or resume:
            await self.resume(resume)
        while True:
            line = await self._io.ask("> ")
            if line is None:
                return 0
            text = line.strip()
            if not text:
                continue
            if text in {"/exit", "/quit"}:
                return 0
            if text == "/help":
                self._io.note(HELP)
                continue
            if text == "/new":
                self.run_id = None
                self._io.note("new conversation")
                continue
            if text == "/status":
                await self._status()
                continue
            if text == "/resume" or text.startswith("/resume "):
                choice = text[len("/resume"):].strip()
                if choice:
                    await self.resume(self._listed[int(choice) - 1]
                                      if choice.isdigit() and 0 < int(choice) <= len(self._listed) else choice)
                else:
                    await self._list_runs()
                continue
            await self.send(text)

    async def resume(self, run_id: str | None = None) -> bool:
        """Continue an earlier conversation: ``run_id`` (or its first 8+ characters), or the latest."""

        runs = await self._recent_runs(50)
        if run_id is None:
            found = runs[0] if runs else None
        else:
            found = next((r for r in runs if r.id == run_id or (len(run_id) >= 8 and r.id.startswith(run_id))),
                         None)
            if found is None and len(run_id) == 32:
                found = await self._app.store.get_run(run_id)
        if found is None:
            self._io.note("no earlier conversation to resume" if run_id is None else f"no run {run_id!r}")
            return False
        self.run_id = found.id
        title = getattr(found, "title", "") or ""
        self._io.note(f"resumed run {found.id} ({found.status.value}){': ' + title if title else ''}")
        return True

    async def _recent_runs(self, limit: int) -> list[Any]:
        list_runs = getattr(self._app.store, "list_runs", None)
        if list_runs is None:
            return []
        runs = (await list_runs(RunQuery(limit=limit))).runs
        return sorted(runs, key=lambda r: r.updated_at, reverse=True)  # the latest used first

    async def _list_runs(self) -> None:
        runs = await self._recent_runs(10)
        self._listed = [r.id for r in runs]
        if not runs:
            self._io.note("no earlier conversations")
            return
        for number, r in enumerate(runs, 1):
            self._io.note(f"  {number:>2}. {r.id[:8]}  {r.updated_at:%Y-%m-%d %H:%M}  {r.status.value:<16} {r.title}")
        self._io.note("/resume N continues one")

    async def send(self, text: str) -> RunRecord:
        """Send one user message, run it to a stop, and show the result."""

        text, images = split_images([text])  # an image file named in the line is attached
        prompt, template = expand_prompt(text, self._app.templates)
        orchestrator = self._app.orchestrator
        if self._stream is not None:
            self._stream.printed = False
        run = await self._continue(prompt, images) if self.run_id is not None else None
        if run is None:
            run = await self._create(prompt, template, images)
            note = linked_note(await self._app.store.list_events(run.id)) if self._options.link else None
            if note:
                self._io.note(f"({note})")
        self.run_id = run.id
        run = await orchestrator.run(run.id)
        run = await self._settle(run)
        self._show(run)
        if run.plan is not None:
            left = resumable_steps(run.plan, await self._app.store.list_events(run.id))
            if left:
                self._io.note(f"({steps_left_note(left)})")
        return run

    async def _create(self, prompt: str, template: Any, images: Sequence[str] = ()) -> RunRecord:
        options = self._options
        messages: list[Message] = []
        if options.system:
            messages.append(Message(role=Role.SYSTEM, content=options.system))
        messages.append(Message(role=Role.USER, content=prompt,
                                parts=tuple(ContentPart(image_url=url) for url in images)))
        return await self._app.orchestrator.create_run(
            model_profile=options.profile,
            route=options.route,
            mode=options.mode,
            versions=self._app.run_versions(options.profile),
            template={"name": template.name, "sha256": template.sha256} if template else None,
            planning=options.planning,
            messages=messages,
            budget=options.budget,
            link=options.link,
        )

    async def _continue(self, prompt: str, images: Sequence[str] = ()) -> RunRecord | None:
        """Continue the current run; ``None`` when it cannot be continued."""

        assert self.run_id is not None
        orchestrator = self._app.orchestrator
        current = await self._app.store.get_run(self.run_id)
        if current is None or current.status not in {
            RunStatus.COMPLETED, RunStatus.BUDGET_EXHAUSTED,
        }:
            status = current.status.value if current is not None else "missing"
            self._io.note(f"(the previous run is {status}; starting a new conversation)")
            return None
        usage = await orchestrator.usage(self.run_id)
        turn = self._options.budget
        # Budgets are cumulative across turns, so each turn adds its own allowance.
        budget = replace(
            current.budget,
            max_steps=usage.steps + turn.max_steps,
            max_wall_seconds=usage.active_seconds + turn.max_wall_seconds,
        )
        return await orchestrator.continue_run(self.run_id, prompt, budget=budget, images=list(images))

    async def _settle(self, run: RunRecord) -> RunRecord:
        """Answer approvals and questions inline until the run stops waiting."""

        orchestrator = self._app.orchestrator
        while run.status is RunStatus.AWAITING_INPUT:
            pending = await orchestrator.pending_approval(run.id)
            if pending is None:
                break
            if self._stream is not None and self._stream.printed:
                self._io.say("")
                self._stream.printed = False
            if pending.kind == "question":
                self._io.note(f"? {pending.arguments.get('question', '')}")
                for option in pending.arguments.get("options") or ():
                    self._io.note(f"  - {option}")
                reply = await self._io.ask("answer (empty to decline)> ")
                if reply is None or not reply.strip():
                    await orchestrator.resolve_approval(run.id, call_id=pending.call_id, approved=False)
                else:
                    await orchestrator.answer(run.id, call_id=pending.call_id, answer=reply.strip())
            else:
                what = "plan" if pending.kind == "plan" else f"tool {pending.tool_name} ({pending.risk.value})"
                self._io.note(f"approval needed: {what}")
                self._io.note(json.dumps(dict(pending.arguments), indent=2, ensure_ascii=False))
                if pending.kind == "plan":
                    reply = await self._io.ask("allow? [y/N] ")
                else:
                    reply = await self._io.ask(
                        "allow? [y/N/a] (a = yes, and no untrusted-output prompts until your next message) ")
                answer = (reply or "").strip().lower()
                rest_of_turn = pending.kind != "plan" and answer in {"a", "all"}
                approved = rest_of_turn or answer in {"y", "yes"}
                feedback = None
                if not approved and pending.kind == "plan":
                    feedback = (await self._io.ask("feedback (optional)> ") or "").strip() or None
                await orchestrator.resolve_approval(
                    run.id, call_id=pending.call_id, approved=approved, feedback=feedback,
                    rest_of_turn=rest_of_turn,
                )
            run = await orchestrator.resume(run.id)
        return run

    def _show(self, run: RunRecord) -> None:
        streamed = self._stream is not None and self._stream.printed
        if run.output is not None:
            if streamed:
                self._io.say("")
            self._io.say(json.dumps(run.output, indent=2, ensure_ascii=False))
        elif streamed:
            self._io.say("")
        else:
            for message in reversed(run.messages):
                if message.role is Role.ASSISTANT and not message.tool_calls:
                    self._io.say(message.content)
                    break
        if run.status is not RunStatus.COMPLETED:
            detail = f": {run.error}" if run.error else ""
            self._io.note(f"(run {run.id} is {run.status.value}{detail})")

    async def _status(self) -> None:
        if self.run_id is None:
            self._io.note("no conversation yet")
            return
        usage = await self._app.orchestrator.usage(self.run_id)
        self._io.note(
            f"run {usage.run_id} {usage.status.value} on {usage.model_profile}: "
            f"{usage.steps}/{usage.max_steps} steps, {usage.total_tokens} tokens, "
            f"${usage.cost_usd:.4f}"
        )


class TerminalIO:
    """ChatIO on stdin/stdout/stderr; input is read off the event loop."""

    def __init__(self, reader: Callable[[str], str] | None = None) -> None:
        self._reader = reader or input

    async def ask(self, prompt: str) -> str | None:
        import asyncio

        try:
            return await asyncio.to_thread(self._reader, prompt)
        except EOFError:
            return None

    def say(self, text: str) -> None:
        import sys

        print(text, file=sys.stdout, flush=True)

    def note(self, text: str) -> None:
        import sys

        print(text, file=sys.stderr, flush=True)

