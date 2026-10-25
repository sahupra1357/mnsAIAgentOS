"""Run handoffs: what a finished turn leaves for later runs of the same project.

A new chat used to start blind. On 2026-10-25 "Continue the digital-document-docling video
from where you stopped: ..." started a fresh run that never saw the earlier run's status
report, and spent about 50 steps rediscovering the project (and searching the disk for
scripts the earlier run had already found).

* :func:`build_handoff` turns a top-level run into a :class:`RunHandoff` (request, final
  answer, todo or plan state, files written with ``apply_patch``, last summary), shortened.
  The orchestrator saves one at the end of every turn when the store keeps handoffs.
* :func:`find_link` scores a new prompt against recent handoffs. Distinctive terms (slugs,
  paths, file names: anything with ``-``, ``_``, ``.``, ``/`` or a digit) weigh most; plain
  words a little; continuation cues ("continue", "where you stopped") a little more. The
  best match above :data:`LINK_THRESHOLD` (ties: the newest) is linked; a prompt with no
  match links nothing, so unrelated chats start clean. No model call: it is deterministic.
  A run that left nothing (no answer, task list, summary, or files) is never linked: the
  failed docling retry had only the same request, and would have outscored the run that
  holds the status report.
* :func:`linked_block` is the system message a linked run starts with.
* :func:`ensure_handoffs` backfills handoffs for runs made before this existed.
* :func:`previous_runs_tool` is the read-only ``previous_runs`` tool.

Handoffs hold model output from earlier runs; the block says to check files on disk
before relying on it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from agent_harness.domain import (
    EventType,
    JsonObject,
    Message,
    Role,
    RunEvent,
    RunHandoff,
    RunQuery,
    RunRecord,
    RunStatus,
    ToolRisk,
    ToolSpec,
    ToolTrust,
    utc_now,
)

from .context import ORCHESTRATOR_NOTE_PREFIXES

PREVIOUS_RUNS_TOOL = "previous_runs"
LINK_THRESHOLD = 4
MAX_AGE = timedelta(days=30)
RECENT = 50

_REQUEST_CHARS = 1_500
_ANSWER_CHARS = 4_000
_SUMMARY_CHARS = 3_000
_MAX_TODOS = 30
_MAX_FILES = 40
_FILE_TOOLS = frozenset({"apply_patch"})
_HOOK_PREFIX = "[Context from hooks]"
_TERM = re.compile(r"[a-z0-9][a-z0-9_./-]*[a-z0-9]")
_CUES = ("continue", "where you stopped", "where you left", "left off", "resume", "remaining",
         "pick up", "carry on", "finish the")
_STOPWORDS = frozenset(
    "about above after again against also because been before being below between both could "
    "does doing down during each first from further have having here into just more most once "
    "only other over same should some such than that their them then there these they this "
    "those through under until very what when where which while will with would your please "
    "make sure using need want like".split()
)


@dataclass(frozen=True, slots=True)
class Link:
    """A new run's match: the earlier run's handoff, its score, and the terms that matched."""

    handoff: RunHandoff
    score: int
    terms: tuple[str, ...]


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 15].rstrip() + " [... shortened]"


def _user_requests(messages: Sequence[Message]) -> list[str]:
    return [m.content for m in messages
            if m.role is Role.USER and m.content.strip()
            and not m.content.startswith((*ORCHESTRATOR_NOTE_PREFIXES, _HOOK_PREFIX))]


def build_handoff(run: RunRecord, events: Sequence[RunEvent] = ()) -> RunHandoff:
    """The handoff of a top-level run as it stands (``events`` give the run's error)."""

    requests = _user_requests(run.messages)
    answer = next((m.content for m in reversed(run.messages)
                   if m.role is Role.ASSISTANT and not m.tool_calls and m.content.strip()), "")
    if run.plan is not None:
        todos = [f"[{step.status.value}] {step.id}: {step.title}" for step in run.plan.steps]
    else:
        todos = [f"[{item.status.value}] {item.content}" for item in run.todos]
    files: list[str] = []
    for message in [*run.archived_messages, *run.messages]:
        for call in message.tool_calls:
            path = call.arguments.get("path") if call.name in _FILE_TOOLS else None
            if isinstance(path, str) and path and path not in files:
                files.append(path)
    error = run.error
    if error is None and run.status is RunStatus.BUDGET_EXHAUSTED:  # not an earlier turn's
        failed = next((e for e in reversed(events) if e.type is EventType.BUDGET_EXHAUSTED), None)
        error = str(failed.payload.get("reason")) if failed is not None and failed.payload.get("reason") else None
    return RunHandoff(
        run_id=run.id,
        tenant=run.tenant,
        status=run.status.value,
        created_at=run.created_at,
        updated_at=run.updated_at,
        request=_shorten(requests[0], _REQUEST_CHARS) if requests else "",
        latest_request=_shorten(requests[-1], _REQUEST_CHARS) if len(requests) > 1 else "",
        answer=_shorten(answer, _ANSWER_CHARS),
        todos=tuple(_shorten(t, 300) for t in todos[:_MAX_TODOS]),
        files=tuple(files[-_MAX_FILES:]),
        summary=_shorten(run.context_summary.content, _SUMMARY_CHARS) if run.context_summary else "",
        error=_shorten(error, 300) if error else None,
    )


def handoff_text(handoff: RunHandoff) -> str:
    """Searchable text of a handoff (also what the block shows)."""

    parts = [handoff.request, handoff.latest_request, handoff.answer, *handoff.todos,
             *handoff.files, handoff.summary]
    return "\n".join(part for part in parts if part)


def prompt_terms(prompt: str) -> tuple[list[str], list[str]]:
    """(distinctive terms, plain words) of a prompt, lower-cased, without duplicates."""

    distinct: list[str] = []
    words: list[str] = []
    for term in _TERM.findall(prompt.lower()):
        term = term.strip("./-_")
        if len(term) < 4:
            continue
        if any(c in term for c in "-_./") or any(c.isdigit() for c in term):
            if len(term) >= 5 and term not in distinct:
                distinct.append(term)
        elif len(term) >= 5 and term not in _STOPWORDS and term not in words:
            words.append(term)
    return distinct, words


def has_cue(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(cue in lowered for cue in _CUES)


def score(prompt: str, handoff: RunHandoff) -> tuple[int, tuple[str, ...]]:
    """How well ``prompt`` matches ``handoff``, and the distinctive terms that matched."""

    distinct, words = prompt_terms(prompt)
    text = handoff_text(handoff).lower()
    matched = tuple(term for term in distinct if term in text)
    if not matched:
        return 0, ()
    plain = sum(1 for word in words if word in text)
    return 3 * len(matched) + min(plain, 6) + (2 if has_cue(prompt) else 0), matched


def has_content(handoff: RunHandoff) -> bool:
    """Whether a run left anything beyond its request."""

    return bool(handoff.answer or handoff.todos or handoff.summary or handoff.files)


def find_link(prompt: str, handoffs: Sequence[RunHandoff], *, now: datetime | None = None,
              exclude: str | None = None) -> Link | None:
    """The best-matching recent handoff (ties: the newest), or None below the threshold."""

    now = now or utc_now()
    best: Link | None = None
    for handoff in sorted(handoffs, key=lambda h: h.updated_at, reverse=True):
        if handoff.run_id == exclude or now - handoff.updated_at > MAX_AGE or not has_content(handoff):
            continue
        points, terms = score(prompt, handoff)
        if points >= LINK_THRESHOLD and (best is None or points > best.score):
            best = Link(handoff, points, terms)
    return best


def linked_block(link: Link) -> str:
    """The system message a linked run starts with."""

    h = link.handoff
    lines = [
        f"[Earlier run {h.run_id[:8]} of this project, {h.updated_at:%Y-%m-%d %H:%M} UTC, "
        f"{h.status}{f': {h.error}' if h.error else ''}]",
        "The user's message matches what this earlier run worked on "
        f"({', '.join(link.terms[:6])}). This is what it recorded when it stopped; files may have "
        "changed since, so check them before relying on it. The previous_runs tool shows more.",
    ]
    if h.request:
        lines += ["", "Its request:", h.request]
    if h.latest_request:
        lines += ["", "Its latest request:", h.latest_request]
    if h.answer:
        lines += ["", "Its final answer:", h.answer]
    if h.todos:
        lines += ["", "Its task list when it stopped:", *h.todos]
    if h.files:
        lines += ["", "Files it wrote with apply_patch:", *h.files]
    if h.summary:
        lines += ["", "Its last context summary:", h.summary]
    return "\n".join(lines)


_TERMINAL = (RunStatus.COMPLETED, RunStatus.BUDGET_EXHAUSTED, RunStatus.FAILED, RunStatus.CANCELLED)


async def ensure_handoffs(store: Any, tenant: str, *, limit: int = RECENT) -> list[RunHandoff]:
    """Recent handoffs of ``tenant``, newest first, backfilling runs that have none yet.

    Runs made before handoffs existed (or before an upgrade) get one built from their
    snapshot the first time they are looked for; later calls only read the table.
    """

    saved = {h.run_id: h for h in await store.recent_handoffs(tenant, limit)}
    list_runs = getattr(store, "list_runs", None)
    if list_runs is not None:
        page = await list_runs(RunQuery(tenant=tenant, statuses=_TERMINAL, limit=limit))
        for summary in page.runs:
            known = saved.get(summary.id)
            if known is not None and known.updated_at >= summary.updated_at:
                continue
            run = await store.get_run(summary.id)
            if run is None or run.parent_run_id is not None:
                continue
            handoff = build_handoff(run, await store.list_events(run.id))
            await store.save_handoff(handoff)
            saved[handoff.run_id] = handoff
    return sorted(saved.values(), key=lambda h: h.updated_at, reverse=True)[:limit]


def link_payload(link: Link) -> JsonObject:
    """The ``run_linked`` event payload (no handoff text)."""

    return {"linked_run_id": link.handoff.run_id, "score": link.score, "terms": list(link.terms[:10]),
            "linked_status": link.handoff.status}


def linked_note(events: Sequence[RunEvent]) -> str | None:
    """A one-line note for the terminal when the run was linked, else None."""

    event = next((e for e in events if e.type is EventType.RUN_LINKED), None)
    if event is None:
        return None
    terms = ", ".join(str(t) for t in event.payload.get("terms", ())[:3])
    return f"linked to earlier run {str(event.payload.get('linked_run_id', ''))[:8]} ({terms})"


PREVIOUS_RUNS_SPEC = ToolSpec(
    name=PREVIOUS_RUNS_TOOL,
    description=(
        "Look up earlier runs (conversations) of this project: what each was asked, its final "
        "answer, task list, and the files it wrote. Without arguments it lists recent runs; "
        "with query it lists the runs that match; with run_id (or its first 8 characters) it "
        "shows that run's full handoff. Use it when the user refers to earlier work."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "words, a slug, or a path to match"},
            "run_id": {"type": "string", "description": "a run id or its first 8+ characters"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
    idempotent=True,
    trust=ToolTrust.TRUSTED,
)


def previous_runs_tool(store: Any) -> Any:
    """The handler of ``previous_runs`` for a store that keeps handoffs."""

    from .runcontext import current_run_id

    async def previous_runs(arguments: JsonObject) -> JsonObject:
        current = current_run_id()
        run = await store.get_run(current) if current else None
        tenant = run.tenant if run is not None else "default"
        handoffs = [h for h in await ensure_handoffs(store, tenant) if h.run_id != current]
        wanted = str(arguments.get("run_id") or "").strip()
        if wanted:
            if len(wanted) < 8:
                return {"error": "run_id needs at least 8 characters"}
            found = next((h for h in handoffs if h.run_id.startswith(wanted)), None)
            if found is None:
                found = await store.get_handoff(wanted) if len(wanted) == 32 else None
            if found is None:
                return {"error": f"no earlier run {wanted!r} in this project"}
            return {"run": linked_block(Link(found, 0, ()))}
        limit = int(arguments.get("limit") or 10)
        query = str(arguments.get("query") or "").strip()
        if query:
            scored = [(score(query, h)[0] or _plain_score(query, h), h) for h in handoffs]
            handoffs = [h for points, h in sorted(scored, key=lambda p: (-p[0], -p[1].updated_at.timestamp()))
                        if points > 0]
        return {"runs": [_row(h) for h in handoffs[:limit]]}

    return previous_runs


def _plain_score(query: str, handoff: RunHandoff) -> int:
    distinct, words = prompt_terms(query)
    text = handoff_text(handoff).lower()
    return sum(1 for term in (*distinct, *words) if term in text)


def _row(handoff: RunHandoff) -> JsonObject:
    return {"run_id": handoff.run_id, "updated_at": handoff.updated_at.isoformat(timespec="minutes"),
            "status": handoff.status, "request": _shorten(handoff.latest_request or handoff.request, 200)}


def handoff_to_json(handoff: RunHandoff) -> JsonObject:
    """The stored form of a handoff (all three stores)."""

    return {
        "run_id": handoff.run_id, "tenant": handoff.tenant, "status": handoff.status,
        "created_at": handoff.created_at.isoformat(), "updated_at": handoff.updated_at.isoformat(),
        "request": handoff.request, "latest_request": handoff.latest_request, "answer": handoff.answer,
        "todos": list(handoff.todos), "files": list(handoff.files), "summary": handoff.summary,
        "error": handoff.error,
    }


def handoff_from_json(data: JsonObject) -> RunHandoff:
    return RunHandoff(
        run_id=str(data["run_id"]), tenant=str(data["tenant"]), status=str(data["status"]),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        updated_at=datetime.fromisoformat(str(data["updated_at"])),
        request=str(data.get("request", "")), latest_request=str(data.get("latest_request", "")),
        answer=str(data.get("answer", "")), todos=tuple(str(t) for t in data.get("todos", ())),
        files=tuple(str(f) for f in data.get("files", ())), summary=str(data.get("summary", "")),
        error=str(data["error"]) if data.get("error") is not None else None,
    )
