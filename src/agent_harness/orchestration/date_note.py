"""Tell the model today's date without touching the cached prompt prefix.

Models do not know the current date, so they answer time-sensitive questions from
stale training data ("the next Super Bowl is February 2026"). A date in the system
prompt would change the prefix every day and defeat prompt caching, so the date goes
at the end of the user message that starts a run, and again when a continued run
reaches a new day. It is part of the stored message, so resume sees the same text.
Transcripts hide it with ``strip_date_note``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from agent_harness.domain import Message, Role

_NOTE = re.compile(r"\n\n\[Current date: [A-Za-z]+, (\d{4}-\d{2}-\d{2})\]\Z")


def date_note(now: datetime) -> str:
    """``now`` in the harness's local time zone, as ``[Current date: Tuesday, 2026-10-06]``."""

    local = now.astimezone()
    return f"\n\n[Current date: {local:%A}, {local:%Y-%m-%d}]"


def noted_date(message: Message) -> str | None:
    match = _NOTE.search(message.content) if message.role is Role.USER else None
    return match.group(1) if match else None


def last_noted_date(messages: Sequence[Message]) -> str | None:
    for message in reversed(messages):
        found = noted_date(message)
        if found is not None:
            return found
    return None


def with_date_note(message: Message, now: datetime) -> Message:
    """The message with today's date appended (unchanged if it already carries a note)."""

    if noted_date(message) is not None:
        return message
    return replace(message, content=message.content + date_note(now))


def strip_date_note(text: str) -> str:
    return _NOTE.sub("", text)
