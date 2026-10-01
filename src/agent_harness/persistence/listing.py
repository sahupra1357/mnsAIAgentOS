"""Helpers shared by the run stores' ``list_runs`` (summaries and page cursors).

Runs are listed newest first by ``(created_at, id)``; a cursor is the
position of the last run on a page, so pages stay stable while new runs are
created.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime

from agent_harness.domain import Role, RunRecord, RunSummary

TITLE_CHARS = 80


def summarize_run(run: RunRecord) -> RunSummary:
    return RunSummary(
        id=run.id,
        status=run.status,
        model_profile=run.model_profile,
        tenant=run.tenant,
        created_at=run.created_at,
        updated_at=run.updated_at,
        step_count=run.step_count,
        consumed_tokens=run.consumed_tokens,
        consumed_cost_usd=run.consumed_cost_usd,
        created_by=run.created_by,
        parent_run_id=run.parent_run_id,
        depth=run.depth,
        error=run.error,
        title=_title(run),
    )


def _title(run: RunRecord) -> str:
    first = next((m.content for m in run.messages if m.role is Role.USER and m.content.strip()), "")
    line = first.strip().splitlines()[0] if first.strip() else ""
    return line if len(line) <= TITLE_CHARS else line[: TITLE_CHARS - 1] + "…"


def encode_cursor(created_at: datetime, run_id: str) -> str:
    raw = json.dumps([created_at.isoformat(), run_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_at, run_id = json.loads(base64.urlsafe_b64decode(padded.encode()))
        if not isinstance(run_id, str) or not isinstance(created_at, str):
            raise ValueError("cursor fields must be strings")
        return datetime.fromisoformat(created_at), run_id
    except (binascii.Error, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ValueError("invalid cursor") from exc
