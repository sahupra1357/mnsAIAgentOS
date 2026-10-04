"""Dynamic tool loading for large tool catalogs.

With ``mode = "dynamic"`` a model is offered only the ``core`` tools, the tools
it loaded in this run, and two meta-tools: ``search_tools`` to find tools by
description and ``load_tools`` to add them. Loaded tools are recorded on the
run and appended in load order, so the tool list only grows at the end (the
prompt prefix stays cacheable) and a resumed run offers the same tools.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from agent_harness.domain import ToolRisk, ToolSpec

SEARCH_TOOLS = "search_tools"
LOAD_TOOLS = "load_tools"
META_TOOLS = frozenset({SEARCH_TOOLS, LOAD_TOOLS})
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class ToolLoadingPolicy:
    mode: str = "all"  # "all" | "dynamic"
    core: tuple[str, ...] = ()
    max_loaded: int = 40
    search_limit: int = 8

    def __post_init__(self) -> None:
        if self.mode not in {"all", "dynamic"}:
            raise ValueError("tool_loading.mode must be 'all' or 'dynamic'")
        if not 1 <= self.max_loaded <= 500:
            raise ValueError("tool_loading.max_loaded must be between 1 and 500")
        if not 1 <= self.search_limit <= 50:
            raise ValueError("tool_loading.search_limit must be between 1 and 50")

    @property
    def dynamic(self) -> bool:
        return self.mode == "dynamic"


SEARCH_TOOLS_SPEC = ToolSpec(
    name=SEARCH_TOOLS,
    description=(
        "Find tools you can load. Many tools exist but only a few are loaded. Describe "
        "the capability you need; you get matching tool names and descriptions. Then "
        "call load_tools with the names you want."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 300},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)

LOAD_TOOLS_SPEC = ToolSpec(
    name=LOAD_TOOLS,
    description=(
        "Load tools by exact name (from search_tools) so you can call them. Loaded tools "
        "stay available for the rest of this run."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
                "minItems": 1,
                "maxItems": 20,
            }
        },
        "required": ["names"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)


def search(specs: Sequence[ToolSpec], query: str, limit: int) -> list[ToolSpec]:
    """Rank tools for ``query``: SQLite FTS5 (BM25) when available, else word overlap."""

    candidates = [spec for spec in specs if spec.name not in META_TOOLS]
    if not candidates:
        return []
    terms = _WORD.findall(query.lower())
    if not terms:
        return candidates[:limit]
    try:
        return _fts_search(candidates, terms, limit)
    except sqlite3.OperationalError:
        return _overlap_search(candidates, terms, limit)


def _fts_search(specs: Sequence[ToolSpec], terms: list[str], limit: int) -> list[ToolSpec]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE tools USING fts5(name, body, tokenize='porter')")
        connection.executemany(
            "INSERT INTO tools(rowid, name, body) VALUES (?, ?, ?)",
            [(i, _name_words(spec.name), spec.description) for i, spec in enumerate(specs)],
        )
        match = " OR ".join(f'"{term}"' for term in terms)
        rows = connection.execute(
            "SELECT rowid FROM tools WHERE tools MATCH ? ORDER BY bm25(tools, 2.0, 1.0) LIMIT ?",
            (match, limit),
        ).fetchall()
    finally:
        connection.close()
    return [specs[row[0]] for row in rows]


def _overlap_search(specs: Sequence[ToolSpec], terms: list[str], limit: int) -> list[ToolSpec]:
    wanted = set(terms)
    scored = []
    for index, spec in enumerate(specs):
        words = set(_WORD.findall(f"{_name_words(spec.name)} {spec.description}".lower()))
        score = len(wanted & words)
        if score:
            scored.append((-score, index, spec))
    return [spec for _, _, spec in sorted(scored)[:limit]]


def _name_words(name: str) -> str:
    return " ".join(_WORD.findall(name.lower().replace("__", " ").replace("_", " ")))
