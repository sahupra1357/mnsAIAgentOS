"""Instruction files and skills, assembled into a run's system context.

Both are read once when a top-level run is created and added as one system
message; they are never re-read mid-run, which keeps the prompt prefix (and
server-side KV cache) stable. The content hash is recorded in the run's
version record.

Skills use progressive disclosure: only each skill's name and description go
into the system message; the model loads a skill's full text with the
read-only ``load_skill`` tool, which takes a *name* (never a path).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_harness.domain import JsonObject, ToolSpec

MAX_SKILL_BYTES = 64_000
_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class SystemContext:
    text: str
    sources: tuple[str, ...] = ()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    path: Path

    def body(self) -> str:
        text = self.path.read_text(encoding="utf-8")[:MAX_SKILL_BYTES]
        return _strip_frontmatter(text).strip()


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_instructions(files: Iterable[str], *, base: Path, max_bytes: int) -> tuple[str, list[str]]:
    """Concatenate existing instruction files in order, with source headers."""

    blocks: list[str] = []
    sources: list[str] = []
    remaining = max_bytes
    for entry in files:
        path = resolve_path(entry, base)
        if not path.is_file() or remaining <= 0:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if len(text.encode("utf-8")) > remaining:
            text = text.encode("utf-8")[:remaining].decode("utf-8", "ignore") + "\n[... truncated]"
        remaining -= len(text.encode("utf-8"))
        blocks.append(f"## Instructions from {entry}\n\n{text}")
        sources.append(str(path))
    return "\n\n".join(blocks), sources


def discover_skills(directories: Iterable[Path]) -> dict[str, Skill]:
    """Index ``<dir>/<name>/SKILL.md`` files by their frontmatter name."""

    skills: dict[str, Skill] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for skill_file in sorted(directory.glob("*/SKILL.md")):
            meta = _frontmatter(skill_file.read_text(encoding="utf-8")[:MAX_SKILL_BYTES])
            name = meta.get("name") or skill_file.parent.name
            description = meta.get("description", "").strip()
            if not _SKILL_NAME.fullmatch(name) or not description or name in skills:
                continue  # first definition wins; invalid ones are skipped
            skills[name] = Skill(name, description, skill_file)
    return skills


def skills_index(skills: Mapping[str, Skill]) -> str:
    lines = [f"- {skill.name}: {skill.description}" for skill in skills.values()]
    return (
        "## Skills\n\nThese skills are available. Before working on a matching task, load "
        "the skill with the load_skill tool and follow it.\n" + "\n".join(lines)
    )


def load_skill_spec(skills: Mapping[str, Skill]) -> ToolSpec:
    return ToolSpec(
        name="load_skill",
        description="Load the full instructions of a named skill.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "enum": sorted(skills)}},
            "required": ["name"],
            "additionalProperties": False,
        },
    )


def load_skill_handler(skills: Mapping[str, Skill]):
    async def load_skill(arguments: JsonObject) -> JsonObject:
        skill = skills.get(str(arguments["name"]))
        if skill is None:
            return {"error": "unknown skill"}
        return {"name": skill.name, "instructions": skill.body()}

    return load_skill


@dataclass(slots=True)
class SystemContextBuilder:
    """Callable handed to the orchestrator; evaluated once per new run."""

    instruction_files: Sequence[str] = ()
    base: Path = field(default_factory=Path)
    max_bytes: int = 32_768
    skills: Mapping[str, Skill] = field(default_factory=dict)

    def __call__(self) -> SystemContext:
        text, sources = load_instructions(self.instruction_files, base=self.base, max_bytes=self.max_bytes)
        parts = [text] if text else []
        if self.skills:
            parts.append(skills_index(self.skills))
            sources = [*sources, *(str(skill.path) for skill in self.skills.values())]
        return SystemContext("\n\n".join(parts), tuple(sources))


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and not line.startswith((" ", "\t")):
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text


def frontmatter(text: str) -> dict[str, Any]:
    return dict(_frontmatter(text))


def strip_frontmatter(text: str) -> str:
    return _strip_frontmatter(text)
