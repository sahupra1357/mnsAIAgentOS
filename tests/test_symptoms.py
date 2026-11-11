"""Phase 25: one test per symptom of the loop, context and memory flaws.

Every test builds the full application from the repository's ``models.toml`` (the one
configuration the user runs) with a scripted model, so a fix counts only once the root
config turns it on. The model here answers summary, guard and memory-extraction calls by
their ``purpose`` and takes chat replies from a script; no network is used.

The symptoms (see docs/audit and CLAUDE.md, Phases 25-29):

* the latest question is answered as the first one after a trim;
* a file read earlier in the task is forgotten once its turn is dropped;
* a call repeated after its result was dropped is refused instead of answered;
* a new chat knows nothing an earlier chat was told;
* a run that ran out of budget leaves tool calls without results, and the next message
  runs them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness.application import build_application
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    Usage,
)

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "models.toml"
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise.toml",
}


class SymptomModel:
    """Chat replies come from the script; every other purpose gets a fixed reply."""

    dimensions = 64

    def __init__(self, replies: Sequence[ModelResponse], *, notes: Sequence[str] = ()) -> None:
        self.replies = deque(replies)
        self.notes = list(notes)
        self.requests: list[ModelRequest] = []
        self.other: list[ModelRequest] = []

    def resolve(self, profile: str) -> "SymptomModel":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        purpose = request.metadata.get("purpose")
        if purpose == "guard":
            return ModelResponse(content="safe", finish_reason=FinishReason.STOP)
        if purpose == "summary":
            self.other.append(request)
            return ModelResponse(content="The agent looked at some files.", usage=Usage(50, 10))
        if purpose == "memory":
            self.other.append(request)
            notes = [{"text": text, "kind": "fact", "scope": "project"} for text in self.notes]
            return ModelResponse(content=json.dumps({"notes": notes}), usage=Usage(50, 10))
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("the script has no reply left")
        return self.replies.popleft()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(word.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def say(text: str, tokens: int = 30) -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP, usage=Usage(tokens, 10))


def read(index: int, path: str, tokens: int = 30) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(f"call-{index}", "read_file", {"path": path}),),
        finish_reason=FinishReason.TOOL_CALLS, usage=Usage(tokens, 10),
    )


def text_of(request: ModelRequest) -> str:
    return "\n".join(message.content for message in request.messages)


class SymptomCase(unittest.TestCase):
    """Runs the root configuration in a scratch folder: workspace, state and memory."""

    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory(prefix="agent-harness-symptoms-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        patcher = mock.patch.dict("os.environ", HERMETIC)
        patcher.start()
        self.addCleanup(patcher.stop)

    def overrides(self) -> dict[str, Any]:
        return {
            "project": {"root": "config"},
            "coding": {"workspace": str(self.workspace)},
            "artifacts": {"directory": str(self.root / "artifacts")},
            "memory": {"database": str(self.root / "memory.sqlite3")},
            "langfuse": {"enabled": False},
        }

    def app(self, model: SymptomModel, database: str = "runs.sqlite3") -> Any:
        return build_application(config_path=FULL, database_path=self.root / database,
                                 config_overrides=self.overrides(), model_resolver=model)

    def big_file(self, name: str, marker: str, chars: int = 30_000) -> str:
        line = f"{marker} " + "lorem ipsum dolor sit amet " * 3 + "\n"
        (self.workspace / name).write_text(line * (chars // len(line)), encoding="utf-8")
        return name


class LatestRequestTests(SymptomCase):
    def test_the_latest_question_stays_its_own_message_after_a_trim(self) -> None:
        files = [self.big_file(f"notes{i}.txt", f"marker{i}") for i in range(8)]
        model = SymptomModel(
            [say("Paris.")] + [read(i, name) for i, name in enumerate(files)] + [say("Done reading.")]
        )

        async def scenario() -> None:
            async with self.app(model) as app:
                profile = app.config.resolve_model_name(None)
                run = await app.orchestrator.create_run(
                    model_profile=profile,
                    messages=[Message(role=Role.USER, content="What is the capital of France?")],
                )
                await app.orchestrator.run(run.id)
                await app.orchestrator.continue_run(run.id, "Now read the eight notes files and say done.")
                done = await app.orchestrator.run(run.id)
                self.assertIs(done.status, RunStatus.COMPLETED)

        asyncio.run(scenario())
        final = model.requests[-1]
        asked = [m for m in final.messages
                 if m.role is Role.USER and m.content.startswith("Now read the eight notes files")]
        self.assertEqual(len(asked), 1, "the latest request must be a message of its own")
        self.assertNotIn("[Latest user message - answer this one]", text_of(final))
        first = next(m for m in final.messages if m.role is Role.USER)
        self.assertTrue(first.content.startswith("What is the capital of France?"))
        self.assertNotIn("[Summary of", first.content)  # nothing is glued onto the first request
        self.assertNotIn("[Context note", first.content)


class ForgottenFileTests(SymptomCase):
    def test_a_file_read_earlier_is_still_named_after_its_turn_was_dropped(self) -> None:
        files = [self.big_file(f"module{i}.py", f"marker{i}") for i in range(5)]
        model = SymptomModel([read(i, name) for i, name in enumerate(files)] + [say("Reviewed.")])

        async def scenario() -> None:
            async with self.app(model) as app:
                run = await app.orchestrator.create_run(
                    model_profile=app.config.resolve_model_name(None),
                    messages=[Message(role=Role.USER, content="Review the five modules.")],
                )
                done = await app.orchestrator.run(run.id)
                self.assertIs(done.status, RunStatus.COMPLETED)

        asyncio.run(scenario())
        final = text_of(model.requests[-1])
        self.assertNotIn("marker0 lorem", final)  # the first file's text really was dropped
        # The model's summary here names no file; the harness's own record must.
        self.assertIn("module0.py", final)
        self.assertIn("module1.py", final)


class RepeatedCallTests(SymptomCase):
    def test_a_repeated_read_is_answered_from_the_earlier_result_not_refused(self) -> None:
        (self.workspace / "small.txt").write_text("the answer is 42\n", encoding="utf-8")
        model = SymptomModel([read(i, "small.txt") for i in range(7)] + [say("It is 42.")])

        async def scenario() -> tuple[RunStatus, list[Message]]:
            async with self.app(model) as app:
                run = await app.orchestrator.create_run(
                    model_profile=app.config.resolve_model_name(None),
                    messages=[Message(role=Role.USER, content="What does small.txt say?")],
                )
                done = await app.orchestrator.run(run.id)
                return done.status, list(done.messages)

        status, messages = asyncio.run(scenario())
        self.assertIs(status, RunStatus.COMPLETED)
        results = [m.content for m in messages if m.role is Role.TOOL]
        self.assertEqual(len(results), 7)
        self.assertIn("the answer is 42", results[-1])  # the sixth and seventh still carry the text
        self.assertIn("already", results[-1].lower())


class NewChatTests(SymptomCase):
    @unittest.expectedFailure  # fixed in a later phase (see the module docstring)
    def test_a_new_chat_is_given_what_an_earlier_chat_was_told(self) -> None:
        model = SymptomModel(
            [say("Noted."), say("You deploy to falcon-prod.")],
            notes=["The deploy target of this project is falcon-prod."],
        )

        async def scenario() -> None:
            async with self.app(model) as app:
                profile = app.config.resolve_model_name(None)
                first = await app.orchestrator.create_run(
                    model_profile=profile,
                    messages=[Message(role=Role.USER, content="For later: our deploy target is falcon-prod.")],
                )
                await app.orchestrator.run(first.id)
                second = await app.orchestrator.create_run(
                    model_profile=profile,
                    messages=[Message(role=Role.USER, content="Which deploy target do we use?")],
                )
                done = await app.orchestrator.run(second.id)
                self.assertIs(done.status, RunStatus.COMPLETED)
                events = await app.store.list_events(second.id)
                self.assertIn("memory_recalled", [e.type.value for e in events])

        asyncio.run(scenario())
        self.assertIn("falcon-prod", text_of(model.requests[-1]))


class OrphanToolCallTests(SymptomCase):
    @unittest.expectedFailure  # fixed in a later phase (see the module docstring)
    def test_budget_exhaustion_leaves_no_tool_call_unanswered(self) -> None:
        (self.workspace / "a.txt").write_text("alpha\n", encoding="utf-8")
        model = SymptomModel([read(0, "a.txt", tokens=5_000), say("Stopped as asked.")])

        async def scenario() -> None:
            async with self.app(model) as app:
                run = await app.orchestrator.create_run(
                    model_profile=app.config.resolve_model_name(None),
                    messages=[Message(role=Role.USER, content="Read a.txt.")],
                    budget=RunBudget(max_steps=10, max_wall_seconds=300, max_tokens=1_000),
                )
                stopped = await app.orchestrator.run(run.id)
                self.assertIs(stopped.status, RunStatus.BUDGET_EXHAUSTED)
                asked = {c.id for m in stopped.messages for c in m.tool_calls}
                answered = {m.tool_call_id for m in stopped.messages if m.role is Role.TOOL}
                self.assertEqual(asked, answered, "every tool call needs a result in the history")
                await app.orchestrator.continue_run(
                    run.id, "Never mind, do not read it.",
                    budget=RunBudget(max_steps=20, max_wall_seconds=600, max_tokens=100_000),
                )
                done = await app.orchestrator.run(run.id)
                self.assertIs(done.status, RunStatus.COMPLETED)
                events = await app.store.list_events(run.id)
                turn = [e.type for e in events]
                after = turn[len(turn) - turn[::-1].index(EventType.USER_MESSAGE_ADDED):]
                self.assertNotIn(EventType.TOOL_REQUESTED, after)  # the old call is not run now

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
