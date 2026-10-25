"""Run handoffs and linking, previous_runs, and resuming a chat (2026-10-25).

"Continue the digital-document-docling video from where you stopped: ..." started a fresh
run that never saw the earlier run's status report and spent about 50 steps rediscovering
the project. A new run whose prompt matches an earlier run now starts with its handoff.
"""

from __future__ import annotations

import tempfile
import unittest
from collections import deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from agent_harness.cli.repl import ChatOptions, ChatSession
from agent_harness.config.loader import load_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunHandoff,
    RunStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    utc_now,
)
from agent_harness.orchestration.handoff import (
    PREVIOUS_RUNS_SPEC,
    build_handoff,
    ensure_handoffs,
    find_link,
    linked_note,
    previous_runs_tool,
    prompt_terms,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.orchestration.replay import derive_message_skeleton, message_skeleton
from agent_harness.orchestration.runcontext import CURRENT_RUN_ID
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ToolRegistry, ToolRuntime

ROOT = Path(__file__).resolve().parents[1]

PROMPT = (
    "Continue the digital-document-docling video from where you stopped: write STORYBOARD.md for "
    "horizontal and vertical, build the remaining frames, captions.html, check both projects, render, "
    "thumbnails and poster, add_intro.py, name_final.sh, and the publish folder. Re-read "
    "audio_meta.json first for the frame timings."
)
DOCLING_ANSWER = (
    "Status of the digital-document-docling video (both sizes): narration recorded, audio_meta.json "
    "in place, frames 04-how.html and 06-pin.html built. Remaining steps (in order): 1. Write "
    "STORYBOARD.md for H and V. 2. Build the remaining frames. 3. Build captions.html. 4. check, "
    "render body.mp4, thumbnails, add_intro.py, name_final.sh. 5. Publish folder."
)


class Model:
    def __init__(self, replies: list[ModelResponse]) -> None:
        self.replies = deque(replies)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.replies.popleft()


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


def handoff(run_id: str, answer: str, *, minutes: int = 0, request: str = "") -> RunHandoff:
    when = utc_now() - timedelta(hours=1) + timedelta(minutes=minutes)
    return RunHandoff(run_id, "default", "completed", when, when, request=request, answer=answer)


class ScoringTests(unittest.TestCase):
    def test_the_docling_prompt_links_the_docling_run(self) -> None:
        distinct, _ = prompt_terms(PROMPT)
        self.assertIn("digital-document-docling", distinct)
        self.assertIn("add_intro.py", distinct)
        docling = handoff("d" * 32, DOCLING_ANSWER, request="use skill /aisystems-video for digital-document-docling")
        rag = handoff("a" * 32, "agentic-rag: captions.html and audio_meta.json done; add_intro.py ran.", minutes=5)
        link = find_link(PROMPT, [rag, docling])
        assert link is not None
        self.assertEqual(link.handoff.run_id, "d" * 32)  # more distinctive terms, though older
        self.assertIn("digital-document-docling", link.terms)

    def test_unrelated_or_term_free_prompts_link_nothing(self) -> None:
        docling = handoff("d" * 32, DOCLING_ANSWER)
        self.assertIsNone(find_link("What is the capital of France?", [docling]))
        self.assertIsNone(find_link("continue please", [docling]))  # no identifiers: use chat --continue
        self.assertIsNone(find_link("render the chain-of-thought video", [docling]))

    def test_a_run_that_left_nothing_is_not_linked(self) -> None:
        # The failed retry (6fed5ce7) had the same request and nothing else: it must not win.
        docling = handoff("d" * 32, DOCLING_ANSWER)
        retry = handoff("e" * 32, "", minutes=30, request=PROMPT)
        link = find_link(PROMPT, [retry, docling])
        assert link is not None
        self.assertEqual(link.handoff.run_id, "d" * 32)

    def test_ties_go_to_the_newest_and_old_runs_are_ignored(self) -> None:
        old = handoff("o" * 32, DOCLING_ANSWER)
        new = handoff("n" * 32, DOCLING_ANSWER, minutes=10)
        link = find_link(PROMPT, [old, new])
        assert link is not None
        self.assertEqual(link.handoff.run_id, "n" * 32)
        stale = handoff("s" * 32, DOCLING_ANSWER, minutes=-60 * 24 * 40)
        self.assertIsNone(find_link(PROMPT, [stale]))
        self.assertIsNone(find_link(PROMPT, [new], exclude="n" * 32))


class LinkingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.store = SQLiteRunStore(Path(self.folder.name) / "runs.db")

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.folder.cleanup()

    def orchestrator(self, model: Model) -> AgentOrchestrator:
        return AgentOrchestrator(models=model, tools=ToolRuntime(ToolRegistry()), store=self.store)

    async def finished_run(self, prompt: str, answer: str) -> str:
        orch = self.orchestrator(Model([text(answer)]))
        run = await orch.create_run(model_profile="p", messages=[Message(Role.SYSTEM, "S"), Message(Role.USER, prompt)])
        done = await orch.run(run.id)
        self.assertEqual(done.status, RunStatus.COMPLETED)
        return run.id

    async def test_a_continue_prompt_starts_with_the_earlier_runs_handoff(self) -> None:
        earlier = await self.finished_run("make the digital-document-docling video", DOCLING_ANSWER)
        saved = await self.store.get_handoff(earlier)  # written when the turn ended
        assert saved is not None
        self.assertEqual(saved.answer, DOCLING_ANSWER)
        await self.finished_run("explain the agentic-rag page", "agentic-rag is about tool-using retrieval.")

        model = Model([text("writing STORYBOARD.md")])
        orch = self.orchestrator(model)
        run = await orch.create_run(model_profile="p", link=True,
                                    messages=[Message(Role.SYSTEM, "S"), Message(Role.USER, PROMPT)])
        await orch.run(run.id)
        sent = model.requests[0].messages
        self.assertEqual([m.role for m in sent], [Role.SYSTEM, Role.SYSTEM, Role.USER])
        self.assertTrue(sent[1].content.startswith(f"[Earlier run {earlier[:8]} of this project"))
        self.assertIn("Remaining steps (in order)", sent[1].content)
        events = await self.store.list_events(run.id)
        linked = [e.payload for e in events if e.type is EventType.RUN_LINKED]
        self.assertEqual(linked[0]["linked_run_id"], earlier)
        self.assertEqual(linked_note(events), f"linked to earlier run {earlier[:8]} "
                                               f"({', '.join(linked[0]['terms'][:3])})")
        final = await self.store.get_run(run.id)
        assert final is not None
        self.assertEqual(derive_message_skeleton(events), message_skeleton(final.messages))

    async def test_no_link_without_the_flag_or_a_match(self) -> None:
        await self.finished_run("make the digital-document-docling video", DOCLING_ANSWER)
        for link, prompt in ((False, PROMPT), (True, "What is the capital of France?")):
            model = Model([text("ok")])
            orch = self.orchestrator(model)
            run = await orch.create_run(model_profile="p", link=link,
                                        messages=[Message(Role.SYSTEM, "S"), Message(Role.USER, prompt)])
            await orch.run(run.id)
            self.assertEqual([m.role for m in model.requests[0].messages], [Role.SYSTEM, Role.USER])
            self.assertIsNone(linked_note(await self.store.list_events(run.id)))

    async def test_runs_from_before_handoffs_are_backfilled(self) -> None:
        earlier = await self.finished_run("make the digital-document-docling video", DOCLING_ANSWER)
        self.store._connection.execute("DELETE FROM run_handoffs")  # as if made before the upgrade
        self.assertIsNone(await self.store.get_handoff(earlier))
        found = await ensure_handoffs(self.store, "default")
        self.assertEqual([h.run_id for h in found], [earlier])
        self.assertIsNotNone(await self.store.get_handoff(earlier))

    async def test_previous_runs_lists_matches_and_shows_one(self) -> None:
        earlier = await self.finished_run("make the digital-document-docling video", DOCLING_ANSWER)
        await self.finished_run("explain the agentic-rag page", "agentic-rag is about tool-using retrieval.")
        tool = previous_runs_tool(self.store)
        listed = await tool({})
        self.assertEqual(len(listed["runs"]), 2)
        matched = await tool({"query": "digital-document-docling"})
        self.assertEqual([r["run_id"] for r in matched["runs"]], [earlier])
        shown = await tool({"run_id": earlier[:8]})
        self.assertIn("Remaining steps (in order)", shown["run"])
        self.assertIn("error", await tool({"run_id": "abc"}))
        token = CURRENT_RUN_ID.set(earlier)  # a run does not list itself
        try:
            self.assertNotIn(earlier, [r["run_id"] for r in (await tool({}))["runs"]])
        finally:
            CURRENT_RUN_ID.reset(token)
        self.assertIs(PREVIOUS_RUNS_SPEC.trust.value, "trusted")


class BuildHandoffTests(unittest.TestCase):
    def test_request_answer_todos_files_and_notes_skipped(self) -> None:
        from agent_harness.domain import RunRecord

        run = RunRecord(id="r" * 32, model_profile="p", status=RunStatus.BUDGET_EXHAUSTED, messages=[
            Message(Role.SYSTEM, "S"),
            Message(Role.USER, "make the video"),
            Message(Role.ASSISTANT, "", tool_calls=(ToolCall("c1", "apply_patch", {"path": "v/BRIEF.md"}),)),
            Message(Role.TOOL, "ok", tool_call_id="c1"),
            Message(Role.USER, "[Plan] Before answering, call create_plan."),
            Message(Role.ASSISTANT, "brief written; storyboard next"),
        ], todos=(TodoItem("1", "write brief", TodoStatus.COMPLETED), TodoItem("2", "storyboard")))
        built = build_handoff(run)
        self.assertEqual((built.request, built.latest_request), ("make the video", ""))
        self.assertEqual(built.answer, "brief written; storyboard next")
        self.assertEqual(built.todos, ("[completed] write brief", "[pending] storyboard"))
        self.assertEqual(built.files, ("v/BRIEF.md",))
        self.assertEqual(built.status, "budget_exhausted")


class ChatResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_resume_and_slash_resume(self) -> None:
        store = InMemoryRunStore()
        model = Model([text("first answer"), text("second answer"), text("after resume")])
        orch = AgentOrchestrator(models=model, tools=ToolRuntime(ToolRegistry()), store=store)
        app = SimpleNamespace(orchestrator=orch, store=store, templates={}, run_versions=lambda p: None)
        notes: list[str] = []
        lines = deque(["hello", "/new", "other topic", "/resume", "/resume 2", "and again", "/exit"])
        # /resume lists the latest used first: "other topic" (1), then "hello" (2).

        class IO:
            async def ask(self, prompt: str) -> str | None:
                return lines.popleft() if lines else None

            def say(self, text: str) -> None:
                pass

            def note(self, text: str) -> None:
                notes.append(text)

        session = ChatSession(app, ChatOptions(profile="p"), IO())
        await session.loop()
        page = await store.list_runs(__import__("agent_harness.domain", fromlist=["RunQuery"]).RunQuery())
        self.assertEqual(len(page.runs), 2)
        first = next(r for r in page.runs if r.title.startswith("hello"))
        self.assertEqual(session.run_id, first.id)  # /resume 2 picked the older one
        self.assertTrue(any(n.startswith(f"resumed run {first.id}") for n in notes), notes)
        self.assertEqual([m.content for m in model.requests[-1].messages if m.role is Role.USER],
                         ["hello", "and again"])

        latest = ChatSession(app, ChatOptions(profile="p"), IO())
        self.assertTrue(await latest.resume())  # chat --continue
        self.assertEqual(latest.run_id, first.id)  # the conversation used last, not the newest
        self.assertFalse(await latest.resume("zzzzzzzz"))


class RootConfigTests(unittest.TestCase):
    def test_linking_is_on_and_previous_runs_is_core(self) -> None:
        config = load_config(ROOT / "models.toml")
        self.assertTrue(config.project.link_runs)
        self.assertIn("previous_runs", config.tool_loading.core)
        import tomllib

        raw = tomllib.loads((ROOT / "models.toml").read_text())  # modal_shared is off until .env enables it
        self.assertEqual(raw["models"]["modal_shared"]["parameters"]["repetition_penalty"], 1.05)


if __name__ == "__main__":
    unittest.main()
