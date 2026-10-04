"""Network-free tests for Phase 6e: file snapshots and rewind."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from typing import Any

from agent_harness.server.api import HarnessAPI
from agent_harness.application import build_application
from agent_harness.knowledge.artifacts import InMemoryArtifactStore
from agent_harness.config.loader import HarnessConfig, ModelProfile
from agent_harness.domain import (
    EventType,
    FinishReason,
    JsonObject,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.orchestration.snapshots import WorkspaceSnapshotter
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime

WRITE = ToolSpec(
    name="write_file", description="write",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}},
    risk=ToolRisk.SIDE_EFFECT, idempotent=False, writes_paths=("path",),
)
NOTIFY = ToolSpec(name="notify", description="send", input_schema={"type": "object"},
                  risk=ToolRisk.SIDE_EFFECT, idempotent=False)


class Model:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = deque(responses)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.popleft()


def write(path: str, text: str, call_id: str) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, "write_file", {"path": path, "text": text}),),
                         finish_reason=FinishReason.TOOL_CALLS)


def say(text: str) -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP)


def tools(workspace: Path, approval: ApprovalMode = ApprovalMode.ELEVATED) -> ToolRuntime:
    async def write_file(arguments: JsonObject) -> JsonObject:
        (workspace / arguments["path"]).write_text(arguments["text"])
        return {"written": arguments["path"]}

    async def notify(arguments: JsonObject) -> JsonObject:
        return {"sent": True}

    registry = ToolRegistry()
    registry.register(WRITE, write_file)
    registry.register(NOTIFY, notify)
    return ToolRuntime(registry, policy=DefaultToolPolicy(approval))


class Workspace:
    def __init__(self, directory: str) -> None:
        self.root = Path(directory, "ws")
        self.root.mkdir()
        self.artifacts = InMemoryArtifactStore()
        self.snapshotter = WorkspaceSnapshotter(self.root, self.artifacts)

    def orchestrator(self, model: Model, store: Any = None, **kwargs: Any) -> tuple[AgentOrchestrator, Any]:
        store = store or InMemoryRunStore()
        return AgentOrchestrator(models=model, tools=kwargs.pop("tools", tools(self.root)), store=store,
                                 snapshots=self.snapshotter, **kwargs), store

    def read(self, name: str) -> str | None:
        path = self.root / name
        return path.read_text() if path.exists() else None


class SnapshotterTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            (ws.root / "a.txt").write_text("original")
            (ws.root / "big.bin").write_bytes(b"x" * 20)
            (ws.root / "link").symlink_to(ws.root / "a.txt")
            small = WorkspaceSnapshotter(ws.root, ws.artifacts, max_file_bytes=10)
            entries = await small.capture(["a.txt", "new.txt", "../escape", "link", "big.bin", "a.txt"])
            self.assertEqual([e["path"] for e in entries], ["a.txt", "new.txt", "../escape", "link", "big.bin"])
            self.assertTrue(entries[0]["existed"])
            self.assertFalse(entries[1]["existed"])
            self.assertIn("error", entries[2])
            self.assertIn("error", entries[3])
            self.assertEqual(entries[4]["skipped"], "too_large")

            (ws.root / "a.txt").write_text("changed")
            (ws.root / "new.txt").write_text("created later")
            notes = await small.restore(entries)
            self.assertEqual(ws.read("a.txt"), "original")
            self.assertIsNone(ws.read("new.txt"))
            self.assertIn("not restored (no snapshot): big.bin", notes)
            self.assertEqual([p.name for p in ws.root.iterdir() if p.name.startswith(".rewind-")], [])


class RewindTests(unittest.IsolatedAsyncioTestCase):
    async def _edited_run(self, ws: Workspace, **kwargs: Any):
        (ws.root / "a.txt").write_text("v1")
        model = Model(
            write("a.txt", "v2", "c1"),
            write("b.txt", "new", "c2"),
            write("a.txt", "v3", "c3"),
            say("all edits done"),
            say("switched approach"),
        )
        orchestrator, store = ws.orchestrator(model, **kwargs)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "edit files")])
        finished = await orchestrator.run(run.id)
        return orchestrator, store, model, finished

    async def test_dry_run_then_rewind_restores_files_and_archives_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            orchestrator, store, model, finished = await self._edited_run(ws)
            self.assertEqual((ws.read("a.txt"), ws.read("b.txt")), ("v3", "new"))

            plan = await orchestrator.rewind(finished.id, 2)
            self.assertFalse(plan.applied)
            self.assertEqual(plan.files, [{"path": "b.txt", "action": "delete"}, {"path": "a.txt", "action": "restore"}])
            self.assertEqual(ws.read("a.txt"), "v3")  # dry run changed nothing

            applied = await orchestrator.rewind(finished.id, 2, confirm=True)
            self.assertTrue(applied.applied)
            self.assertEqual((ws.read("a.txt"), ws.read("b.txt")), ("v2", None))
            rewound = await store.get_run(finished.id)
            self.assertEqual(rewound.status, RunStatus.COMPLETED)
            self.assertEqual(len(rewound.messages) + len(rewound.archived_messages), len(finished.messages))
            self.assertEqual([m.role for m in rewound.messages], [Role.USER, Role.ASSISTANT, Role.TOOL])
            events = await store.list_events(finished.id)
            self.assertEqual(sum(e.type is EventType.FILE_SNAPSHOT for e in events), 3)
            self.assertEqual(events[-1].type, EventType.REWOUND)

            await orchestrator.continue_run(finished.id, "Use a different approach for b")
            done = await orchestrator.run(finished.id)
            self.assertEqual(done.messages[-1].content, "switched approach")
            sent = [m.content for m in model.requests[-1].messages if m.role is Role.ASSISTANT]
            self.assertEqual(len(sent), 1)  # only step 1's assistant message remains visible

    async def test_second_rewind_goes_further_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            orchestrator, _, model, finished = await self._edited_run(ws)
            await orchestrator.rewind(finished.id, 3, confirm=True)
            self.assertEqual(ws.read("a.txt"), "v2")
            model.responses.extendleft([say("ok"), write("c.txt", "later", "c9")])  # reversed on insert
            await orchestrator.continue_run(finished.id, "also add c")
            await orchestrator.run(finished.id)
            self.assertEqual(ws.read("c.txt"), "later")
            await orchestrator.rewind(finished.id, 1, confirm=True)
            self.assertEqual((ws.read("a.txt"), ws.read("b.txt"), ws.read("c.txt")), ("v1", None, None))

    async def test_irreversible_side_effects_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            model = Model(
                write("a.txt", "x", "c1"),
                ModelResponse(tool_calls=(ToolCall("n1", "notify", {}),), finish_reason=FinishReason.TOOL_CALLS),
                say("done"),
            )
            orchestrator, _ = ws.orchestrator(model)
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "go")])
            await orchestrator.run(run.id)
            plan = await orchestrator.rewind(run.id, 1)
            self.assertEqual([i["tool"] for i in plan.irreversible], ["notify"])

    async def test_snapshot_precedes_execution_and_is_not_retaken_on_replay(self) -> None:
        class CrashOnResult(SQLiteRunStore):
            crashed = False

            async def save_checkpoint(self, run, events, *, expected_version):
                if not self.crashed and any(e.type is EventType.TOOL_RESPONDED for e in events):
                    self.crashed = True
                    raise KeyboardInterrupt
                await super().save_checkpoint(run, events, expected_version=expected_version)

        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            (ws.root / "a.txt").write_text("before")
            path = Path(directory, "db.sqlite3")
            store = CrashOnResult(path)
            orchestrator, _ = ws.orchestrator(Model(write("a.txt", "after", "c1")), store=store)
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
            with self.assertRaises(KeyboardInterrupt):
                await orchestrator.run(run.id)
            await store.close()
            self.assertEqual(ws.read("a.txt"), "after")  # the tool ran before the crash

            store = SQLiteRunStore(path)
            orchestrator, _ = ws.orchestrator(Model(say("done")), store=store)
            await orchestrator.run(run.id)
            snapshots = [e for e in await store.list_events(run.id) if e.type is EventType.FILE_SNAPSHOT]
            await orchestrator.rewind(run.id, 1, confirm=True)
            await store.close()
            self.assertEqual(ws.read("a.txt"), "before")  # the pre-image, not the post-image
        self.assertEqual(len(snapshots), 1)

    async def test_rules_and_archived_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            model = Model(say("first"), write("a.txt", "x", "c1"))
            orchestrator, store = ws.orchestrator(model, tools=tools(ws.root, ApprovalMode.SIDE_EFFECTS))
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
            with self.assertRaisesRegex(ValueError, "cancel it or let it finish"):
                await orchestrator.rewind(run.id, 1)
            await orchestrator.run(run.id)
            await orchestrator.continue_run(run.id, "now edit")
            paused = await orchestrator.run(run.id)
            self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
            with self.assertRaisesRegex(ValueError, "between 1 and 2"):
                await orchestrator.rewind(run.id, 3)
            await orchestrator.rewind(run.id, 2, confirm=True)
            self.assertIsNone(await orchestrator.pending_approval(run.id))
            self.assertEqual((await store.get_run(run.id)).status, RunStatus.COMPLETED)


class SurfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_rewind_is_a_dry_run_unless_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(directory)
            (ws.root / "a.txt").write_text("v1")
            orchestrator, store = ws.orchestrator(Model(write("a.txt", "v2", "c1"), say("done")))
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
            await orchestrator.run(run.id)
            profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
            api = HarnessAPI(orchestrator=orchestrator, store=store, queue=store,
                             config=HarnessConfig(models={"p": profile}, default_model_profile="p"), auth_token=None)

            async def post(body: Any) -> tuple[int, Any]:
                sent: list[dict] = []

                async def receive() -> dict:
                    return {"type": "http.request", "body": json.dumps(body).encode()}

                async def send(message: dict) -> None:
                    sent.append(message)

                await api({"type": "http", "method": "POST", "path": f"/runs/{run.id}/rewind", "headers": []},
                          receive, send)
                return sent[0]["status"], json.loads(sent[1]["body"])

            status, plan = await post({"to_step": 1})
            self.assertEqual((status, plan["applied"], ws.read("a.txt")), (200, False, "v2"))
            status, plan = await post({"to_step": 1, "confirm": True})
            self.assertEqual((status, plan["applied"], ws.read("a.txt")), (200, True, "v1"))
            self.assertEqual((await post({"to_step": "1"}))[0], 400)
            self.assertEqual((await post({"to_step": 9}))[0], 409)

    async def test_coding_application_snapshots_apply_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                              '[coding]\nenabled = true\nworkspace = "."\nexecution_runtime = "local"\n'
                              'allow_local_execution = true\n')
            app = build_application(config_path=config, database_path=":memory:")
            async with app:
                self.assertIsInstance(app.orchestrator._snapshots, WorkspaceSnapshotter)
                patch = next(spec for spec in app.tools.specs() if spec.name == "apply_patch")
                self.assertEqual(patch.writes_paths, ("path",))


if __name__ == "__main__":
    unittest.main()
