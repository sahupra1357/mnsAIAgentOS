"""Network-free tests for Phase 4a: durable idempotency, versions, cached tokens."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from agent_harness.application import build_application
from agent_harness.config.loader import ModelProfile
from agent_harness.domain import (
    EventType,
    FinishReason,
    JsonObject,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.model_gateway import HttpResponse, OpenAICompatibleClient
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ToolRegistry, ToolRuntime
from agent_harness.orchestration.versioning import profile_sha256, sha256_of, tools_sha256


SEND = ToolSpec(
    name="send_email",
    description="send an email",
    input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
    risk=ToolRisk.SIDE_EFFECT,
    idempotent=False,
)
USER = Message(role=Role.USER, content="notify the team")


class Crash(BaseException):
    """A worker dying mid-run; not caught by the orchestrator."""


class Mailer:
    def __init__(self) -> None:
        self.sent: list[JsonObject] = []

    async def __call__(self, arguments: JsonObject) -> JsonObject:
        self.sent.append(arguments)
        return {"message_id": f"m{len(self.sent)}"}


def runtime(handler: Mailer, store=None, specs: Sequence[ToolSpec] = (SEND,)) -> ToolRuntime:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec, handler)
    return ToolRuntime(registry, idempotency_store=store)


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "ScriptedModel":
        return self


def send_call() -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall("call-1", "send_email", {"to": "team@example.test"}),),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def done() -> ModelResponse:
    return ModelResponse(content="sent", finish_reason=FinishReason.STOP)


class CrashAfterToolStore(SQLiteRunStore):
    """Crash on the first checkpoint that would record a tool result."""

    crashed = False

    async def save_checkpoint(
        self, run: RunRecord, events: Sequence[RunEvent], *, expected_version: int
    ) -> None:
        if not self.crashed and any(e.type is EventType.TOOL_RESPONDED for e in events):
            self.crashed = True
            raise Crash()
        await super().save_checkpoint(run, events, expected_version=expected_version)


class IdempotencyStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_stores_round_trip_and_first_write_wins(self) -> None:
        first = ToolResult("c1", "send_email", ToolResultStatus.SUCCESS, {"id": 1}, "ok")
        second = ToolResult("c2", "send_email", ToolResultStatus.SUCCESS, {"id": 2}, "ok")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "runs.sqlite3")
            for store in (InMemoryRunStore(), SQLiteRunStore(path)):
                with self.subTest(store=type(store).__name__):
                    self.assertIsNone(await store.get_tool_result("send_email", "k"))
                    await store.put_tool_result("send_email", "k", "fp1", first)
                    await store.put_tool_result("send_email", "k", "fp2", second)
                    self.assertEqual(await store.get_tool_result("send_email", "k"), ("fp1", first))
                    if isinstance(store, SQLiteRunStore):
                        await store.close()
            reopened = SQLiteRunStore(path)
            self.assertEqual(await reopened.get_tool_result("send_email", "k"), ("fp1", first))
            await reopened.close()

    async def test_runtime_restart_does_not_repeat_side_effect(self) -> None:
        store = InMemoryRunStore()
        mailer = Mailer()
        call = ToolCall("c1", "send_email", {"to": "a@example.test"})

        first = await runtime(mailer, store).execute(call, idempotency_key="run:1:c1")
        replay = await runtime(mailer, store).execute(call, idempotency_key="run:1:c1")

        self.assertEqual(first.status, ToolResultStatus.SUCCESS)
        self.assertEqual(replay.output, first.output)
        self.assertEqual(len(mailer.sent), 1)

    async def test_reused_key_with_different_arguments_is_rejected_after_restart(self) -> None:
        store = InMemoryRunStore()
        mailer = Mailer()
        await runtime(mailer, store).execute(
            ToolCall("c1", "send_email", {"to": "a@example.test"}), idempotency_key="k"
        )
        conflict = await runtime(mailer, store).execute(
            ToolCall("c1", "send_email", {"to": "b@example.test"}), idempotency_key="k"
        )
        self.assertEqual(conflict.status, ToolResultStatus.VALIDATION_ERROR)
        self.assertEqual(conflict.output.get("error"), "idempotency_key_conflict")
        self.assertEqual(len(mailer.sent), 1)

    async def test_failed_results_are_not_recorded(self) -> None:
        store = InMemoryRunStore()

        async def failing(arguments: JsonObject) -> JsonObject:
            raise RuntimeError("smtp down")

        registry = ToolRegistry()
        registry.register(SEND, failing)
        result = await ToolRuntime(registry, idempotency_store=store).execute(
            ToolCall("c1", "send_email", {"to": "a"}), idempotency_key="k"
        )
        self.assertNotEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertIsNone(await store.get_tool_result("send_email", "k"))

    async def test_crash_between_side_effect_and_checkpoint_does_not_resend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "runs.sqlite3")
            mailer = Mailer()
            crashing_store = CrashAfterToolStore(path)
            first = AgentOrchestrator(
                models=ScriptedModel([send_call()]),
                tools=runtime(mailer, crashing_store),
                store=crashing_store,
            )
            run = await first.create_run(model_profile="p", messages=[USER])
            with self.assertRaises(Crash):
                await first.run(run.id)
            await crashing_store.close()
            self.assertEqual(len(mailer.sent), 1)

            store = SQLiteRunStore(path)
            second = AgentOrchestrator(
                models=ScriptedModel([done()]),
                tools=runtime(mailer, store),
                store=store,
            )
            finished = await second.run(run.id)
            await store.close()

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(mailer.sent), 1)
        tool_message = next(m for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("m1", tool_message.content)


class VersionTests(unittest.IsolatedAsyncioTestCase):
    def _orchestrator(self, store, specs: Sequence[ToolSpec] = (SEND,), model=None):
        return AgentOrchestrator(
            models=model or ScriptedModel([]), tools=runtime(Mailer(), specs=specs), store=store
        )

    async def test_run_created_records_prompt_tool_and_extra_versions(self) -> None:
        store = InMemoryRunStore()
        messages = [Message(role=Role.SYSTEM, content="be careful"), USER]
        run = await self._orchestrator(store).create_run(
            model_profile="p", messages=messages, versions={"profile_sha256": "abc"}
        )
        versions = (await store.list_events(run.id))[0].payload["versions"]

        self.assertEqual(versions["tools_sha256"], tools_sha256((SEND,)))
        self.assertEqual(versions["profile_sha256"], "abc")
        other = await self._orchestrator(store).create_run(
            model_profile="p", messages=[Message(role=Role.SYSTEM, content="be bold"), USER]
        )
        other_versions = (await store.list_events(other.id))[0].payload["versions"]
        self.assertNotEqual(versions["system_prompt_sha256"], other_versions["system_prompt_sha256"])
        with self.assertRaises(ValueError):
            await self._orchestrator(store).create_run(
                model_profile="p", messages=[USER], versions={"bad": 1}  # type: ignore[dict-item]
            )

    async def test_resume_with_changed_tools_records_drift_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "runs.sqlite3")
            crashing = CrashAfterToolStore(path)
            first = AgentOrchestrator(
                models=ScriptedModel([send_call()]), tools=runtime(Mailer(), crashing), store=crashing
            )
            run = await first.create_run(model_profile="p", messages=[USER])
            with self.assertRaises(Crash):
                await first.run(run.id)
            await crashing.close()

            extra = ToolSpec(name="lookup", description="read", input_schema={"type": "object"})
            store = SQLiteRunStore(path)
            second = AgentOrchestrator(
                models=ScriptedModel([done()]),
                tools=runtime(Mailer(), store, specs=(SEND, extra)),
                store=store,
            )
            await second.run(run.id)
            drift = [e for e in await store.list_events(run.id) if e.type is EventType.CONFIG_DRIFT]
            await store.close()

        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0].payload["field"], "tools_sha256")
        self.assertEqual(drift[0].payload["current"], tools_sha256((SEND, extra)))

    def test_hashes_are_canonical(self) -> None:
        self.assertEqual(sha256_of({"b": 1, "a": {2, 1}}), sha256_of({"a": [1, 2], "b": 1}))
        base = ModelProfile(adapter="a", model="m", endpoint="e", parameters={"t": 1})
        self.assertEqual(
            profile_sha256(base),
            profile_sha256(ModelProfile(adapter="a", model="m", endpoint="e", parameters={"t": 1})),
        )
        self.assertNotEqual(
            profile_sha256(base),
            profile_sha256(ModelProfile(adapter="a", model="m", endpoint="e", parameters={"t": 2})),
        )

    def test_application_attaches_store_and_reports_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text(
                '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n',
                encoding="utf-8",
            )
            app = build_application(config_path=config, database_path=Path(directory, "db.sqlite3"))
            try:
                self.assertIs(app.tools._idempotency_store, app.store)
                versions = app.run_versions("p")
                self.assertEqual(versions["harness_version"], "0.1.0")
                self.assertEqual(versions["profile_sha256"], profile_sha256(app.config.models["p"]))
            finally:
                app.store._connection.close()


class CachedTokenTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_and_events_carry_cached_input_tokens(self) -> None:
        body = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 64},
            },
        }

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                return HttpResponse(200, {}, json.dumps(body).encode())

        client = OpenAICompatibleClient(
            ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1"),
            transport=Transport(),
            environ={},
        )
        response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.usage.cached_input_tokens, 64)

        class Resolver:
            def resolve(self, profile: str) -> OpenAICompatibleClient:
                return client

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=Resolver(), tools=runtime(Mailer()), store=store)
        run = await orchestrator.create_run(model_profile="p", messages=[USER])
        await orchestrator.run(run.id)
        responded = [e for e in await store.list_events(run.id) if e.type is EventType.MODEL_RESPONDED]
        self.assertEqual(responded[0].payload["cached_input_tokens"], 64)

    def test_usage_default_is_unknown(self) -> None:
        self.assertIsNone(Usage().cached_input_tokens)


if __name__ == "__main__":
    unittest.main()
