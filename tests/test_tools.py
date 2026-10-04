from __future__ import annotations

import asyncio
import json
import unittest

from agent_harness.domain import (
    FinishReason,
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
)
from agent_harness.model_gateway import ScriptedModelClient
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.tools import (
    RetryableToolError,
    SchemaDefinitionError,
    ToolRegistry,
    ToolRuntime,
    create_default_runtime,
)


def spec(
    name: str = "test",
    *,
    risk: ToolRisk = ToolRisk.READ_ONLY,
    timeout: float = 1.0,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="A test tool.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        timeout_seconds=timeout,
        risk=risk,
    )


class RegistryTests(unittest.TestCase):
    def test_registration_is_explicit_and_duplicates_are_rejected(self) -> None:
        registry = ToolRegistry()

        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            return arguments

        registry.register(spec(), handler)
        self.assertEqual([item.name for item in registry.specs()], ["test"])
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(spec(), handler)

    def test_unsupported_schema_is_rejected_at_registration(self) -> None:
        registry = ToolRegistry()

        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            return arguments

        invalid = ToolSpec(
            name="bad",
            description="bad schema",
            input_schema={"type": "object", "oneOf": []},
        )
        with self.assertRaises(SchemaDefinitionError):
            registry.register(invalid, handler)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_example_tools_succeed_and_validate_schema(self) -> None:
        runtime = create_default_runtime()
        result = await runtime.execute(
            ToolCall("c1", "echo", {"text": "hello"}), idempotency_key=""
        )
        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(result.output, {"text": "hello"})
        self.assertEqual(result.model_message, "")

        invalid = await runtime.execute(
            ToolCall("c2", "echo", {"text": 7}), idempotency_key=""
        )
        self.assertEqual(invalid.status, ToolResultStatus.VALIDATION_ERROR)
        self.assertIn("expected string", invalid.model_message)

    async def test_orchestrator_sends_successful_output_to_next_model_turn(self) -> None:
        model = ScriptedModelClient(
            [
                ModelResponse(
                    tool_calls=(ToolCall("call-1", "echo", {"text": "hello"}),),
                    finish_reason=FinishReason.TOOL_CALLS,
                ),
                ModelResponse(content="done", finish_reason=FinishReason.STOP),
            ]
        )

        class Resolver:
            def resolve(self, profile: str) -> ScriptedModelClient:
                self.profile = profile
                return model

        orchestrator = AgentOrchestrator(
            models=Resolver(),
            tools=create_default_runtime(),
            store=InMemoryRunStore(),
        )
        run = await orchestrator.create_run(
            model_profile="test",
            messages=[Message(role=Role.USER, content="echo hello")],
            run_id="tool-output-integration",
        )
        completed = await orchestrator.run(run.id)

        self.assertEqual(len(model.requests), 2)
        tool_messages = [
            message
            for message in model.requests[1].messages
            if message.role is Role.TOOL
        ]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(
            json.loads(tool_messages[0].content),
            {"output": {"text": "hello"}, "status": "success"},
        )
        self.assertEqual(completed.messages[-1].content, "done")

    async def test_unknown_tool_is_normalized(self) -> None:
        runtime = ToolRuntime(ToolRegistry())
        result = await runtime.execute(
            ToolCall("c1", "missing", {}), idempotency_key=""
        )
        self.assertEqual(result.status, ToolResultStatus.PERMANENT_ERROR)
        self.assertEqual(result.output["error"], "unknown_tool")

    async def test_invalid_call_envelope_is_rejected(self) -> None:
        runtime = create_default_runtime()
        result = await runtime.execute(
            ToolCall("", "echo", {"text": "hello"}), idempotency_key=""
        )
        self.assertEqual(result.status, ToolResultStatus.VALIDATION_ERROR)
        self.assertEqual(result.output["error"], "invalid_call_id")

        non_json = await runtime.execute(
            ToolCall("c2", "echo", {"text": float("nan")}), idempotency_key=""
        )
        self.assertEqual(non_json.status, ToolResultStatus.VALIDATION_ERROR)

    async def test_timeout_cancels_handler_and_is_retryable(self) -> None:
        cancelled = asyncio.Event()

        async def slow(arguments: dict[str, object]) -> dict[str, object]:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return arguments

        registry = ToolRegistry()
        registry.register(spec(timeout=0.01), slow)
        result = await ToolRuntime(registry).execute(
            ToolCall("c1", "test", {"value": "x"}), idempotency_key=""
        )
        self.assertEqual(result.status, ToolResultStatus.TIMEOUT)
        self.assertTrue(result.retryable)
        self.assertTrue(cancelled.is_set())

    async def test_policy_requires_approval_for_high_risk_tool(self) -> None:
        calls = 0

        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return arguments

        registry = ToolRegistry()
        registry.register(spec(risk=ToolRisk.HIGH_RISK), handler)
        runtime = ToolRuntime(registry)
        denied = await runtime.execute(
            ToolCall("c1", "test", {"value": "x"}),
            idempotency_key="key",
        )
        self.assertEqual(denied.status, ToolResultStatus.APPROVAL_REQUIRED)
        self.assertEqual(calls, 0)

        allowed = await runtime.execute(
            ToolCall("c2", "test", {"value": "x"}),
            idempotency_key="key",
            approved=True,
        )
        self.assertEqual(allowed.status, ToolResultStatus.SUCCESS)
        self.assertEqual(calls, 1)

    async def test_side_effect_requires_idempotency_key_and_replays_result(self) -> None:
        calls = 0

        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {"sequence": calls}

        registry = ToolRegistry()
        registry.register(spec(risk=ToolRisk.SIDE_EFFECT), handler)
        runtime = ToolRuntime(registry)
        missing = await runtime.execute(
            ToolCall("c1", "test", {"value": "x"}), idempotency_key="   "
        )
        self.assertEqual(missing.status, ToolResultStatus.VALIDATION_ERROR)
        self.assertEqual(missing.output["error"], "idempotency_key_required")

        first = await runtime.execute(
            ToolCall("c2", "test", {"value": "x"}), idempotency_key="same"
        )
        replay = await runtime.execute(
            ToolCall("c3", "test", {"value": "x"}), idempotency_key="same"
        )
        self.assertEqual(first.output, {"sequence": 1})
        self.assertEqual(replay.output, {"sequence": 1})
        self.assertEqual(replay.call_id, "c3")
        self.assertEqual(calls, 1)

        conflict = await runtime.execute(
            ToolCall("c4", "test", {"value": "different"}), idempotency_key="same"
        )
        self.assertEqual(conflict.status, ToolResultStatus.VALIDATION_ERROR)
        self.assertEqual(conflict.output["error"], "idempotency_key_conflict")

    async def test_concurrent_idempotent_calls_execute_once(self) -> None:
        calls = 0

        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return {"sequence": calls}

        registry = ToolRegistry()
        registry.register(spec(risk=ToolRisk.SIDE_EFFECT), handler)
        runtime = ToolRuntime(registry)
        first, second = await asyncio.gather(
            runtime.execute(
                ToolCall("c1", "test", {"value": "x"}), idempotency_key="same"
            ),
            runtime.execute(
                ToolCall("c2", "test", {"value": "x"}), idempotency_key="same"
            ),
        )
        self.assertEqual(first.output, second.output)
        self.assertEqual(calls, 1)

    async def test_output_size_is_limited(self) -> None:
        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            return {"payload": "x" * 100}

        registry = ToolRegistry()
        registry.register(spec(), handler)
        result = await ToolRuntime(registry, max_output_bytes=20).execute(
            ToolCall("c1", "test", {"value": "x"}), idempotency_key=""
        )
        self.assertEqual(result.status, ToolResultStatus.PERMANENT_ERROR)
        self.assertEqual(result.output["error"], "output_too_large")

    async def test_failures_are_sanitized_and_classified(self) -> None:
        async def handler(arguments: dict[str, object]) -> dict[str, object]:
            raise RetryableToolError("secret diagnostic")

        registry = ToolRegistry()
        registry.register(spec(), handler)
        result = await ToolRuntime(registry).execute(
            ToolCall("c1", "test", {"value": "x"}), idempotency_key=""
        )
        self.assertEqual(result.status, ToolResultStatus.RETRYABLE_ERROR)
        self.assertTrue(result.retryable)
        self.assertNotIn("secret", result.model_message)


if __name__ == "__main__":
    unittest.main()
