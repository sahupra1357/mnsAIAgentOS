"""Network-free tests for the optional LiteLLM adapter (a fake ``litellm`` module)."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness.config.loader import ConfigError, HarnessConfig, ModelPricing, ModelProfile, load_config
from agent_harness.domain import (
    ContextWindowExceededError,
    FinishReason,
    Message,
    ModelRequest,
    Role,
    RunStatus,
    ToolCall,
    ToolSpec,
)
from agent_harness.model_gateway import (
    AuthenticationError,
    LiteLLMClient,
    MalformedResponseError,
    ModelConfigurationError,
    ModelGateway,
    ModelTimeoutError,
    ProviderError,
    RetryableProviderError,
    ThrottlingError,
    classify_litellm_error,
)
from agent_harness.model_gateway.diagnostics import CheckStatus, diagnose
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.tools import create_default_runtime


# Exception classes named like LiteLLM's (mapping is by class name along the MRO).
class APIError(Exception):
    def __init__(self, message: str = "", status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationErr(APIError):
    pass


AuthenticationErr.__name__ = "AuthenticationError"


class RateLimitError(APIError):
    pass


class ServiceUnavailableError(APIError):
    pass


class BadRequestError(APIError):
    pass


class ContextWindowExceeded(BadRequestError):
    pass


ContextWindowExceeded.__name__ = "ContextWindowExceededError"


class Timeout(APIError):
    pass


class FakeResponse:
    def __init__(self, payload: dict, cost: float | None = None) -> None:
        self._payload = payload
        self._hidden_params = {"response_cost": cost} if cost is not None else {}

    def model_dump(self) -> dict:
        return self._payload


def completion(content: str = "hello", *, tool_calls: list | None = None, usage: dict | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-test",
        "choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 3},
    }


def fake_litellm(outcomes: list, *, cost_calculator=None) -> types.ModuleType:
    module = types.ModuleType("litellm")
    module.calls = []  # type: ignore[attr-defined]
    queue = deque(outcomes)

    async def acompletion(**kwargs: Any) -> Any:
        module.calls.append(kwargs)  # type: ignore[attr-defined]
        outcome = queue.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    module.acompletion = acompletion  # type: ignore[attr-defined]
    if cost_calculator is not None:
        module.completion_cost = cost_calculator  # type: ignore[attr-defined]
    return module


def profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {
        "adapter": "litellm",
        "model": "anthropic/claude-test",
        "endpoint": "",
        "api_key_env": "ANTHROPIC_KEY",
        "max_retries": 0,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ModelProfile(**values)


ENV = {"ANTHROPIC_KEY": "sk-test-value"}


class ConfigTests(unittest.TestCase):
    def test_litellm_profiles_may_omit_endpoint_others_may_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(
                '[models.claude]\nadapter = "litellm"\nmodel = "anthropic/claude-test"\n'
                'api_key_env = "ANTHROPIC_KEY"\n',
                encoding="utf-8",
            )
            config = load_config(path, environ={})
            self.assertEqual(config.models["claude"].endpoint, "")
            path.write_text('[models.local]\nadapter = "ollama"\nmodel = "m"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "missing: endpoint"):
                load_config(path, environ={})
        with self.assertRaises(ConfigError):
            ModelProfile(adapter="openai_compatible", model="m", endpoint="")

    def test_gateway_builds_litellm_client_only_for_litellm_profiles(self) -> None:
        config = HarnessConfig(
            models={
                "claude": profile(),
                "local": ModelProfile(adapter="ollama", model="q", endpoint="http://127.0.0.1:11434/v1"),
            },
            default_model_profile="local",
        )
        with mock.patch.dict(sys.modules, {"litellm": None}):  # import would fail if attempted
            gateway = ModelGateway(config, environ=ENV)
            self.assertNotIsInstance(gateway.resolve("local"), LiteLLMClient)
            self.assertIsInstance(gateway.resolve("claude"), LiteLLMClient)


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_arguments_and_normalization(self) -> None:
        module = fake_litellm(
            [
                FakeResponse(
                    completion(
                        content="",
                        tool_calls=[{"id": "t1", "type": "function",
                                     "function": {"name": "lookup", "arguments": '{"q":"x"}'}}],
                        usage={"prompt_tokens": 50, "completion_tokens": 5,
                               "prompt_tokens_details": {"cached_tokens": 40}},
                    ),
                    cost=0.0123,
                )
            ]
        )
        client = LiteLLMClient(
            profile(endpoint="https://gateway.test/v1/", headers_env={"X-Team": "TEAM"},
                    parameters={"temperature": 0.1}, timeout_seconds=45),
            module=module,
            environ={**ENV, "TEAM": "agents"},
        )
        response = await client.complete(
            ModelRequest(
                messages=(Message(Role.USER, "hi"),),
                tools=(ToolSpec(name="lookup", description="d", input_schema={"type": "object"}),),
                max_output_tokens=200,
            )
        )

        call = module.calls[0]
        self.assertEqual(call["model"], "anthropic/claude-test")
        self.assertEqual(call["num_retries"], 0)
        self.assertEqual(call["timeout"], 45)
        self.assertEqual(call["api_key"], "sk-test-value")
        self.assertEqual(call["api_base"], "https://gateway.test/v1")
        self.assertEqual(call["extra_headers"], {"X-Team": "agents"})
        self.assertEqual((call["temperature"], call["max_tokens"]), (0.1, 200))
        self.assertEqual(call["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(call["tools"][0]["function"]["name"], "lookup")
        self.assertEqual(response.tool_calls, (ToolCall("t1", "lookup", {"q": "x"}),))
        self.assertEqual(response.finish_reason, FinishReason.TOOL_CALLS)
        self.assertEqual((response.usage.input_tokens, response.usage.cached_input_tokens), (50, 40))
        self.assertEqual(response.usage.cost_usd, 0.0123)
        self.assertEqual(response.metadata["cost_source"], "litellm")

    async def test_no_endpoint_means_no_api_base_and_missing_key_fails_first(self) -> None:
        module = fake_litellm([FakeResponse(completion())])
        await LiteLLMClient(profile(), module=module, environ=ENV).complete(ModelRequest(messages=()))
        self.assertNotIn("api_base", module.calls[0])
        with self.assertRaisesRegex(ModelConfigurationError, "ANTHROPIC_KEY"):
            await LiteLLMClient(profile(), module=fake_litellm([]), environ={}).complete(
                ModelRequest(messages=())
            )

    async def test_cost_sources(self) -> None:
        calculated = fake_litellm([FakeResponse(completion())], cost_calculator=lambda **_: 0.5)
        response = await LiteLLMClient(profile(), module=calculated, environ=ENV).complete(ModelRequest(messages=()))
        self.assertEqual(response.usage.cost_usd, 0.5)

        def unknown_model(**_: Any) -> float:
            raise ValueError("model not mapped")

        unpriced = fake_litellm([FakeResponse(completion())], cost_calculator=unknown_model)
        response = await LiteLLMClient(profile(), module=unpriced, environ=ENV).complete(ModelRequest(messages=()))
        self.assertIsNone(response.usage.cost_usd)

        explicit = fake_litellm([FakeResponse(completion(), cost=9.0)])
        client = LiteLLMClient(
            profile(pricing=ModelPricing(input_usd_per_million_tokens=1_000_000.0)),
            module=explicit, environ=ENV,
        )
        response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.usage.cost_usd, 10.0)  # profile pricing wins over LiteLLM's

    async def test_retries_are_the_profiles_not_litellms(self) -> None:
        module = fake_litellm([ServiceUnavailableError("overloaded"), RateLimitError("slow"), FakeResponse(completion("ok"))])
        response = await LiteLLMClient(profile(max_retries=2), module=module, environ=ENV).complete(
            ModelRequest(messages=())
        )
        self.assertEqual(response.content, "ok")
        self.assertEqual(len(module.calls), 3)
        self.assertTrue(all(call["num_retries"] == 0 for call in module.calls))

        exhausted = fake_litellm([Timeout("t1"), Timeout("t2")])
        with self.assertRaises(ModelTimeoutError) as caught:
            await LiteLLMClient(profile(max_retries=1), module=exhausted, environ=ENV).complete(ModelRequest(messages=()))
        self.assertEqual(caught.exception.provider_attempts, 2)

        permanent = fake_litellm([AuthenticationErr("bad key"), FakeResponse(completion())])
        with self.assertRaises(AuthenticationError):
            await LiteLLMClient(profile(max_retries=3), module=permanent, environ=ENV).complete(ModelRequest(messages=()))
        self.assertEqual(len(permanent.calls), 1)

    def test_error_classification(self) -> None:
        cases = [
            (AuthenticationErr("x"), AuthenticationError),
            (RateLimitError("x"), ThrottlingError),
            (Timeout("x"), ModelTimeoutError),
            (ServiceUnavailableError("x"), RetryableProviderError),
            (ContextWindowExceeded("x"), ContextWindowExceededError),
            (BadRequestError("x"), ProviderError),
            (APIError("x", status_code=502), RetryableProviderError),
            (APIError("x", status_code=429), ThrottlingError),
            (APIError("x", status_code=400), ProviderError),
            (RuntimeError("x"), ProviderError),
        ]
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assertIsInstance(classify_litellm_error(error), expected)

    async def test_lazy_import_and_clear_install_error(self) -> None:
        client = LiteLLMClient(profile(), environ=ENV)
        with mock.patch.dict(sys.modules, {"litellm": None}):
            with self.assertRaisesRegex(ModelConfigurationError, r"agent-harness\[litellm\]"):
                await client.complete(ModelRequest(messages=()))
        installed = fake_litellm([FakeResponse(completion("from sys.modules"))])
        with mock.patch.dict(sys.modules, {"litellm": installed}):
            response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.content, "from sys.modules")

    async def test_malformed_response_is_classified(self) -> None:
        module = fake_litellm([object()])
        with self.assertRaises(MalformedResponseError):
            await LiteLLMClient(profile(), module=module, environ=ENV).complete(ModelRequest(messages=()))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_text_and_assembles_final_response(self) -> None:
        async def chunks() -> AsyncIterator[Any]:
            yield FakeResponse({"choices": [{"delta": {"content": "Hel"}}]})
            yield {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}
            yield {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}

        module = fake_litellm([chunks()])
        events = [e async for e in LiteLLMClient(profile(), module=module, environ=ENV).stream(ModelRequest(messages=()))]

        self.assertEqual([e.text for e in events if e.text], ["Hel", "lo"])
        final = events[-1].response
        assert final is not None
        self.assertEqual((final.content, final.usage.input_tokens), ("Hello", 7))
        self.assertIs(module.calls[0]["stream"], True)
        self.assertEqual(module.calls[0]["stream_options"], {"include_usage": True})

    async def test_stream_errors_are_classified(self) -> None:
        async def failing() -> AsyncIterator[Any]:
            yield {"choices": [{"delta": {"content": "par"}}]}
            raise ServiceUnavailableError("dropped")

        with self.assertRaises(RetryableProviderError):
            async for _ in LiteLLMClient(profile(), module=fake_litellm([failing()]), environ=ENV).stream(
                ModelRequest(messages=())
            ):
                pass


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_from_local_profile_to_litellm_profile(self) -> None:
        module = fake_litellm([FakeResponse(completion("answered by fallback"))])

        class Local:
            async def complete(self, request: ModelRequest):
                error = RetryableProviderError("local server down")
                error.provider_attempts = 3
                raise error

        clients = {"local": Local(), "claude": LiteLLMClient(profile(), module=module, environ=ENV)}

        class Resolver:
            def resolve(self, name: str):
                return clients[name]

        orchestrator = AgentOrchestrator(
            models=Resolver(), tools=create_default_runtime(), store=InMemoryRunStore(),
            fallbacks=lambda name: ("claude",) if name == "local" else (),
        )
        run = await orchestrator.create_run(model_profile="local", messages=[Message(Role.USER, "hi")])
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.model_profile, "claude")
        self.assertEqual(finished.messages[-1].content, "answered by fallback")

    async def test_doctor_skips_http_probes(self) -> None:
        module = fake_litellm([FakeResponse(completion("OK"))])
        checks = await diagnose(
            LiteLLMClient(profile(), module=module, environ=ENV),
            transport=object(),  # type: ignore[arg-type]
            check_tools=False,
            check_cache=False,
        )
        self.assertEqual(
            [(c.name, c.status) for c in checks],
            [("credentials", CheckStatus.OK), ("health", CheckStatus.SKIP), ("completion", CheckStatus.OK)],
        )


if __name__ == "__main__":
    unittest.main()
