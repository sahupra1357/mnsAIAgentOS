"""Network-free tests for the Ollama, llama.cpp, and Modal adapters."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.config.loader import ConfigError, HarnessConfig, ModelProfile, load_config
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelRequest,
    Role,
    ToolCall,
    ToolSpec,
)
from agent_harness.model_gateway import (
    AuthenticationError,
    HttpResponse,
    HttpTransportError,
    HttpTransportTimeout,
    LlamaCppClient,
    MalformedResponseError,
    ModalClient,
    ModelConfigurationError,
    ModelGateway,
    OllamaClient,
    OpenAICompatibleClient,
)
from agent_harness.model_gateway.diagnostics import CheckStatus, diagnose


class FakeTransport:
    """Scripted POST and GET outcomes; records every call."""

    def __init__(
        self,
        posts: list[HttpResponse | Exception] | None = None,
        gets: dict[str, HttpResponse | Exception] | None = None,
    ) -> None:
        self.posts = list(posts or [])
        self.gets = dict(gets or {})
        self.calls: list[tuple[str, object, dict[str, str], float]] = []
        self.get_calls: list[tuple[str, dict[str, str], float]] = []

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.calls.append((url, payload, dict(headers), timeout_seconds))
        if not self.posts:
            raise AssertionError("fake transport exhausted")
        outcome = self.posts.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def get(self, url, headers, timeout_seconds):
        self.get_calls.append((url, dict(headers), timeout_seconds))
        outcome = self.gets.get(url)
        if outcome is None:
            raise AssertionError(f"unexpected GET {url}")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def json_response(payload: object, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=json.dumps(payload).encode())


def text_completion(content: str = "OK", **extra: object) -> HttpResponse:
    body: dict[str, object] = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    body.update(extra)
    return json_response(body)


def tool_completion(name: str = "report_status") -> HttpResponse:
    return json_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": name, "arguments": '{"status":"ok"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )


def make_profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "adapter": "openai_compatible",
        "model": "test-model",
        "endpoint": "http://models.test/v1",
        "max_retries": 0,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


def modal_profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "adapter": "modal",
        "model": "org/model",
        "endpoint": "https://ws--app-serve.modal.run/v1",
        "api_key_env": "LLM_KEY",
        "headers_env": {"Modal-Key": "PROXY_ID", "Modal-Secret": "PROXY_SECRET"},
        "timeout_seconds": 30,
        "cold_start_timeout_seconds": 400,
    }
    values.update(overrides)
    return make_profile(**values)


MODAL_ENV = {"LLM_KEY": "bearer-value", "PROXY_ID": "id-value", "PROXY_SECRET": "secret-value"}


class ProfileConfigTests(unittest.TestCase):
    def test_new_fields_parse_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(
                """
default_model_profile = "modal"

[models.modal]
adapter = "modal"
server = "sglang"
model = "org/model"
endpoint = "https://ws--app-serve.modal.run/v1"
api_key_env = "LLM_KEY"
context_window = 32768
cold_start_timeout_seconds = 600
fallback = ["local"]

[models.modal.headers_env]
Modal-Key = "PROXY_ID"
Modal-Secret = "PROXY_SECRET"

[models.modal.pricing]
gpu_usd_per_second = 0.0005

[models.local]
adapter = "ollama"
model = "qwen2.5-coder:7b"
endpoint = "http://127.0.0.1:11434/v1"
api_mode = "native"
""",
                encoding="utf-8",
            )
            config = load_config(path, environ={})

        modal = config.models["modal"]
        self.assertEqual(modal.server, "sglang")
        self.assertEqual(modal.context_window, 32768)
        self.assertEqual(modal.cold_start_timeout_seconds, 600.0)
        self.assertEqual(modal.fallback, ("local",))
        self.assertEqual(dict(modal.headers_env), {"Modal-Key": "PROXY_ID", "Modal-Secret": "PROXY_SECRET"})
        assert modal.pricing is not None
        self.assertEqual(modal.pricing.gpu_usd_per_second, 0.0005)
        self.assertEqual(config.models["local"].api_mode, "native")

    def test_existing_profiles_keep_defaults(self) -> None:
        profile = make_profile()
        self.assertEqual(dict(profile.headers_env), {})
        self.assertIsNone(profile.context_window)
        self.assertEqual(profile.api_mode, "openai")
        self.assertEqual(profile.fallback, ())

    def test_environment_overrides_new_fields(self) -> None:
        config = load_config(
            environ={
                "AGENT_HARNESS_MODEL__M__ADAPTER": "modal",
                "AGENT_HARNESS_MODEL__M__MODEL": "org/model",
                "AGENT_HARNESS_MODEL__M__ENDPOINT": "https://x.modal.run/v1",
                "AGENT_HARNESS_MODEL__M__SERVER": "vllm",
                "AGENT_HARNESS_MODEL__M__CONTEXT_WINDOW": "4096",
                "AGENT_HARNESS_MODEL__M__COLD_START_TIMEOUT_SECONDS": "300",
                "AGENT_HARNESS_MODEL__M__HEADERS_ENV": '{"Modal-Key": "PROXY_ID"}',
            }
        )
        profile = config.models["m"]
        self.assertEqual(profile.context_window, 4096)
        self.assertEqual(profile.cold_start_timeout_seconds, 300.0)
        self.assertEqual(dict(profile.headers_env), {"Modal-Key": "PROXY_ID"})

    def test_rejects_invalid_values(self) -> None:
        cases = [
            {"server": "tgi"},
            {"api_mode": "grpc"},
            {"context_window": 0},
            {"cold_start_timeout_seconds": 0},
            {"headers_env": {"Authorization": "KEY"}},
            {"headers_env": {"Bad Header": "KEY"}},
            {"headers_env": {"X-Key": "not-an-env-name"}},
            {"fallback": ("same", "same")},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ConfigError):
                make_profile(**overrides)

    def test_fallback_must_reference_other_configured_profile(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown profile"):
            HarnessConfig(
                models={"a": make_profile(fallback=("missing",))}, default_model_profile="a"
            )
        with self.assertRaisesRegex(ConfigError, "itself"):
            HarnessConfig(models={"a": make_profile(fallback=("a",))}, default_model_profile="a")

    def test_unknown_pricing_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(
                """
[models.m]
adapter = "openai_compatible"
model = "m"
endpoint = "http://x/v1"
[models.m.pricing]
per_request_usd = 1
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "per_request_usd"):
                load_config(path, environ={})


class BaseClientHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_headers_env_resolved_at_request_time(self) -> None:
        transport = FakeTransport([text_completion()])
        client = OpenAICompatibleClient(
            make_profile(headers_env={"X-Tenant": "TENANT_ID"}),
            transport=transport,
            environ={"TENANT_ID": "tenant-7"},
        )
        await client.complete(ModelRequest(messages=()))
        self.assertEqual(transport.calls[0][2]["X-Tenant"], "tenant-7")

    async def test_missing_header_variable_fails_before_transport_without_values(self) -> None:
        transport = FakeTransport([])
        client = OpenAICompatibleClient(
            make_profile(headers_env={"X-Tenant": "TENANT_ID"}), transport=transport, environ={}
        )
        with self.assertRaisesRegex(ModelConfigurationError, "TENANT_ID.*X-Tenant"):
            await client.complete(ModelRequest(messages=()))
        self.assertEqual(transport.calls, [])

    async def test_request_without_timeout_uses_profile_timeout(self) -> None:
        transport = FakeTransport([text_completion()])
        client = OpenAICompatibleClient(
            make_profile(timeout_seconds=240), transport=transport, environ={}
        )
        await client.complete(ModelRequest(messages=()))
        self.assertEqual(transport.calls[0][3], 240)

    async def test_cold_start_timeout_applies_until_first_success(self) -> None:
        transport = FakeTransport([text_completion(), text_completion()])
        client = OpenAICompatibleClient(
            make_profile(timeout_seconds=30, cold_start_timeout_seconds=500),
            transport=transport,
            environ={},
        )
        await client.complete(ModelRequest(messages=()))
        await client.complete(ModelRequest(messages=()))
        self.assertEqual([call[3] for call in transport.calls], [500, 30])

    async def test_cached_tokens_reported_in_metadata(self) -> None:
        transport = FakeTransport(
            [
                json_response(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 96},
                        },
                    }
                )
            ]
        )
        client = OpenAICompatibleClient(make_profile(), transport=transport, environ={})
        response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.metadata["cached_input_tokens"], 96)

    def test_native_api_mode_rejected_for_openai_adapter(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "api_mode"):
            OpenAICompatibleClient(make_profile(api_mode="native"), environ={})


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_tool_call_ids_are_synthesized_uniquely(self) -> None:
        transport = FakeTransport(
            [
                json_response(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {"function": {"name": "a", "arguments": {"x": 1}}},
                                        {"function": {"name": "b", "arguments": "{}"}},
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                )
            ]
        )
        client = OllamaClient(
            make_profile(adapter="ollama", endpoint="http://127.0.0.1:11434/v1"),
            transport=transport,
            environ={},
        )
        response = await client.complete(ModelRequest(messages=()))
        ids = [call.id for call in response.tool_calls]
        self.assertEqual(len(set(ids)), 2)
        self.assertTrue(all(call_id.startswith("call_") for call_id in ids))
        self.assertEqual(response.tool_calls[0].arguments, {"x": 1})
        self.assertEqual(transport.calls[0][0], "http://127.0.0.1:11434/v1/chat/completions")

    async def test_native_mode_payload_and_normalization(self) -> None:
        transport = FakeTransport(
            [
                json_response(
                    {
                        "model": "qwen2.5-coder:7b",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"function": {"name": "weather", "arguments": {"city": "Pune"}}}
                            ],
                        },
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 42,
                        "eval_count": 7,
                    }
                )
            ]
        )
        client = OllamaClient(
            make_profile(
                adapter="ollama",
                model="qwen2.5-coder:7b",
                endpoint="http://127.0.0.1:11434/v1",
                api_mode="native",
                context_window=8192,
                parameters={"top_p": 0.9, "keep_alive": "10m"},
            ),
            transport=transport,
            environ={},
        )
        response = await client.complete(
            ModelRequest(
                messages=(
                    Message(role=Role.USER, content="weather?"),
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=(ToolCall("c0", "weather", {"city": "Delhi"}),),
                    ),
                    Message(role=Role.TOOL, content="sunny", name="weather", tool_call_id="c0"),
                ),
                tools=(ToolSpec(name="weather", description="w", input_schema={"type": "object"}),),
                temperature=0.1,
                max_output_tokens=64,
            )
        )

        url, payload, _, _ = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["keep_alive"], "10m")
        self.assertEqual(
            payload["options"],
            {"top_p": 0.9, "num_ctx": 8192, "temperature": 0.1, "num_predict": 64},
        )
        self.assertEqual(
            payload["messages"][1]["tool_calls"][0]["function"]["arguments"], {"city": "Delhi"}
        )
        self.assertEqual(payload["messages"][2]["tool_name"], "weather")
        self.assertEqual(payload["tools"][0]["function"]["name"], "weather")
        self.assertEqual(response.finish_reason, FinishReason.TOOL_CALLS)
        self.assertEqual(response.tool_calls[0].name, "weather")
        self.assertEqual(response.tool_calls[0].arguments, {"city": "Pune"})
        self.assertEqual(response.usage.input_tokens, 42)
        self.assertEqual(response.usage.output_tokens, 7)
        self.assertEqual(response.metadata["provider_model"], "qwen2.5-coder:7b")

    async def test_native_mode_length_finish_and_malformed_message(self) -> None:
        transport = FakeTransport(
            [
                json_response({"message": {"content": "partial"}, "done_reason": "length"}),
                json_response({"done": True}),
            ]
        )
        client = OllamaClient(
            make_profile(adapter="ollama", endpoint="http://h:11434", api_mode="native"),
            transport=transport,
            environ={},
        )
        response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.finish_reason, FinishReason.LENGTH)
        self.assertEqual(transport.calls[0][0], "http://h:11434/api/chat")
        with self.assertRaises(MalformedResponseError):
            await client.complete(ModelRequest(messages=()))

    def test_health_and_model_urls(self) -> None:
        client = OllamaClient(
            make_profile(adapter="ollama", endpoint="http://127.0.0.1:11434/v1"), environ={}
        )
        self.assertEqual(client.health_url(), "http://127.0.0.1:11434/api/version")
        self.assertEqual(client.models_url(), "http://127.0.0.1:11434/v1/models")


class LlamaCppClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_loading_503_is_retried_then_succeeds(self) -> None:
        transport = FakeTransport(
            [
                json_response({"error": {"message": "Loading model"}}, status=503),
                text_completion("ready"),
            ]
        )
        client = LlamaCppClient(
            make_profile(adapter="llama_cpp", endpoint="http://127.0.0.1:8080/v1", max_retries=1),
            transport=transport,
            environ={},
        )
        response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.content, "ready")
        self.assertEqual(len(transport.calls), 2)

    async def test_cached_tokens_from_timings_fallback(self) -> None:
        transport = FakeTransport([text_completion(timings={"cache_n": 64, "prompt_n": 8})])
        client = LlamaCppClient(make_profile(adapter="llama_cpp"), transport=transport, environ={})
        response = await client.complete(ModelRequest(messages=()))
        self.assertEqual(response.metadata["cached_input_tokens"], 64)

    def test_health_url_is_server_root(self) -> None:
        client = LlamaCppClient(
            make_profile(adapter="llama_cpp", endpoint="http://127.0.0.1:8080/v1"), environ={}
        )
        self.assertEqual(client.health_url(), "http://127.0.0.1:8080/health")


class ModalClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_bearer_and_proxy_headers(self) -> None:
        transport = FakeTransport([text_completion()])
        client = ModalClient(modal_profile(), transport=transport, environ=MODAL_ENV)
        await client.complete(ModelRequest(messages=()))
        url, _, headers, _ = transport.calls[0]
        self.assertEqual(url, "https://ws--app-serve.modal.run/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer bearer-value")
        self.assertEqual(headers["Modal-Key"], "id-value")
        self.assertEqual(headers["Modal-Secret"], "secret-value")

    def test_both_auth_layers_are_required_by_configuration(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "api_key_env"):
            ModalClient(modal_profile(api_key_env=None), environ=MODAL_ENV)
        with self.assertRaisesRegex(ModelConfigurationError, "Modal-Secret"):
            ModalClient(modal_profile(headers_env={"Modal-Key": "PROXY_ID"}), environ=MODAL_ENV)

    async def test_missing_proxy_secret_fails_before_transport_without_leaking(self) -> None:
        transport = FakeTransport([])
        environ = {"LLM_KEY": "bearer-value", "PROXY_ID": "id-value"}
        client = ModalClient(modal_profile(), transport=transport, environ=environ)
        with self.assertRaises(ModelConfigurationError) as caught:
            await client.complete(ModelRequest(messages=()))
        message = str(caught.exception)
        self.assertIn("PROXY_SECRET", message)
        self.assertNotIn("bearer-value", message)
        self.assertNotIn("id-value", message)
        self.assertEqual(transport.calls, [])

    async def test_cold_start_gateway_timeout_retried_with_cold_timeout(self) -> None:
        transport = FakeTransport(
            [
                json_response({"error": "upstream"}, status=504),
                HttpTransportTimeout("slow start"),
                text_completion(),
                text_completion(),
            ]
        )
        client = ModalClient(
            modal_profile(max_retries=2), transport=transport, environ=MODAL_ENV
        )
        await client.complete(ModelRequest(messages=()))
        await client.complete(ModelRequest(messages=()))
        self.assertEqual([call[3] for call in transport.calls], [400, 400, 400, 30])

    async def test_default_cold_start_timeout_and_server(self) -> None:
        transport = FakeTransport([text_completion()])
        client = ModalClient(
            modal_profile(cold_start_timeout_seconds=None), transport=transport, environ=MODAL_ENV
        )
        await client.complete(ModelRequest(messages=()))
        self.assertEqual(transport.calls[0][3], 600.0)
        self.assertEqual(client.server, "vllm")

    async def test_proxy_rejection_is_authentication_error(self) -> None:
        transport = FakeTransport([json_response({"error": "missing credentials"}, status=401)])
        client = ModalClient(modal_profile(max_retries=3), transport=transport, environ=MODAL_ENV)
        with self.assertRaises(AuthenticationError):
            await client.complete(ModelRequest(messages=()))
        self.assertEqual(len(transport.calls), 1)


class SingleSystemMessageTests(unittest.IsolatedAsyncioTestCase):
    """Phase 20: Qwen 3.x templates reject a second system message, so clients send one."""

    MESSAGES = (
        Message(Role.SYSTEM, "## Instructions from AGENTS.md"),
        Message(Role.SYSTEM, "You are a careful coding agent."),
        Message(Role.USER, "what is the weather in pittsburgh"),
    )
    MERGED = "## Instructions from AGENTS.md\n\nYou are a careful coding agent."

    def test_helper(self) -> None:
        from agent_harness.model_gateway.system_messages import single_system_message

        merged = single_system_message(ModelRequest(messages=self.MESSAGES))
        self.assertEqual([(m.role, m.content) for m in merged.messages],
                         [(Role.SYSTEM, self.MERGED), (Role.USER, "what is the weather in pittsburgh")])
        one = ModelRequest(messages=self.MESSAGES[1:])
        self.assertIs(single_system_message(one), one)  # nothing to merge: unchanged
        later = ModelRequest(messages=(*self.MESSAGES, Message(Role.SYSTEM, "late note")))
        self.assertEqual([m.role for m in single_system_message(later).messages],
                         [Role.SYSTEM, Role.USER, Role.SYSTEM])  # only the leading ones merge

    async def test_every_adapter_sends_one_leading_system_message(self) -> None:
        clients = {
            "openai_compatible": lambda t: OpenAICompatibleClient(make_profile(), transport=t, environ={}),
            "llama_cpp": lambda t: LlamaCppClient(make_profile(adapter="llama_cpp"), transport=t, environ={}),
            "ollama": lambda t: OllamaClient(make_profile(adapter="ollama"), transport=t, environ={}),
            "modal": lambda t: ModalClient(modal_profile(), transport=t, environ=MODAL_ENV),
        }
        for name, build in clients.items():
            with self.subTest(name):
                transport = FakeTransport([text_completion("ok")])
                await build(transport).complete(ModelRequest(messages=self.MESSAGES))
                sent = transport.calls[0][1]["messages"]  # type: ignore[index]
                self.assertEqual([m["role"] for m in sent], ["system", "user"])
                self.assertEqual(sent[0]["content"], self.MERGED)

    async def test_ollama_native_mode_and_llama_cpp_token_counting(self) -> None:
        transport = FakeTransport([json_response({"message": {"role": "assistant", "content": "ok"},
                                                  "done": True, "done_reason": "stop"})])
        native = OllamaClient(make_profile(adapter="ollama", api_mode="native"), transport=transport, environ={})
        await native.complete(ModelRequest(messages=self.MESSAGES))
        self.assertEqual([m["role"] for m in transport.calls[0][1]["messages"]], ["system", "user"])  # type: ignore[index]
        transport = FakeTransport([json_response({"prompt": "rendered"}), json_response({"tokens": [1, 2, 3]})])
        client = LlamaCppClient(make_profile(adapter="llama_cpp", token_counting="server"),
                                transport=transport, environ={})
        self.assertEqual(await client.count_input_tokens(ModelRequest(messages=self.MESSAGES)), 3)
        rendered = transport.calls[0][1]["messages"]  # type: ignore[index]
        self.assertEqual([m["role"] for m in rendered], ["system", "user"])


class ExtraBodyTests(unittest.IsolatedAsyncioTestCase):
    """A request's extra_body fields reach the payload; adapter-owned fields are kept."""

    async def test_every_adapter_merges_extra_body(self) -> None:
        extra = {"chat_template_kwargs": {"enable_thinking": False}, "model": "hijack", "messages": []}
        request = ModelRequest(messages=(Message(Role.USER, "summarize"),), extra_body=extra)
        native_reply = json_response({"message": {"role": "assistant", "content": "ok"}, "done": True})
        clients = {
            "openai_compatible": (lambda t: OpenAICompatibleClient(make_profile(), transport=t, environ={}),
                                  text_completion("ok")),
            "llama_cpp": (lambda t: LlamaCppClient(make_profile(adapter="llama_cpp"), transport=t, environ={}),
                          text_completion("ok")),
            "ollama native": (lambda t: OllamaClient(make_profile(adapter="ollama", api_mode="native"),
                                                     transport=t, environ={}), native_reply),
        }
        for name, (build, reply) in clients.items():
            with self.subTest(name):
                transport = FakeTransport([reply])
                await build(transport).complete(request)
                payload = transport.calls[0][1]
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})  # type: ignore[index]
                self.assertEqual(payload["model"], "test-model")  # type: ignore[index]
                self.assertEqual(len(payload["messages"]), 1)  # type: ignore[index,arg-type]
        transport = FakeTransport([text_completion("ok")])
        await OpenAICompatibleClient(make_profile(), transport=transport, environ={}).complete(
            ModelRequest(messages=(Message(Role.USER, "hi"),)))
        self.assertNotIn("chat_template_kwargs", transport.calls[0][1])  # unchanged without extra_body


class GatewayRegistrationTests(unittest.TestCase):
    def test_builtin_adapters_resolve_to_provider_clients(self) -> None:
        config = HarnessConfig(
            models={
                "o": make_profile(adapter="ollama"),
                "l": make_profile(adapter="llama_cpp"),
                "m": modal_profile(),
            },
            default_model_profile="o",
        )
        gateway = ModelGateway(config, transport=FakeTransport(), environ=MODAL_ENV)
        self.assertIsInstance(gateway.resolve("o"), OllamaClient)
        self.assertIsInstance(gateway.resolve("l"), LlamaCppClient)
        self.assertIsInstance(gateway.resolve("m"), ModalClient)


class DiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def _modal(self, transport: FakeTransport, server: str = "vllm") -> ModalClient:
        return ModalClient(modal_profile(server=server), transport=transport, environ=MODAL_ENV)

    async def test_all_checks_pass_for_healthy_modal_server(self) -> None:
        cached = json_response(
            {
                "choices": [{"message": {"content": "O"}}],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 896},
                },
            }
        )
        models = json_response({"data": [{"id": "org/model"}]})
        transport = FakeTransport(
            posts=[text_completion(), tool_completion(), text_completion(), cached],
            gets={"https://ws--app-serve.modal.run/v1/models": models},
        )
        checks = await diagnose(self._modal(transport), transport=transport)

        self.assertEqual(
            [(c.name, c.status) for c in checks],
            [
                ("credentials", CheckStatus.OK),
                ("health", CheckStatus.OK),
                ("model", CheckStatus.OK),
                ("completion", CheckStatus.OK),
                ("tool_call", CheckStatus.OK),
                ("prefix_cache", CheckStatus.OK),
            ],
        )
        self.assertIn("896 of 900", checks[-1].detail)
        _, headers, timeout = transport.get_calls[0]
        self.assertEqual(headers["Modal-Key"], "id-value")
        self.assertEqual(timeout, 400)

    async def test_missing_credentials_stop_before_network(self) -> None:
        transport = FakeTransport()
        client = ModalClient(modal_profile(), transport=transport, environ={})
        checks = await diagnose(client, transport=transport)
        self.assertEqual([(c.name, c.status) for c in checks], [("credentials", CheckStatus.FAIL)])
        self.assertEqual(transport.calls, [])
        self.assertEqual(transport.get_calls, [])

    async def test_unreachable_server_fails_health_and_stops(self) -> None:
        transport = FakeTransport(
            gets={"http://127.0.0.1:8080/health": HttpTransportError("connection refused")}
        )
        client = LlamaCppClient(
            make_profile(adapter="llama_cpp", endpoint="http://127.0.0.1:8080/v1"),
            transport=transport,
            environ={},
        )
        checks = await diagnose(client, transport=transport)
        self.assertEqual(checks[-1].name, "health")
        self.assertEqual(checks[-1].status, CheckStatus.FAIL)
        self.assertEqual(transport.calls, [])

    async def test_text_tool_call_and_missing_cache_report_give_hints(self) -> None:
        models = json_response({"data": [{"id": "org/model"}]})
        transport = FakeTransport(
            posts=[
                text_completion(),
                text_completion('{"name": "report_status", "arguments": {"status": "ok"}}'),
                text_completion(),
                text_completion(),
            ],
            gets={"https://ws--app-serve.modal.run/v1/models": models},
        )
        checks = {c.name: c for c in await diagnose(self._modal(transport, "sglang"), transport=transport)}
        self.assertEqual(checks["tool_call"].status, CheckStatus.WARN)
        self.assertIn("--tool-call-parser", checks["tool_call"].detail)
        self.assertEqual(checks["prefix_cache"].status, CheckStatus.WARN)
        self.assertIn("--enable-cache-report", checks["prefix_cache"].detail)

    async def test_unserved_model_fails_model_check(self) -> None:
        transport = FakeTransport(
            posts=[text_completion()],
            gets={
                "http://127.0.0.1:11434/api/version": json_response({"version": "0.9"}),
                "http://127.0.0.1:11434/v1/models": json_response({"data": [{"id": "llama3.1:8b"}]}),
            },
        )
        client = OllamaClient(
            make_profile(adapter="ollama", model="qwen2.5-coder:7b", endpoint="http://127.0.0.1:11434/v1"),
            transport=transport,
            environ={},
        )
        checks = {c.name: c for c in await diagnose(client, transport=transport, check_tools=False)}
        self.assertEqual(checks["model"].status, CheckStatus.FAIL)
        self.assertIn("llama3.1:8b", checks["model"].detail)
        self.assertEqual(checks["prefix_cache"].status, CheckStatus.SKIP)


class DoctorCommandTests(unittest.TestCase):
    def test_doctor_reports_checks_and_exit_code(self) -> None:
        transport = FakeTransport(
            posts=[text_completion()],
            gets={"http://127.0.0.1:8080/health": json_response({"status": "ok"}),
                  # A single-model llama-server lists only the model it was started with.
                  "http://127.0.0.1:8080/v1/models": json_response({"data": [{"id": "model.gguf"}]})},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(
                """
[models.llamacpp]
adapter = "llama_cpp"
model = "local-gguf"
endpoint = "http://127.0.0.1:8080/v1"
max_retries = 0
""",
                encoding="utf-8",
            )
            output = io.StringIO()
            with mock.patch.object(cli, "StdlibHttpTransport", return_value=transport), redirect_stdout(output):
                code = cli.main(["doctor", "--config", str(path), "--skip-tools", "--skip-cache"])

        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("profile=llamacpp adapter=llama_cpp", text)
        self.assertIn("[  OK] health", text)
        self.assertIn("[SKIP] model", text)
        self.assertIn("[  OK] completion", text)


if __name__ == "__main__":
    unittest.main()
