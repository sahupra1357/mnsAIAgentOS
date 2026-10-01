"""Network-free tests for Phase 12d: gateway and operations.

Per-profile concurrency, Retry-After and jitter, structured-output modes, image
inputs, doctor improvements, benchmark resume retrying errors, tenant-scoped
artifacts, artifact garbage collection, and the eval runner's new keys.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tempfile
import time
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from email.utils import format_datetime
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness import cli
from agent_harness.config import ConfigError, ModelProfile, load_config
from agent_harness.domain import (
    ContentPart,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
)
from agent_harness.model_gateway import HttpResponse, OllamaClient, OpenAICompatibleClient
from agent_harness.model_gateway.diagnostics import CheckStatus, diagnose
from agent_harness.model_gateway.errors import ModelConfigurationError, ThrottlingError
from agent_harness.model_gateway.openai_compatible import parse_retry_after
from agent_harness.orchestrator import validate_image_url
from agent_harness.versioning import profile_sha256

PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()


def profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {"adapter": "openai_compatible", "model": "m", "endpoint": "http://x/v1",
                              "max_retries": 0}
    values.update(overrides)
    return ModelProfile(**values)


def ok_body(content: str = "ok", finish: str = "stop") -> HttpResponse:
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    return HttpResponse(200, {}, json.dumps(body).encode())


class Transport:
    def __init__(self, *responses: HttpResponse, delay: float = 0.0) -> None:
        self.responses = deque(responses)
        self.payloads: list[dict] = []
        self.delay = delay
        self.active = 0
        self.peak = 0

    async def post_json(self, url: str, payload: dict, headers: dict, timeout_seconds: float) -> HttpResponse:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            self.payloads.append(json.loads(json.dumps(payload)))
            await asyncio.sleep(self.delay)
            return self.responses.popleft()
        finally:
            self.active -= 1


def complete(client: OpenAICompatibleClient, request: ModelRequest | None = None) -> ModelResponse:
    return asyncio.run(client.complete(request or ModelRequest(messages=(Message(Role.USER, "hi"),))))


class ConcurrencyAndRetryTests(unittest.TestCase):
    def test_max_concurrency_caps_parallel_requests(self) -> None:
        transport = Transport(*(ok_body() for _ in range(6)), delay=0.02)
        client = OpenAICompatibleClient(profile(max_concurrency=2), transport=transport)

        async def burst() -> None:
            await asyncio.gather(*(client.complete(ModelRequest(messages=(Message(Role.USER, "x"),)))
                                   for _ in range(6)))

        asyncio.run(burst())
        self.assertEqual(transport.peak, 2)
        unlimited = Transport(*(ok_body() for _ in range(6)), delay=0.02)
        client = OpenAICompatibleClient(profile(), transport=unlimited)
        asyncio.run(burst())
        self.assertEqual(unlimited.peak, 6)

    def test_retry_after_is_honoured_and_capped(self) -> None:
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        throttled = HttpResponse(429, {"Retry-After": "7"}, b"slow down")
        transport = Transport(throttled, ok_body())
        client = OpenAICompatibleClient(profile(max_retries=1, retry_backoff_seconds=0.25),
                                        transport=transport, sleep=sleep)
        self.assertEqual(complete(client).content, "ok")
        self.assertEqual(sleeps, [7.0])
        capped = OpenAICompatibleClient(profile(max_retry_after_seconds=2), transport=Transport())
        error = ThrottlingError("x")
        error.retry_after = 600
        self.assertEqual(capped.retry_delay(0, error), 2.0)

    def test_jitter_spreads_delays(self) -> None:
        client = OpenAICompatibleClient(profile(retry_backoff_seconds=1.0, retry_jitter=0.5),
                                        transport=Transport(), jitter_random=lambda: 1.0)
        self.assertEqual(client.retry_delay(1), 3.0)  # 2s * (1 + 0.5)
        low = OpenAICompatibleClient(profile(retry_backoff_seconds=1.0, retry_jitter=0.5),
                                     transport=Transport(), jitter_random=lambda: 0.0)
        self.assertEqual(low.retry_delay(1), 1.0)
        self.assertEqual(OpenAICompatibleClient(profile(retry_backoff_seconds=1.0), transport=Transport())
                         .retry_delay(2), 4.0)  # default: no jitter

    def test_parse_retry_after(self) -> None:
        self.assertEqual(parse_retry_after("12"), 12.0)
        future = datetime.now(UTC) + timedelta(seconds=30)
        self.assertAlmostEqual(parse_retry_after(format_datetime(future, usegmt=True)), 30, delta=2)
        for bad in (None, "", "soon", "-5"):
            self.assertIsNone(parse_retry_after(bad))

    def test_profile_fields_validate_and_keep_old_hashes(self) -> None:
        self.assertEqual(profile_sha256(profile()), profile_sha256(profile(structured_output="strict")))
        self.assertNotEqual(profile_sha256(profile()), profile_sha256(profile(max_concurrency=2)))
        for bad in ({"max_concurrency": 0}, {"retry_jitter": 2}, {"structured_output": "loose"},
                    {"max_retry_after_seconds": -1}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                profile(**bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                            'max_concurrency = 3\nretry_jitter = 0.2\nstructured_output = "prompt"\n')
            loaded = load_config(path, environ={"AGENT_HARNESS_MODEL__P__MAX_RETRY_AFTER_SECONDS": "5"}).models["p"]
            self.assertEqual((loaded.max_concurrency, loaded.retry_jitter, loaded.structured_output,
                              loaded.max_retry_after_seconds), (3, 0.2, "prompt", 5.0))


SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


class StructuredOutputModeTests(unittest.TestCase):
    def payload(self, client_type: type = OpenAICompatibleClient, **overrides: Any) -> dict:
        transport = Transport(ok_body("{}") if client_type is OpenAICompatibleClient else HttpResponse(
            200, {}, json.dumps({"message": {"content": "{}"}, "done_reason": "stop"}).encode()))
        client = client_type(profile(**overrides), transport=transport)
        complete(client, ModelRequest(messages=(Message(Role.USER, "q"),), response_schema=SCHEMA))
        return transport.payloads[0]

    def test_modes(self) -> None:
        self.assertTrue(self.payload()["response_format"]["json_schema"]["strict"])
        self.assertFalse(self.payload(structured_output="non_strict")["response_format"]["json_schema"]["strict"])
        prompt = self.payload(structured_output="prompt")
        self.assertNotIn("response_format", prompt)
        self.assertIn('"properties"', prompt["messages"][-1]["content"])
        native = self.payload(OllamaClient, adapter="ollama", api_mode="native", structured_output="prompt")
        self.assertNotIn("format", native)
        self.assertIn("JSON Schema", native["messages"][-1]["content"])


class ImageTests(unittest.TestCase):
    def test_openai_content_array(self) -> None:
        transport = Transport(ok_body("a cat"))
        client = OpenAICompatibleClient(profile(), transport=transport)
        message = Message(Role.USER, "What is this?", parts=(ContentPart(image_url=PNG),))
        complete(client, ModelRequest(messages=(message,)))
        self.assertEqual(transport.payloads[0]["messages"][0]["content"], [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": PNG}},
        ])
        complete(OpenAICompatibleClient(profile(), transport=(plain := Transport(ok_body()))))
        self.assertEqual(plain.payloads[0]["messages"][0]["content"], "hi")  # text-only unchanged

    def test_ollama_native_needs_data_urls(self) -> None:
        reply = HttpResponse(200, {}, json.dumps({"message": {"content": "cat"}, "done_reason": "stop"}).encode())
        transport = Transport(reply)
        client = OllamaClient(profile(adapter="ollama", api_mode="native"), transport=transport)
        complete(client, ModelRequest(messages=(Message(Role.USER, "?", parts=(ContentPart(image_url=PNG),)),)))
        self.assertEqual(transport.payloads[0]["messages"][0]["images"], [PNG.split(",", 1)[1]])
        with self.assertRaises(ModelConfigurationError):
            complete(OllamaClient(profile(adapter="ollama", api_mode="native"), transport=Transport(reply)),
                     ModelRequest(messages=(Message(Role.USER, "?", parts=(
                         ContentPart(image_url="https://example.com/cat.png"),)),)))

    def test_validation(self) -> None:
        validate_image_url(PNG)
        validate_image_url("https://example.com/cat.png")
        for bad in ("http://example.com/cat.png", "data:text/html;base64,PGI+", "file:///etc/passwd", "nope"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_image_url(bad)

    def test_images_persist_and_count_in_context(self) -> None:
        from agent_harness.context import IMAGE_TOKENS, BudgetedContextBuilder, ContextLimits
        from agent_harness.persistence import SQLiteRunStore
        from agent_harness.domain import RunRecord, RunEvent, EventType

        async def body() -> None:
            with tempfile.TemporaryDirectory() as directory:
                async with SQLiteRunStore(Path(directory, "db.sqlite3")) as store:
                    run = RunRecord(id="r", model_profile="p",
                                    messages=[Message(Role.USER, "x", parts=(ContentPart(image_url=PNG),))])
                    await store.create_run(run, RunEvent(run_id="r", type=EventType.RUN_CREATED))
                    loaded = await store.get_run("r")
                    assert loaded is not None
                    self.assertEqual(loaded.messages[0].parts, (ContentPart(image_url=PNG),))

        asyncio.run(body())
        builder = BudgetedContextBuilder(lambda name: ContextLimits(context_window=100_000))
        with_image = builder._message_tokens(Message(Role.USER, "x", parts=(ContentPart(image_url=PNG),)))
        self.assertEqual(with_image - builder._message_tokens(Message(Role.USER, "x")), IMAGE_TOKENS)

    def test_cli_image_flag_builds_a_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "cat.png")
            image.write_bytes(b"\x89PNG fake")
            self.assertEqual(cli._image_url(str(image)), PNG)
            self.assertEqual(cli._image_url("https://example.com/x.png"), "https://example.com/x.png")
            with self.assertRaises(ValueError):
                cli._image_url(str(Path(directory, "notes.txt")))


class DoctorTests(unittest.TestCase):
    class Client:
        def __init__(self, *responses: ModelResponse) -> None:
            self.responses = deque(responses)
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return self.responses.popleft()

    def test_max_tokens_and_out_of_tokens_message(self) -> None:
        client = self.Client(ModelResponse(content="", finish_reason=FinishReason.LENGTH),
                             ModelResponse(content="", finish_reason=FinishReason.LENGTH))
        checks = asyncio.run(diagnose(client, check_cache=False, max_tokens=64))  # type: ignore[arg-type]
        self.assertEqual([r.max_output_tokens for r in client.requests], [64, 64])
        self.assertEqual([c.status for c in checks], [CheckStatus.WARN, CheckStatus.WARN])
        self.assertIn("ran out of tokens (64)", checks[1].detail)

    def test_strict_turns_warnings_into_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')

            class Fake:
                async def post_json(self, url, payload, headers, timeout):
                    return ok_body("I would call report_status now")  # tool call as plain text

                async def get(self, url, headers, timeout):
                    return HttpResponse(200, {}, json.dumps({"data": [{"id": "m"}]}).encode())

            for flags, expected in (([], 0), (["--strict"], 1)):
                out = io.StringIO()
                with mock.patch.object(cli, "StdlibHttpTransport", return_value=Fake()), redirect_stdout(out):
                    code = cli.main(["doctor", "--config", str(path), "--skip-cache", *flags])
                with self.subTest(flags=flags):
                    self.assertEqual(code, expected)
                    self.assertIn("[WARN] tool_call", out.getvalue())

    def test_router_model_check(self) -> None:
        from agent_harness.model_gateway import LlamaCppClient

        async def check(models: list[str]) -> Any:
            class Get:
                async def get(self, url, headers, timeout):
                    if url.endswith("/health"):
                        return HttpResponse(200, {}, b"{}")
                    return HttpResponse(200, {}, json.dumps({"data": [{"id": m} for m in models]}).encode())

            client = LlamaCppClient(profile(adapter="llama_cpp", model="qwen-27b", endpoint="http://r/v1"),
                                    transport=Transport(ok_body()))
            checks = await diagnose(client, transport=Get(), check_tools=False, check_cache=False)  # type: ignore[arg-type]
            return next(c for c in checks if c.name == "model")

        self.assertEqual(asyncio.run(check(["qwen-27b", "other"])).status, CheckStatus.OK)
        self.assertEqual(asyncio.run(check(["a", "b"])).status, CheckStatus.FAIL)  # router without it
        self.assertEqual(asyncio.run(check(["model.gguf"])).status, CheckStatus.SKIP)  # single-model server


class BenchResumeTests(unittest.TestCase):
    def test_errored_items_are_retried_on_resume(self) -> None:
        from agent_harness.bench.core import BenchItem, ItemResult
        from agent_harness.bench.runner import run_benchmark

        calls: list[str] = []

        class Suite:
            async def run(self, item: BenchItem, context: Any) -> ItemResult:
                calls.append(item.id)
                return ItemResult(item.id, "default", True, status="completed")

        items = [BenchItem("a", {}, None), BenchItem("b", {}, None), BenchItem("c", {}, None)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "results.jsonl")
            path.write_text("\n".join(json.dumps(r.as_dict()) for r in (
                ItemResult("a", "default", True, status="completed"),
                ItemResult("b", "default", False, status="error", error="OOM"),
                ItemResult("c", "default", False, status="failed"),
            )) + "\n")
            results = asyncio.run(run_benchmark(Suite(), items, None, results_path=path))  # type: ignore[arg-type]
            self.assertEqual(sorted(calls), ["b", "c"])
            self.assertTrue(all(r.correct for r in results))  # the newest result wins
            calls.clear()
            asyncio.run(run_benchmark(Suite(), items, None, results_path=path, retry_errors=False))  # type: ignore[arg-type]
            self.assertEqual(calls, [])


class ArtifactAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_artifacts_are_scoped_to_runs_the_principal_owns(self) -> None:
        from agent_harness.access import ApiKey, PrincipalDirectory, sha256_hex
        from agent_harness.access import Role as AccessRole
        from agent_harness.api import HarnessAPI
        from agent_harness.artifacts import InMemoryArtifactStore
        from agent_harness.config import HarnessConfig
        from agent_harness.domain import EventType, RunEvent
        from agent_harness.orchestrator import AgentOrchestrator
        from agent_harness.persistence import InMemoryRunStore
        from agent_harness.tools import ToolRegistry, ToolRuntime

        store = InMemoryRunStore()
        artifacts = InMemoryArtifactStore()
        ref = await artifacts.put(b"secret tool output", "text/plain")
        orch = AgentOrchestrator(models=None, tools=ToolRuntime(ToolRegistry()), store=store)  # type: ignore[arg-type]
        run_a = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "x")], tenant="team-a")
        run_b = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "x")], tenant="team-b")
        await store.save_checkpoint(run_a, [RunEvent(run_id=run_a.id, type=EventType.TOOL_RESPONDED,
                                                     payload={"artifact_id": ref.id})], expected_version=0)
        keys = [ApiKey("a", "team-a", AccessRole.VIEWER, sha256_hex("key-a-0123456789")),
                ApiKey("b", "team-b", AccessRole.VIEWER, sha256_hex("key-b-0123456789")),
                ApiKey("root", "*", AccessRole.ADMIN, sha256_hex("key-root-0123456789"))]
        profile_ = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        api = HarnessAPI(orchestrator=orch, store=store, queue=store, auth_token=None, artifacts=artifacts,
                         config=HarnessConfig(models={"p": profile_}, default_model_profile="p"),
                         principals=PrincipalDirectory(keys))

        async def get(path: str, key: str) -> tuple[int, bytes]:
            messages = deque([{"type": "http.request", "body": b"", "more_body": False}])
            sent: list[dict] = []

            async def receive() -> dict:
                return messages.popleft()

            async def send(message: dict) -> None:
                sent.append(message)

            await api({"type": "http", "method": "GET", "path": path, "query_string": b"",
                       "headers": [(b"authorization", f"Bearer {key}".encode())]}, receive, send)
            return sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:])

        self.assertEqual(await get(f"/runs/{run_a.id}/artifacts/{ref.id}", "key-a-0123456789"),
                         (200, b"secret tool output"))
        self.assertEqual((await get(f"/artifacts/{ref.id}", "key-b-0123456789"))[0], 404)  # by hash: no
        self.assertEqual((await get(f"/runs/{run_a.id}/artifacts/{ref.id}", "key-b-0123456789"))[0], 404)
        self.assertEqual((await get(f"/runs/{run_b.id}/artifacts/{ref.id}", "key-b-0123456789"))[0], 404)
        self.assertEqual((await get(f"/artifacts/{ref.id}", "key-root-0123456789"))[0], 200)


class GcTests(unittest.TestCase):
    def test_gc_removes_old_unreferenced_artifacts_only(self) -> None:
        from agent_harness.artifacts import LocalArtifactStore
        from agent_harness.domain import EventType, RunEvent, RunRecord
        from agent_harness.persistence import SQLiteRunStore

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
                "AGENT_HARNESS_USER_SETTINGS": "/nonexistent", "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent"}):
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                              '[artifacts]\nenabled = true\ndirectory = "artifacts"\n')
            database = str(Path(directory, "db.sqlite3"))
            artifacts = LocalArtifactStore(Path(directory, "artifacts"))

            async def seed() -> tuple[str, str, str]:
                kept = await artifacts.put(b"kept", "text/plain")
                orphan = await artifacts.put(b"orphan", "text/plain")
                fresh = await artifacts.put(b"fresh orphan", "text/plain")
                async with SQLiteRunStore(database) as store:
                    run = RunRecord(id="r1", model_profile="p", messages=[Message(Role.USER, "x")])
                    await store.create_run(run, RunEvent(run_id="r1", type=EventType.RUN_CREATED,
                                                         payload={"artifact_id": kept.id}))
                return kept.id, orphan.id, fresh.id

            kept, orphan, fresh = asyncio.run(seed())
            old = time.time() - 40 * 86400
            for artifact_id in (kept, orphan):
                os.utime(artifacts._path(artifact_id), (old, old))
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["gc", "--older-than-days", "30", "--config", str(config),
                                           "--database", database]), 0)
                self.assertTrue(artifacts._path(orphan).exists())  # dry run
                self.assertEqual(cli.main(["gc", "--older-than-days", "30", "--yes", "--config", str(config),
                                           "--database", database]), 0)
            self.assertIn("would delete 1 artifacts", out.getvalue())
            self.assertFalse(artifacts._path(orphan).exists())
            self.assertTrue(artifacts._path(kept).exists())  # still referenced
            self.assertTrue(artifacts._path(fresh).exists())  # too new: may belong to a running step


class EvalRunnerTests(unittest.TestCase):
    def test_new_scenario_keys(self) -> None:
        from agent_harness.evals import ScenarioError, load_scenarios, run_scenario
        from agent_harness.tools import create_default_runtime

        examples = Path(__file__).resolve().parents[1] / "examples" / "evals"
        scenarios = {s.name: s for s in load_scenarios(examples)}
        for name in ("structured_output", "ask_user", "guardrail_refusal"):
            with self.subTest(name=name):
                result = asyncio.run(run_scenario(scenarios[name], tools=create_default_runtime()))
                self.assertEqual(result.outcome, "pass", result.failures)
        with tempfile.TemporaryDirectory() as directory:
            for bad in ('prompt = "x"\nanswers = [1]\n', 'prompt = "x"\noutput_schema = "obj"\n',
                        'prompt = "x"\n[guardrails]\nsurprise = 1\n'):
                path = Path(directory, "bad.toml")
                path.write_text(bad)
                with self.subTest(bad=bad), self.assertRaises(ScenarioError):
                    load_scenarios(path)
            wrong = Path(directory, "wrong.toml")
            wrong.write_text('prompt = "x"\n[output_schema]\ntype = "object"\n[[model]]\n'
                             'tool_calls = [{ name = "submit_result", arguments = { a = 1 } }]\n'
                             '[expect]\nresult = { a = 2 }\n')
            result = asyncio.run(run_scenario(load_scenarios(wrong)[0], tools=create_default_runtime()))
            self.assertEqual(result.outcome, "fail")
            self.assertIn("result {'a': 1} != expected {'a': 2}", result.failures[0])


if __name__ == "__main__":
    unittest.main()
