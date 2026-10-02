"""Network-free tests for Phase 13: context-overflow recovery, input deadlines,
API rate limits, per-user quotas, and the ``[orchestrator]`` / ``AGENT_HARNESS_SET__`` config."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from agent_harness.access import Quota
from agent_harness.api import HarnessAPI
from agent_harness.application import orchestrator_settings, user_quotas_from_config
from agent_harness.config import ConfigError, HarnessConfig, ModelProfile, load_config
from agent_harness.domain import (
    ContextOverflowError,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    QuotaExceededError,
    Role,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
    Usage,
    utc_now,
)
from agent_harness.interaction import ASK_USER_TOOL
from agent_harness.model_gateway import (
    HttpResponse,
    OpenAICompatibleClient,
    ProviderContextOverflowError,
    ProviderError,
)
from agent_harness.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.ratelimit import RateLimiter
from agent_harness.tools import ToolRegistry, ToolRuntime, create_default_runtime
from agent_harness.worker import Worker

PROFILE = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


class Scripted:
    def __init__(self, outcomes: Sequence[ModelResponse | Exception]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def history(turns: int = 20) -> list[Message]:
    """A long conversation: the first user message plus many earlier turns."""

    messages = [Message(Role.SYSTEM, "You are helpful."), Message(Role.USER, "the task")]
    for i in range(turns):
        messages.append(Message(Role.ASSISTANT, f"step {i} " + "x" * 400))
        messages.append(Message(Role.USER, f"more {i} " + "y" * 400))
    return messages


def overflow(limit: int | None = None, requested: int | None = None) -> ContextOverflowError:
    error = ContextOverflowError("model context exceeded (400): too long")
    error.limit_tokens = limit
    error.requested_tokens = requested
    return error


# --- 1. Context overflow -------------------------------------------------------------------


class OverflowDetectionTests(unittest.IsolatedAsyncioTestCase):
    async def _error(self, status: int, body: dict) -> Exception:
        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                return HttpResponse(status=status, headers={}, body=json.dumps(body).encode())

        client = OpenAICompatibleClient(PROFILE, transport=Transport(), environ={})
        try:
            await client.complete(ModelRequest(messages=(Message(Role.USER, "hi"),)))
        except Exception as exc:  # noqa: BLE001 - the error is what is being tested
            return exc
        raise AssertionError("expected an error")

    async def test_vllm_message_is_an_overflow_with_sizes(self) -> None:
        error = await self._error(400, {"error": {"message": (
            "This model's maximum context length is 32768 tokens. However, you requested "
            "40000 tokens (39000 in the messages, 1000 in the completion).")}})
        self.assertIsInstance(error, ProviderContextOverflowError)
        assert isinstance(error, ContextOverflowError)
        self.assertEqual((error.limit_tokens, error.requested_tokens), (32768, 40000))

    async def test_llama_cpp_body_is_an_overflow_with_sizes(self) -> None:
        error = await self._error(400, {"error": {
            "code": 400, "message": "the request exceeds the available context size, try increasing it",
            "type": "exceed_context_size_error", "n_prompt_tokens": 5000, "n_ctx": 4096}})
        assert isinstance(error, ContextOverflowError)
        self.assertEqual((error.limit_tokens, error.requested_tokens), (4096, 5000))

    async def test_other_client_errors_stay_provider_errors(self) -> None:
        error = await self._error(400, {"error": {"message": "unknown parameter: foo"}})
        self.assertIsInstance(error, ProviderError)
        self.assertNotIsInstance(error, ContextOverflowError)


class OverflowRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, outcomes: Sequence[Any], settings: OrchestratorSettings | None = None):
        model = Scripted(outcomes)
        store = InMemoryRunStore()
        orch = AgentOrchestrator(models=model, tools=create_default_runtime(), store=store, settings=settings)
        run = await orch.create_run(model_profile="p", messages=history())
        return await orch.run(run.id), model, store, orch

    async def test_retries_with_a_smaller_context_and_keeps_the_cap(self) -> None:
        finished, model, store, orch = await self._run([overflow(), text("done"), text("again")])

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        full, retried = model.requests
        self.assertLess(len(retried.messages), len(full.messages))
        self.assertEqual(retried.messages[0], full.messages[0])  # system and task are pinned
        self.assertTrue(retried.messages[1].content.startswith("the task"))
        self.assertIn("omitted", retried.messages[1].content)
        events = await store.list_events(finished.id)
        recovered = [e for e in events if e.payload.get("context_overflow")]
        self.assertEqual(len(recovered), 1)
        self.assertLess(recovered[0].payload["max_input_tokens"], recovered[0].payload["estimated_tokens"])

        await orch.continue_run(finished.id, "and once more")
        await orch.run(finished.id)
        self.assertLess(len(model.requests[2].messages), len(full.messages) + 2)  # the cap still applies

    async def test_stated_sizes_shrink_less(self) -> None:
        _, _, store_half, _ = await self._run([overflow(), text("done")])
        _, _, store_ratio, _ = await self._run([overflow(limit=95, requested=100), text("done")])

        async def cap(store: InMemoryRunStore) -> int:
            runs = list(store._runs)
            events = await store.list_events(runs[0])
            return next(e.payload["max_input_tokens"] for e in events if e.payload.get("context_overflow"))

        self.assertGreater(await cap(store_ratio), await cap(store_half))

    async def test_fails_after_the_configured_retries(self) -> None:
        finished, model, _, _ = await self._run(
            [overflow(), overflow(), text("unused")], OrchestratorSettings(context_overflow_retries=1)
        )
        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("context exceeded", finished.error or "")
        self.assertEqual(len(model.requests), 2)

    async def test_can_be_turned_off(self) -> None:
        finished, model, _, _ = await self._run(
            [overflow(), text("unused")], OrchestratorSettings(context_overflow_retries=0)
        )
        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertEqual(len(model.requests), 1)

    async def test_extra_patterns_mark_other_errors(self) -> None:
        custom = PermanentModelError("upstream said: payload too big for model")
        finished, _, _, _ = await self._run(
            [custom, text("done")], OrchestratorSettings(context_overflow_patterns=("payload too big",))
        )
        self.assertEqual(finished.status, RunStatus.COMPLETED)


# --- 2. Approval and question deadlines ----------------------------------------------------


def deploy_tools() -> ToolRuntime:
    registry = ToolRegistry()

    async def deploy(arguments: Any) -> dict:
        return {"ok": True}

    registry.register(ToolSpec(name="deploy", description="d", input_schema={"type": "object"},
                               risk=ToolRisk.HIGH_RISK), deploy)
    return ToolRuntime(registry)


class InputDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def _paused(self, outcomes: Sequence[ModelResponse], **kwargs: Any):
        now = [utc_now()]
        store = InMemoryRunStore()
        model = Scripted(outcomes)
        orch = AgentOrchestrator(models=model, tools=kwargs.pop("tools", deploy_tools()), store=store,
                                 clock=lambda: now[0], **kwargs)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "ship it")])
        paused = await orch.run(run.id)
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        return orch, store, model, paused, now

    def call_deploy(self) -> ModelResponse:
        return ModelResponse(tool_calls=(ToolCall("d1", "deploy", {}),), finish_reason=FinishReason.TOOL_CALLS)

    async def test_overdue_approval_is_denied_and_the_run_continues(self) -> None:
        orch, store, model, paused, now = await self._paused(
            [self.call_deploy(), text("not deployed")], approval_timeout_seconds=600)

        self.assertEqual(await orch.expire_pending_input(), [])  # not yet due
        now[0] += timedelta(seconds=601)
        self.assertEqual(await orch.expire_pending_input(), [paused.id])
        self.assertTrue(orch.expired_input_resumes)

        finished = await orch.run(paused.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertIn("denied", model.requests[1].messages[-1].content)
        resolved = next(e for e in await store.list_events(paused.id) if e.type is EventType.APPROVAL_RESOLVED)
        self.assertEqual((resolved.payload["approved"], resolved.payload["actor"]), (False, "timeout"))

    async def test_cancel_and_fail_actions(self) -> None:
        for action, status in (("cancel", RunStatus.CANCELLED), ("fail", RunStatus.FAILED)):
            with self.subTest(action=action):
                orch, store, _, paused, now = await self._paused(
                    [self.call_deploy()], approval_timeout_seconds=60, on_input_timeout=action)
                now[0] += timedelta(seconds=61)
                self.assertEqual(await orch.expire_pending_input(), [paused.id])
                self.assertFalse(orch.expired_input_resumes)
                self.assertEqual((await store.get_run(paused.id)).status, status)

    async def test_questions_have_their_own_deadline(self) -> None:
        ask = ModelResponse(tool_calls=(ToolCall("q1", ASK_USER_TOOL, {"question": "Which region?"}),),
                            finish_reason=FinishReason.TOOL_CALLS)
        orch, _, model, paused, now = await self._paused(
            [ask, text("assumed eu")], tools=create_default_runtime(), ask_user=True,
            approval_timeout_seconds=60, question_timeout_seconds=3600)
        now[0] += timedelta(seconds=120)
        self.assertEqual(await orch.expire_pending_input(), [])  # approvals' deadline does not apply
        now[0] += timedelta(seconds=3600)
        self.assertEqual(await orch.expire_pending_input(), [paused.id])
        await orch.run(paused.id)
        self.assertIn("declined to answer", model.requests[1].messages[-1].content)

    async def test_no_timeout_waits_forever(self) -> None:
        orch, _, _, _, now = await self._paused([self.call_deploy()])
        now[0] += timedelta(days=30)
        self.assertEqual(await orch.expire_pending_input(), [])

    async def test_worker_expires_and_enqueues(self) -> None:
        orch, store, _, paused, now = await self._paused(
            [self.call_deploy(), text("ok")], approval_timeout_seconds=1)
        now[0] += timedelta(seconds=2)
        worker = Worker(orch, store)
        self.assertEqual(await worker.expire_input(), [paused.id])
        finished = await worker.run_once()
        assert finished is not None
        self.assertEqual(finished.status, RunStatus.COMPLETED)

    def test_rejects_bad_settings(self) -> None:
        with self.assertRaises(ValueError):
            AgentOrchestrator(models=Scripted([]), tools=deploy_tools(), store=InMemoryRunStore(),
                              on_input_timeout="ignore")


# --- 3. API rate limits ----------------------------------------------------------------------


class RateLimiterTests(unittest.TestCase):
    def test_bucket_refills_at_the_configured_rate(self) -> None:
        now = [0.0]
        limiter = RateLimiter(60, burst=2, clock=lambda: now[0])
        self.assertIsNone(limiter.check("a"))
        self.assertIsNone(limiter.check("a"))
        wait = limiter.check("a")
        self.assertAlmostEqual(wait or 0, 1.0)
        self.assertIsNone(limiter.check("b"))  # keys are independent
        now[0] += 1.0
        self.assertIsNone(limiter.check("a"))

    def test_burst_defaults_to_the_per_minute_rate(self) -> None:
        limiter = RateLimiter(3, clock=lambda: 0.0)
        self.assertEqual([limiter.check("a") is None for _ in range(4)], [True, True, True, False])


class ApiRateLimitTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, api: HarnessAPI, path: str, token: str = "test-token-0123456789"):
        sent: list[dict] = []

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict) -> None:
            sent.append(message)

        await api({"type": "http", "method": "GET", "path": path, "query_string": b"",
                   "client": ("10.0.0.1", 5000),
                   "headers": [(b"authorization", f"Bearer {token}".encode())]}, receive, send)
        return sent[0]["status"], dict(sent[0]["headers"])

    def api(self) -> HarnessAPI:
        store = InMemoryRunStore()
        orch = AgentOrchestrator(models=Scripted([]), tools=create_default_runtime(), store=store)
        return HarnessAPI(orchestrator=orch, store=store, queue=store, auth_token="test-token-0123456789",
                          config=HarnessConfig(models={"p": PROFILE}, default_model_profile="p"),
                          rate_limiter=RateLimiter(60, burst=2, clock=lambda: 0.0))

    async def test_requests_beyond_the_burst_get_429_with_retry_after(self) -> None:
        api = self.api()
        self.assertEqual((await self.call(api, "/profiles"))[0], 200)
        self.assertEqual((await self.call(api, "/profiles"))[0], 200)
        status, headers = await self.call(api, "/profiles")
        self.assertEqual(status, 429)
        self.assertEqual(headers[b"retry-after"], b"1")
        self.assertEqual((await self.call(api, "/healthz"))[0], 200)  # health checks are exempt

    async def test_failed_authentication_is_limited_by_address(self) -> None:
        api = self.api()
        statuses = [(await self.call(api, "/profiles", token="wrong-token-0123456789"))[0] for _ in range(3)]
        self.assertEqual(statuses, [401, 401, 429])


# --- 4. Per-user quotas ----------------------------------------------------------------------


def user_quota(**limits: Any) -> Any:
    def lookup(end_user: str) -> Quota | None:
        return Quota(end_user, **limits)
    return lookup


class UserQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def _check(self, store: Any) -> None:
        orch = AgentOrchestrator(models=Scripted([]), tools=create_default_runtime(), store=store,
                                 user_quotas=user_quota(max_runs_per_day=2))
        create = orch.create_run
        message = [Message(Role.USER, "hi")]
        await create(model_profile="p", messages=message, end_user="alice")
        await create(model_profile="p", messages=message, end_user="alice")
        with self.assertRaises(QuotaExceededError) as raised:
            await create(model_profile="p", messages=message, end_user="alice")
        self.assertIn("user 'alice'", str(raised.exception))
        await create(model_profile="p", messages=message, end_user="bob")  # users are separate
        await create(model_profile="p", messages=message)  # runs without a user are not limited
        await create(model_profile="p", messages=message, end_user="alice", tenant="other")  # per tenant

    async def test_runs_per_day_in_memory(self) -> None:
        await self._check(InMemoryRunStore())

    async def test_runs_per_day_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            try:
                await self._check(store)
            finally:
                store.close_now()

    async def test_tokens_cap_the_budget_and_stop_at_the_step_boundary(self) -> None:
        store = InMemoryRunStore()
        model = Scripted([ModelResponse(
            content="", finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(ToolCall("c1", "echo", {"text": "x"}),),
            usage=Usage(input_tokens=600, output_tokens=0),
        )])
        orch = AgentOrchestrator(models=model, tools=create_default_runtime(), store=store,
                                 user_quotas=user_quota(max_tokens_per_day=500), quota_check_seconds=0)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "hi")], end_user="carol")
        self.assertEqual(run.budget.max_tokens, 500)
        finished = await orch.run(run.id)
        self.assertEqual(finished.status, RunStatus.BUDGET_EXHAUSTED)
        with self.assertRaises(QuotaExceededError):
            await orch.create_run(model_profile="p", messages=[Message(Role.USER, "hi")], end_user="carol")


# --- Configuration -----------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def _load(self, toml: str = "", **env: str) -> HarnessConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\n'
                            'endpoint = "http://x/v1"\n' + toml, encoding="utf-8")
            return load_config(path, environ=env)

    def test_defaults_match_the_built_in_settings(self) -> None:
        self.assertEqual(orchestrator_settings(self._load().orchestrator), OrchestratorSettings())
        config = self._load()
        self.assertIsNone(user_quotas_from_config(config))
        self.assertEqual(config.api_rate_limit.requests_per_minute, 0)
        self.assertEqual(config.interaction.approval_timeout_seconds, 0)

    def test_toml_sections(self) -> None:
        config = self._load(
            "[orchestrator]\nmax_tool_calls_per_turn = 0\nmax_refused_repeats = -1\n"
            "context_overflow_retries = 3\ncontext_overflow_patterns = [\"too big\"]\n"
            "[interaction]\napproval_timeout_seconds = 3600\non_timeout = \"cancel\"\n"
            "[api]\nrate_limit_per_minute = 120\nrate_limit_burst = 20\n"
            "[user_quotas]\nmax_runs_per_day = 10\n[user_quotas.users.vip]\nmax_runs_per_day = 100\n"
        )
        settings = orchestrator_settings(config.orchestrator)
        self.assertIsNone(settings.max_tool_calls_per_turn)
        self.assertIsNone(settings.max_refused_repeats)
        self.assertEqual(settings.context_overflow_patterns, ("too big",))
        self.assertEqual((config.interaction.approval_timeout_seconds, config.interaction.on_timeout),
                         (3600.0, "cancel"))
        self.assertEqual((config.api_rate_limit.requests_per_minute, config.api_rate_limit.burst), (120.0, 20))
        lookup = user_quotas_from_config(config)
        assert lookup is not None
        self.assertEqual(lookup("anyone").max_runs_per_day, 10)
        self.assertEqual(lookup("vip").max_runs_per_day, 100)

    def test_environment_overrides_any_table_value(self) -> None:
        config = self._load(
            "[orchestrator]\nmax_tool_calls_per_turn = 16\n",
            AGENT_HARNESS_SET__ORCHESTRATOR__MAX_TOOL_CALLS_PER_TURN="8",
            AGENT_HARNESS_SET__INTERACTION__ON_TIMEOUT="fail",
            AGENT_HARNESS_SET__API__RATE_LIMIT_PER_MINUTE="30",
            AGENT_HARNESS_SET__USER_QUOTAS__USERS__BOB__MAX_TOKENS_PER_DAY="5000",
        )
        self.assertEqual(config.orchestrator.max_tool_calls_per_turn, 8)
        self.assertEqual(config.interaction.on_timeout, "fail")
        self.assertEqual(config.api_rate_limit.requests_per_minute, 30.0)
        self.assertEqual(config.user_quotas.for_user("bob").max_tokens_per_day, 5000)

    def test_invalid_values_are_rejected(self) -> None:
        for toml in ("[orchestrator]\ncontext_overflow_shrink = 1.5\n",
                     "[orchestrator]\nmax_model_attempts = 0\n",
                     "[interaction]\non_timeout = \"ignore\"\n",
                     "[api]\nrate_limit_per_minute = -1\n",
                     "[user_quotas]\nmax_runs_per_day = -2\n",
                     "[orchestrator]\nunknown = 1\n"):
            with self.subTest(toml=toml), self.assertRaises(ConfigError):
                self._load(toml)
        with self.assertRaises(ConfigError):
            self._load(AGENT_HARNESS_SET__ORCHESTRATOR="1")

    def test_config_show_reports_environment_overrides(self) -> None:
        import io
        import os
        from contextlib import redirect_stdout
        from unittest import mock

        from agent_harness import cli

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\n'
                            'endpoint = "http://x/v1"\n', encoding="utf-8")
            env = {"AGENT_HARNESS_SET__API__RATE_LIMIT_PER_MINUTE": "120",
                   "AGENT_HARNESS_USER_SETTINGS": str(Path(directory) / "none.toml"),
                   "AGENT_HARNESS_ENTERPRISE_SETTINGS": str(Path(directory) / "none2.toml")}
            output = io.StringIO()
            with mock.patch.dict(os.environ, env), redirect_stdout(output):
                self.assertEqual(cli.main(["config", "show", "--config", str(path)]), 0)
        self.assertIn("api.rate_limit_per_minute = 120    # from env AGENT_HARNESS_SET__API__RATE_LIMIT_PER_MINUTE",
                      output.getvalue())


if __name__ == "__main__":
    unittest.main()
