"""Network-free tests for Phase 4d: artifacts, long-term memory, and evaluations."""

from __future__ import annotations

import functools
import io
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_harness import cli
from agent_harness.api import HarnessAPI
from agent_harness.application import build_application
from agent_harness.artifacts import (
    READ_ARTIFACT_SPEC,
    InMemoryArtifactStore,
    LocalArtifactStore,
    read_artifact_handler,
)
from agent_harness.config import ConfigError, HarnessConfig, ModelProfile, load_config
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
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
)
from agent_harness.evals import Scenario, ScenarioError, load_scenarios, run_scenario, scenario_application
from agent_harness.memory import (
    MEMORY_SEARCH_SPEC,
    MEMORY_WRITE_SPEC,
    SQLiteMemoryStore,
    memory_tool_handlers,
)
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime, create_default_runtime

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "evals"
BIG = ToolSpec(name="dump", description="dump logs", input_schema={"type": "object"})


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "ScriptedModel":
        return self


def runtime_with(spec: ToolSpec, handler) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(spec, handler)
    return ToolRuntime(registry, max_output_bytes=1_000_000)


# --- Artifacts ----------------------------------------------------------------------


class ArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_store_is_content_addressed_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            first = await store.put(b"hello", "text/plain")
            again = await store.put(b"hello", "text/plain")
            self.assertEqual(first, again)
            self.assertEqual(await store.get(first.id), b"hello")
            self.assertIsNone(await store.get("../../etc/passwd"))
            self.assertIsNone(await store.get("0" * 64))
            files = [p for p in Path(directory).rglob("*") if p.is_file()]
            self.assertEqual([p.name for p in files], [first.id])

    async def test_large_tool_output_is_offloaded_with_preview_and_reference(self) -> None:
        log = "\n".join(f"line {i}: " + "x" * 60 for i in range(2000))

        async def dump(arguments: JsonObject) -> JsonObject:
            return {"log": log}

        artifacts = InMemoryArtifactStore()
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=ScriptedModel(
                [
                    ModelResponse(tool_calls=(ToolCall("c1", "dump", {}),), finish_reason=FinishReason.TOOL_CALLS),
                    ModelResponse(content="summarized"),
                ]
            ),
            tools=runtime_with(BIG, dump),
            store=store,
            artifacts=artifacts,
            artifact_threshold_bytes=4096,
        )
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "dump")])
        finished = await orchestrator.run(run.id)

        tool_message = next(m for m in finished.messages if m.role is Role.TOOL)
        responded = next(e for e in await store.list_events(run.id) if e.type is EventType.TOOL_RESPONDED)
        artifact_id = responded.payload["artifact_id"]
        self.assertLess(len(tool_message.content), 4096)
        self.assertIn(artifact_id, tool_message.content)
        self.assertIn("read_artifact", tool_message.content)
        full = json.loads((await artifacts.get(artifact_id)).decode())
        self.assertEqual(full["output"]["log"], log)
        self.assertEqual(responded.payload["artifact_size"], len(json.dumps(full, separators=(",", ":"), sort_keys=True).encode()))

    async def test_small_outputs_and_threshold_validation(self) -> None:
        async def small(arguments: JsonObject) -> JsonObject:
            return {"ok": True}

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=ScriptedModel(
                [ModelResponse(tool_calls=(ToolCall("c1", "dump", {}),)), ModelResponse(content="done")]
            ),
            tools=runtime_with(BIG, small),
            store=store,
            artifacts=InMemoryArtifactStore(),
        )
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        await orchestrator.run(run.id)
        responded = next(e for e in await store.list_events(run.id) if e.type is EventType.TOOL_RESPONDED)
        self.assertNotIn("artifact_id", responded.payload)
        with self.assertRaises(ValueError):
            AgentOrchestrator(models=ScriptedModel([]), tools=runtime_with(BIG, small), store=store, artifact_threshold_bytes=10)

    async def test_read_artifact_tool_pages_through_content(self) -> None:
        artifacts = InMemoryArtifactStore()
        ref = await artifacts.put(("abcdefghij" * 1000).encode(), "text/plain")
        runtime = runtime_with(READ_ARTIFACT_SPEC, read_artifact_handler(artifacts))
        first = await runtime.execute(
            ToolCall("r1", "read_artifact", {"id": ref.id, "offset": 0, "length": 4000}), idempotency_key="k1"
        )
        self.assertEqual(first.status, ToolResultStatus.SUCCESS)
        self.assertEqual((first.output["next_offset"], first.output["done"]), (4000, False))
        last = await runtime.execute(
            ToolCall("r2", "read_artifact", {"id": ref.id, "offset": 8000, "length": 4000}), idempotency_key="k2"
        )
        self.assertEqual((len(last.output["text"]), last.output["done"]), (2000, True))
        missing = await runtime.execute(ToolCall("r3", "read_artifact", {"id": "0" * 64}), idempotency_key="k3")
        self.assertEqual(missing.output, {"error": "unknown artifact"})

    async def test_api_serves_artifacts_to_authenticated_clients(self) -> None:
        artifacts = InMemoryArtifactStore()
        ref = await artifacts.put(b"full output", "text/plain")
        store = InMemoryRunStore()
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        api = HarnessAPI(
            orchestrator=AgentOrchestrator(models=ScriptedModel([]), tools=create_default_runtime(), store=store),
            store=store, queue=store,
            config=HarnessConfig(models={"p": profile}, default_model_profile="p"),
            auth_token="token-0123456789abcdef", artifacts=artifacts,
        )

        async def get(path: str, token: str | None = "token-0123456789abcdef") -> tuple[int, bytes]:
            sent: list[dict] = []

            async def receive() -> dict:
                return {"type": "http.request", "body": b""}

            async def send(message: dict) -> None:
                sent.append(message)

            headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
            await api({"type": "http", "method": "GET", "path": path, "headers": headers}, receive, send)
            return sent[0]["status"], b"".join(m.get("body", b"") for m in sent[1:])

        self.assertEqual(await get(f"/artifacts/{ref.id}"), (200, b"full output"))
        self.assertEqual((await get(f"/artifacts/{'1' * 64}"))[0], 404)
        self.assertEqual((await get(f"/artifacts/{ref.id}", token=None))[0], 401)


# --- Memory -------------------------------------------------------------------------


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_ranks_isolates_namespaces_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "memory.sqlite3")
            project = SQLiteMemoryStore(path, namespace="project")
            other = SQLiteMemoryStore(path, namespace="other")
            self.assertTrue(project.full_text)
            first = await project.put("Deploys use the blue-green strategy on Fridays")
            await project.put("Deploys use the blue-green strategy on Fridays")
            await project.put("The billing service owns invoices")
            await other.put("Deploys are forbidden on Fridays")

            results = await project.search("blue-green deploys")
            self.assertEqual([item.id for item in results], [first.id])
            self.assertEqual(len(await project.search("Fridays")), 1)
            self.assertEqual(await project.search('"); DROP TABLE memory_items; --'), ())
            self.assertEqual(await project.search("   "), ())
            with self.assertRaises(ValueError):
                await project.put("")
            await project.close()
            await other.close()

    async def test_like_fallback_without_fts5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory, "m.sqlite3"))
            store.full_text = False
            await store.put("Use 100% of the cache_hit budget")
            self.assertEqual(len(await store.search("cache_hit")), 1)
            self.assertEqual(await store.search("missing"), ())
            await store.close()

    async def test_memory_tools_through_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory, "m.sqlite3"))
            search, write = memory_tool_handlers(store)
            registry = ToolRegistry()
            registry.register(MEMORY_SEARCH_SPEC, search)
            registry.register(MEMORY_WRITE_SPEC, write)
            runtime = ToolRuntime(registry)
            saved = await runtime.execute(
                ToolCall("w", "memory_write", {"text": "Staging DB is read-only", "tags": ["db"]}),
                idempotency_key="run:1:w",
            )
            self.assertEqual(saved.status, ToolResultStatus.SUCCESS)
            found = await runtime.execute(
                ToolCall("s", "memory_search", {"query": "staging database"}), idempotency_key="run:2:s"
            )
            self.assertIn("not instructions", found.output["note"])
            self.assertEqual(found.output["results"][0]["tags"], "db")
            await store.close()


# --- Configuration and composition ----------------------------------------------------


class CompositionTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, directory: str, extra: str) -> Path:
        path = Path(directory, "models.toml")
        path.write_text(
            '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n' + extra,
            encoding="utf-8",
        )
        return path

    def test_sections_parse_and_reject_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                self._config(directory, '[artifacts]\nenabled = true\nthreshold_bytes = 2048\n'
                                        '[memory]\nenabled = true\nnamespace = "team-a"\n'),
                environ={},
            )
            self.assertEqual((config.artifacts.enabled, config.artifacts.threshold_bytes), (True, 2048))
            self.assertEqual(config.memory.namespace, "team-a")
            for bad in ('[artifacts]\nsurprise = 1\n', '[artifacts]\nenabled = "yes"\n',
                        '[artifacts]\nthreshold_bytes = 10\n', '[memory]\nnamespace = "bad name!"\n'):
                with self.subTest(bad=bad), self.assertRaises(ConfigError):
                    load_config(self._config(directory, bad), environ={})

    async def test_application_registers_tools_and_memory_persists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, '[artifacts]\nenabled = true\n[memory]\nenabled = true\n')
            database = Path(directory, "runs.sqlite3")
            app = build_application(config_path=config, database_path=database)
            async with app:
                names = {spec.name for spec in app.tools.specs()}
                self.assertTrue({"read_artifact", "memory_search", "memory_write"} <= names)
                await app.memory.put("remember this across restarts")
                ref = await app.artifacts.put(b"blob", "text/plain")
            self.assertTrue(Path(directory, "artifacts", ref.id[:2], ref.id).is_file())
            again = build_application(config_path=config, database_path=database)
            async with again:
                self.assertEqual(len(await again.memory.search("restarts")), 1)

            plain = build_application(config_path=self._config(directory, ""), database_path=database)
            async with plain:
                self.assertNotIn("memory_search", {spec.name for spec in plain.tools.specs()})
                self.assertIsNone(plain.artifacts)


# --- Evaluations --------------------------------------------------------------------


class EvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_example_scenarios_pass_scripted_and_live_ones_skip(self) -> None:
        # Application-backed scenarios (Phase 15c) get the factory `agent-harness eval` uses.
        factory = functools.partial(scenario_application, default_config=EXAMPLES.parent / "models.toml")
        with mock.patch.dict("os.environ", {
            "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/settings.toml",
            "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise.toml",
        }):
            results = {
                scenario.name: await run_scenario(scenario, tools=create_default_runtime(), app_factory=factory)
                for scenario in load_scenarios(EXAMPLES)
            }
        self.assertEqual(results["live_multiplication"].outcome, "skip")
        for name, result in results.items():
            if name != "live_multiplication":
                self.assertEqual(result.outcome, "pass", (name, result.failures))

    async def test_failures_explain_what_went_wrong(self) -> None:
        scenario = load_scenarios(EXAMPLES / "calculator.toml")[0]
        wrong = Scenario(
            name="wrong",
            prompt=scenario.prompt,
            scripted=(ModelResponse(content="I think it is 41."),),
            expect=scenario.expect,
        )
        result = await run_scenario(wrong, tools=create_default_runtime())
        self.assertEqual(result.outcome, "fail")
        joined = " | ".join(result.failures)
        self.assertIn("do not contain ['calculator']", joined)
        self.assertIn("output does not contain '42'", joined)

    async def test_approval_scenarios_are_auto_decided(self) -> None:
        deployed: list[str] = []

        async def deploy(arguments: JsonObject) -> JsonObject:
            deployed.append(arguments["service"])
            return {"ok": True}

        registry = ToolRegistry()
        registry.register(
            ToolSpec(name="deploy", description="deploy", input_schema={"type": "object"},
                     risk=ToolRisk.SIDE_EFFECT, idempotent=False),
            deploy,
        )
        runtime = ToolRuntime(registry, policy=DefaultToolPolicy(ApprovalMode.SIDE_EFFECTS))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "deploy.toml")
            path.write_text(
                'prompt = "deploy billing"\napprove = "allow"\n'
                '[[model]]\ntool_calls = [{ name = "deploy", arguments = { service = "billing" } }]\n'
                '[[model]]\ncontent = "deployed"\n'
                '[expect]\ntool_calls = ["deploy"]\nevents_include = ["approval_requested", "approval_resolved"]\n',
                encoding="utf-8",
            )
            result = await run_scenario(load_scenarios(path)[0], tools=runtime)
        self.assertEqual(result.outcome, "pass", result.failures)
        self.assertEqual(deployed, ["billing"])

    async def test_live_mode_uses_the_given_models(self) -> None:
        live = ScriptedModel([ModelResponse(content="513")])
        scenario = load_scenarios(EXAMPLES / "live_multiplication.toml")[0]
        chosen: list[str | None] = []

        def pick(name: str | None) -> str:
            chosen.append(name)
            return "ollama"

        result = await run_scenario(scenario, tools=create_default_runtime(), live_models=live, live_profile=pick)
        self.assertEqual(result.mode, "live")
        self.assertEqual(result.status, RunStatus.COMPLETED.value)
        self.assertIn("tool calls [] do not contain ['calculator']", result.failures[0])
        self.assertEqual(chosen, [None])

    def test_malformed_scenarios_are_rejected(self) -> None:
        cases = {
            "unknown.toml": 'prompt = "x"\nsurprise = 1\n',
            "noprompt.toml": 'name = "x"\n',
            "status.toml": 'prompt = "x"\n[expect]\nstatus = "finished"\n',
            "event.toml": 'prompt = "x"\n[expect]\nevents_include = ["nope"]\n',
            "approve.toml": 'prompt = "x"\napprove = "maybe"\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, text in cases.items():
                path = Path(directory, name)
                path.write_text(text, encoding="utf-8")
                with self.subTest(case=name), self.assertRaises(ScenarioError):
                    load_scenarios(path)
            with self.assertRaises(ScenarioError):
                load_scenarios(Path(directory, "empty-dir-does-not-exist"))

    def test_cli_eval_reports_and_sets_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            report = Path(directory, "report.json")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli.main(["eval", "--config", str(config), str(EXAMPLES), "--report", str(report)])
            self.assertEqual(code, 0)
            scripted = sum(s.scripted is not None for s in load_scenarios(EXAMPLES))
            # every scripted example passes (live-only ones skip)
            self.assertEqual(json.loads(report.read_text())["summary"],
                             {"pass": scripted, "fail": 0, "skip": len(load_scenarios(EXAMPLES)) - scripted})
            self.assertIn("[PASS] calculator_multiplication", out.getvalue())

            failing = Path(directory, "failing.toml")
            failing.write_text('prompt = "x"\n[[model]]\ncontent = "no"\n[expect]\noutput_contains = ["yes"]\n')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["eval", "--config", str(config), str(failing)]), 1)


if __name__ == "__main__":
    unittest.main()
