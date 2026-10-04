"""Network-free tests for Phase 6b: hooks, permission rules and modes, plan mode."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_harness.application import build_application
from agent_harness.config.loader import ConfigError, load_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    JsonObject,
    Message,
    ModelRequest,
    ModelResponse,
    PermissionMode,
    Role,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
)
from agent_harness.extensions.hooks import CallableHook, CommandHook, HookEvent, HookInput, HookOutcome, HookRunner
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.extensions.permissions import (
    EXIT_PLAN_TOOL,
    PermissionPolicy,
    PermissionRule,
    PermissionRuleError,
)
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime


READ = ToolSpec(name="read_file", description="read", input_schema={"type": "object"})
WRITE = ToolSpec(
    name="write_file", description="write",
    input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    risk=ToolRisk.SIDE_EFFECT, idempotent=False,
)
RUN = ToolSpec(
    name="run_command", description="run",
    input_schema={"type": "object", "properties": {"argv": {"type": "array", "items": {"type": "string"}}}},
    risk=ToolRisk.SIDE_EFFECT, idempotent=False,
)


class Model:
    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "Model":
        return self


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, JsonObject]] = []

    def handler(self, name: str):
        async def handle(arguments: JsonObject) -> JsonObject:
            self.calls.append((name, dict(arguments)))
            return {"ok": True, "tool": name}

        return handle


def runtime(recorder: Recorder, approval: ApprovalMode = ApprovalMode.SIDE_EFFECTS) -> ToolRuntime:
    registry = ToolRegistry()
    for spec in (READ, WRITE, RUN):
        registry.register(spec, recorder.handler(spec.name))
    return ToolRuntime(registry, policy=DefaultToolPolicy(approval))


def call(name: str, arguments: dict | None = None, call_id: str = "c1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments or {}),), finish_reason=FinishReason.TOOL_CALLS)


def done(text: str = "done") -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP)


def tool_messages(run: RunRecord) -> list[str]:
    return [m.content for m in run.messages if m.role is Role.TOOL]


class Setup:
    def __init__(self, outcomes: Sequence[ModelResponse], **kwargs: Any) -> None:
        self.store = kwargs.pop("store", None) or InMemoryRunStore()
        self.model = Model(outcomes)
        self.recorder = Recorder()
        self.orchestrator = AgentOrchestrator(
            models=self.model, tools=kwargs.pop("tools", None) or runtime(self.recorder),
            store=self.store, **kwargs,
        )

    async def start(self, prompt: str = "go", **kwargs: Any) -> RunRecord:
        run = await self.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, prompt)], **kwargs)
        return await self.orchestrator.run(run.id)

    async def events(self, run_id: str, kind: EventType) -> list[RunEvent]:
        return [e for e in await self.store.list_events(run_id) if e.type is kind]


# --- Rules --------------------------------------------------------------------------


class RuleTests(unittest.TestCase):
    def test_rule_matching(self) -> None:
        pytest = ToolCall("c", "run_command", {"argv": ["pytest", "-q"]})
        cases = [
            ("run_command", pytest, True),
            ("run_*", pytest, True),
            ("run_command(argv.0=pytest)", pytest, True),
            ("run_command(argv.0=git)", pytest, False),
            ("run_command(argv.5=x)", pytest, False),
            ("read_file(path=src/*)", ToolCall("c", "read_file", {"path": "src/app.py"}), True),
            ("read_file(path=src/*, mode=w)", ToolCall("c", "read_file", {"path": "src/a"}), False),
        ]
        for text, tool_call, expected in cases:
            with self.subTest(rule=text):
                self.assertIs(PermissionRule.parse(text).matches(tool_call), expected)
        for bad in ("", "run(argv.0)", "run(=x)", "bad name!"):
            with self.subTest(bad=bad), self.assertRaises(PermissionRuleError):
                PermissionRule.parse(bad)

    def test_decision_order(self) -> None:
        policy = PermissionPolicy.from_lists(
            allow=["write_file(path=docs/*)"], ask=["read_file(path=*.env)"], deny=["run_command(argv.0=rm)"]
        )
        rm = ToolCall("c", "run_command", {"argv": ["rm", "-rf", "x"]})
        ls = ToolCall("c", "run_command", {"argv": ["ls"]})
        docs = ToolCall("c", "write_file", {"path": "docs/a.md"})
        env = ToolCall("c", "read_file", {"path": ".env"})
        patch = ToolCall("c", "apply_patch", {})
        patch_spec = ToolSpec(name="apply_patch", description="", input_schema={"type": "object"}, risk=ToolRisk.SIDE_EFFECT)

        self.assertEqual(policy.decide(RUN, rm, PermissionMode.BYPASS).action, "deny")
        self.assertEqual(policy.decide(RUN, ls, PermissionMode.BYPASS).action, "allow")
        self.assertEqual(policy.decide(RUN, ls, PermissionMode.PLAN).action, "deny")
        self.assertIsNone(policy.decide(READ, ToolCall("c", "read_file", {"path": "a"}), PermissionMode.PLAN).action)
        self.assertIsNone(policy.decide(READ, ToolCall("c", EXIT_PLAN_TOOL, {}), PermissionMode.PLAN).action)
        self.assertEqual(policy.decide(patch_spec, patch, PermissionMode.ACCEPT_EDITS).action, "allow")
        self.assertIsNone(policy.decide(RUN, ls, PermissionMode.ACCEPT_EDITS).action)
        self.assertEqual(policy.decide(READ, env, PermissionMode.DEFAULT).action, "ask")
        self.assertEqual(policy.decide(WRITE, docs, PermissionMode.DEFAULT).action, "allow")
        self.assertIsNone(policy.decide(RUN, ls, PermissionMode.DEFAULT).action)


# --- Permissions in the orchestrator -------------------------------------------------


class PermissionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_deny_rule_blocks_and_explains(self) -> None:
        setup = Setup([call("run_command", {"argv": ["rm", "-rf", "/"]}), done()],
                      permissions=PermissionPolicy.from_lists(deny=["run_command(argv.0=rm)"]),
                      default_mode=PermissionMode.BYPASS)
        finished = await setup.start()
        self.assertEqual(setup.recorder.calls, [])
        self.assertIn("Denied: denied by permission rule", tool_messages(finished)[0])
        responded = await setup.events(finished.id, EventType.TOOL_RESPONDED)
        self.assertEqual(responded[0].payload["denied_by"], "permission")

    async def test_ask_rule_pauses_even_read_only_tools(self) -> None:
        setup = Setup([call("read_file", {"path": "prod.env"}), done()],
                      permissions=PermissionPolicy.from_lists(ask=["read_file(path=*.env)"]))
        paused = await setup.start()
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        self.assertEqual(setup.recorder.calls, [])
        with self.assertRaisesRegex(ValueError, "can only be chosen when approving a plan"):
            await setup.orchestrator.resolve_approval(paused.id, call_id="c1", approved=True, mode="bypass")
        await setup.orchestrator.resolve_approval(paused.id, call_id="c1", approved=True)
        finished = await setup.orchestrator.run(paused.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(setup.recorder.calls, [("read_file", {"path": "prod.env"})])

    async def test_allow_rule_and_bypass_skip_the_approval_pause(self) -> None:
        allowed = Setup([call("write_file", {"path": "docs/x.md"}), done()],
                        permissions=PermissionPolicy.from_lists(allow=["write_file(path=docs/*)"]))
        self.assertEqual((await allowed.start()).status, RunStatus.COMPLETED)
        self.assertEqual(len(allowed.recorder.calls), 1)

        default = Setup([call("write_file", {"path": "src/x.py"}), done()])
        self.assertEqual((await default.start()).status, RunStatus.AWAITING_INPUT)

        bypass = Setup([call("write_file", {"path": "src/x.py"}), call("write_file", {}, "c2"), done()])
        finished = await bypass.start(mode=PermissionMode.BYPASS)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(bypass.recorder.calls), 1)  # invalid arguments still rejected
        self.assertIn("Invalid tool arguments", tool_messages(finished)[1])

    async def test_plan_mode_round_trip(self) -> None:
        setup = Setup([
            call("write_file", {"path": "a.py"}),
            call("read_file", {"path": "a.py"}, "c2"),
            call(EXIT_PLAN_TOOL, {"plan": "Edit a.py"}, "c3"),
            call(EXIT_PLAN_TOOL, {"plan": "Edit a.py, with tests"}, "c4"),
            call("write_file", {"path": "a.py"}, "c5"),
            done(),
        ])
        paused = await setup.start(mode=PermissionMode.PLAN)
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        self.assertIn(EXIT_PLAN_TOOL, [t.name for t in setup.model.requests[0].tools])
        self.assertEqual(setup.recorder.calls, [("read_file", {"path": "a.py"})])
        self.assertIn("plan mode allows only read-only tools", tool_messages(paused)[0])
        pending = await setup.orchestrator.pending_approval(paused.id)
        self.assertEqual((pending.kind, pending.arguments), ("plan", {"plan": "Edit a.py"}))

        await setup.orchestrator.resolve_approval(paused.id, call_id="c3", approved=False, feedback="Add tests too")
        again = await setup.orchestrator.run(paused.id)
        self.assertIn("Their feedback: Add tests too", tool_messages(again)[2])
        await setup.orchestrator.resolve_approval(again.id, call_id="c4", approved=True, mode=PermissionMode.ACCEPT_EDITS)
        paused_for_write = await setup.orchestrator.run(again.id)

        self.assertEqual(paused_for_write.mode, PermissionMode.ACCEPT_EDITS)
        self.assertNotIn(EXIT_PLAN_TOOL, [t.name for t in setup.model.requests[-1].tools])
        changed = await setup.events(again.id, EventType.MODE_CHANGED)
        self.assertEqual((changed[0].payload["from"], changed[0].payload["to"]), ("plan", "accept_edits"))
        # accept_edits does not cover write_file (only configured edit tools), so it asks.
        self.assertEqual(paused_for_write.status, RunStatus.AWAITING_INPUT)

    async def test_set_mode(self) -> None:
        setup = Setup([done()])
        run = await setup.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        updated = await setup.orchestrator.set_mode(run.id, "plan")
        self.assertEqual(updated.mode, PermissionMode.PLAN)
        stored = await setup.store.get_run(run.id)
        self.assertEqual(stored.mode, PermissionMode.PLAN)
        await setup.orchestrator.cancel(run.id)
        with self.assertRaises(ValueError):
            await setup.orchestrator.set_mode(run.id, "default")


# --- Hooks ----------------------------------------------------------------------------


def runner(*hooks: CallableHook) -> HookRunner:
    return HookRunner(list(hooks))


class ToolHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_pre_tool_deny_and_record(self) -> None:
        hook = CallableHook("no-prod", HookEvent.PRE_TOOL,
                            lambda i: HookOutcome("deny", "production is off limits"), matcher="write_*")
        setup = Setup([call("write_file", {"path": "prod.cfg"}), done()], hooks=runner(hook),
                      default_mode=PermissionMode.BYPASS)
        finished = await setup.start()
        self.assertEqual(setup.recorder.calls, [])
        self.assertIn("production is off limits", tool_messages(finished)[0])
        decided = await setup.events(finished.id, EventType.HOOK_DECIDED)
        self.assertEqual((decided[0].payload["hook_event"], decided[0].payload["decision"]), ("pre_tool", "deny"))

    async def test_modified_arguments_are_executed_and_rechecked_by_permissions(self) -> None:
        def rewrite(hook_input: HookInput) -> HookOutcome:
            return HookOutcome("modify", arguments={"argv": ["pytest", "-q"]})

        setup = Setup([call("run_command", {"argv": ["pytest"]}), done()],
                      hooks=runner(CallableHook("quiet", HookEvent.PRE_TOOL, rewrite)),
                      default_mode=PermissionMode.BYPASS)
        finished = await setup.start()
        self.assertEqual(setup.recorder.calls, [("run_command", {"argv": ["pytest", "-q"]})])
        requested = await setup.events(finished.id, EventType.TOOL_REQUESTED)
        self.assertEqual(requested[0].payload["arguments"], {"argv": ["pytest", "-q"]})

        sneaky = Setup([call("run_command", {"argv": ["ls"]}), done()],
                       hooks=runner(CallableHook("x", HookEvent.PRE_TOOL,
                                                 lambda i: HookOutcome("modify", arguments={"argv": ["rm", "-rf"]}))),
                       permissions=PermissionPolicy.from_lists(deny=["run_command(argv.0=rm)"]),
                       default_mode=PermissionMode.BYPASS)
        await sneaky.start()
        self.assertEqual(sneaky.recorder.calls, [])

    async def test_failing_pre_tool_hook_fails_closed_unless_configured(self) -> None:
        def broken(hook_input: HookInput) -> HookOutcome:
            raise RuntimeError("policy service down")

        closed = Setup([call("read_file", {"path": "a"}), done()],
                       hooks=runner(CallableHook("guard", HookEvent.PRE_TOOL, broken)))
        finished = await closed.start()
        self.assertEqual(closed.recorder.calls, [])
        self.assertIn("hook 'guard' failed", tool_messages(finished)[0])

        opened = Setup([call("read_file", {"path": "a"}), done()],
                       hooks=runner(CallableHook("guard", HookEvent.PRE_TOOL, broken, on_error="allow")))
        await opened.start()
        self.assertEqual(len(opened.recorder.calls), 1)

    async def test_post_tool_context_is_added_to_the_result(self) -> None:
        hook = CallableHook("lint", HookEvent.POST_TOOL, lambda i: HookOutcome(context=f"lint ok after {i.tool}"))
        setup = Setup([call("read_file", {"path": "a"}), done()], hooks=runner(hook))
        finished = await setup.start()
        self.assertIn("[Context from hooks]\nlint ok after read_file", tool_messages(finished)[0])

    async def test_resume_replays_recorded_hook_decision(self) -> None:
        class CrashOnResult(SQLiteRunStore):
            crashed = False

            async def save_checkpoint(self, run, events, *, expected_version):
                if not self.crashed and any(e.type is EventType.TOOL_RESPONDED for e in events):
                    self.crashed = True
                    raise KeyboardInterrupt
                await super().save_checkpoint(run, events, expected_version=expected_version)

        calls: list[str] = []

        def rewrite(hook_input: HookInput) -> HookOutcome:
            calls.append("first")
            return HookOutcome("modify", arguments={"path": "rewritten.txt"})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "db.sqlite3")
            store = CrashOnResult(path)
            first = Setup([call("read_file", {"path": "orig.txt"})], store=store,
                          hooks=runner(CallableHook("h", HookEvent.PRE_TOOL, rewrite)))
            run = await first.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
            with self.assertRaises(KeyboardInterrupt):
                await first.orchestrator.run(run.id)
            await store.close()

            def would_deny(hook_input: HookInput) -> HookOutcome:
                calls.append("second")
                return HookOutcome("deny", "changed my mind")

            second_store = SQLiteRunStore(path)
            second = Setup([done()], store=second_store,
                           hooks=runner(CallableHook("h", HookEvent.PRE_TOOL, would_deny)))
            finished = await second.orchestrator.run(run.id)
            await second_store.close()

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(calls, ["first"])
        self.assertEqual(second.recorder.calls, [("read_file", {"path": "rewritten.txt"})])


class LifecycleHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_prompt_deny_modify_and_context(self) -> None:
        def screen(hook_input: HookInput) -> HookOutcome:
            if "password" in (hook_input.prompt or ""):
                return HookOutcome("deny", "prompts must not contain credentials")
            return HookOutcome("modify", content=(hook_input.prompt or "").strip(), context="Repo: harness")

        setup = Setup([done(), done()], hooks=runner(CallableHook("screen", HookEvent.USER_PROMPT, screen)))
        with self.assertRaisesRegex(ValueError, "credentials"):
            await setup.start("my password is hunter2")
        finished = await setup.start("   summarize   ")
        user_texts = [m.content for m in finished.messages if m.role is Role.USER]
        self.assertEqual(user_texts, ["summarize", "[Context from hooks]\nRepo: harness"])
        created = (await setup.store.list_events(finished.id))[0]
        self.assertEqual(created.payload["user_prompt_hook"]["decision"], "modify")
        with self.assertRaisesRegex(ValueError, "credentials"):
            await setup.orchestrator.continue_run(finished.id, "password: x")

    async def test_session_start_context_and_deny(self) -> None:
        setup = Setup([done()], hooks=runner(CallableHook("ctx", HookEvent.SESSION_START,
                                                          lambda i: HookOutcome(context="Today is release day"))))
        finished = await setup.start()
        self.assertIn("Today is release day", setup.model.requests[0].messages[-1].content)
        self.assertEqual(finished.status, RunStatus.COMPLETED)

        blocked = Setup([], hooks=runner(CallableHook("gate", HookEvent.SESSION_START,
                                                      lambda i: HookOutcome("deny", "maintenance window"))))
        failed = await blocked.start()
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertIn("maintenance window", failed.error or "")

    async def test_pre_model_context_is_ephemeral_and_deny_fails(self) -> None:
        hook = CallableHook("clock", HookEvent.PRE_MODEL, lambda i: HookOutcome(context=f"step {i.step}"))
        setup = Setup([call("read_file", {"path": "a"}), done()], hooks=runner(hook))
        finished = await setup.start()
        self.assertEqual(setup.model.requests[1].messages[-1].content, "[Context from hooks]\nstep 2")
        self.assertFalse(any("step 1" in m.content for m in finished.messages))
        self.assertEqual(setup.model.requests[1].messages[:-1][:3], tuple(finished.messages[:3]))

        blocked = Setup([], hooks=runner(CallableHook("quota", HookEvent.PRE_MODEL,
                                                      lambda i: HookOutcome("deny", "daily quota reached"))))
        failed = await blocked.start()
        self.assertIn("daily quota reached", failed.error or "")

    async def test_stop_hook_can_request_more_work_but_is_bounded(self) -> None:
        hook = CallableHook("tests", HookEvent.STOP, lambda i: HookOutcome("continue", "Run the tests before finishing."))
        setup = Setup([done("a"), done("b"), done("c")], hooks=runner(hook), max_stop_continuations=2)
        finished = await setup.start()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(setup.model.requests), 3)
        self.assertEqual(sum(m.content == "Run the tests before finishing." for m in finished.messages), 2)

    async def test_annotation_after_final_answer_does_not_trigger_another_model_call(self) -> None:
        class CrashOnComplete(SQLiteRunStore):
            crashed = False

            async def save_checkpoint(self, run, events, *, expected_version):
                if not self.crashed and any(e.type is EventType.RUN_COMPLETED for e in events):
                    self.crashed = True
                    raise KeyboardInterrupt
                await super().save_checkpoint(run, events, expected_version=expected_version)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "db.sqlite3")
            store = CrashOnComplete(path)
            first = Setup([done("final")], store=store)
            run = await first.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
            with self.assertRaises(KeyboardInterrupt):
                await first.orchestrator.run(run.id)
            await store.close()
            second_store = SQLiteRunStore(path)
            registry = ToolRegistry()
            registry.register(READ, Recorder().handler("read_file"))  # different tools -> config_drift
            second = Setup([], store=second_store, tools=ToolRuntime(registry))
            finished = await second.orchestrator.run(run.id)
            drift = await second.events(run.id, EventType.CONFIG_DRIFT)
            await second_store.close()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(second.model.requests, [])
        self.assertEqual(len(drift), 1)


class CommandHookTests(unittest.IsolatedAsyncioTestCase):
    def _hook(self, script: str, **kwargs: Any) -> CommandHook:
        return CommandHook("cmd", HookEvent.PRE_TOOL, [sys.executable, "-c", script], **kwargs)

    async def test_json_protocol(self) -> None:
        script = (
            "import json,sys; data=json.load(sys.stdin); "
            "print(json.dumps({'decision':'modify','arguments':{'path':data['arguments']['path'].upper()}}))"
        )
        outcome = await self._hook(script).run(HookInput(HookEvent.PRE_TOOL, "r", tool="read_file", arguments={"path": "a.py"}))
        self.assertEqual((outcome.decision, outcome.arguments), ("modify", {"path": "A.PY"}))
        empty = await self._hook("pass").run(HookInput(HookEvent.PRE_TOOL, "r"))
        self.assertEqual(empty.decision, "allow")

    async def test_errors(self) -> None:
        from agent_harness.extensions.hooks import HookError

        for script, message in (
            ("import sys; sys.exit(3)", "exited with 3"),
            ("print('not json')", "invalid output"),
            ("print('{\"decision\": \"explode\"}')", "invalid output"),
            ("import time; time.sleep(5)", "timed out"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(HookError, message):
                await self._hook(script, timeout_seconds=0.5).run(HookInput(HookEvent.PRE_TOOL, "r"))
        with self.assertRaises(ValueError):
            CommandHook("bad", HookEvent.PRE_TOOL, [])


class ConfigTests(unittest.IsolatedAsyncioTestCase):
    def _write(self, directory: str, extra: str) -> Path:
        path = Path(directory, "models.toml")
        path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n' + extra)
        return path

    async def test_sections_parse_validate_and_wire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hook_script = Path(directory, "hook.py")
            hook_script.write_text("import json; print(json.dumps({'context': 'from config hook'}))")
            path = self._write(directory, f'''
[permissions]
mode = "plan"
deny = ["run_command(argv.0=rm)"]
allow = ["read_file"]

[[hooks]]
event = "session_start"
command = ["{sys.executable}", "hook.py"]
timeout_seconds = 5
''')
            config = load_config(path, environ={})
            self.assertEqual((config.permissions.mode, config.permissions.deny), ("plan", ("run_command(argv.0=rm)",)))
            self.assertEqual(config.hooks[0].command, (sys.executable, "hook.py"))
            app = build_application(config_path=path, database_path=Path(directory, "db.sqlite3"))
            async with app:
                run = await app.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
                self.assertEqual(run.mode, PermissionMode.PLAN)
            for bad in ('[permissions]\nmode = "yolo"\n', '[permissions]\ndeny = ["bad rule!"]\n',
                        '[permissions]\nsurprise = 1\n', '[[hooks]]\nevent = "nope"\ncommand = ["x"]\n',
                        '[[hooks]]\nevent = "stop"\ncommand = "rm -rf /"\n'):
                with self.subTest(bad=bad), self.assertRaises(ConfigError):
                    load_config(self._write(directory, bad), environ={})


if __name__ == "__main__":
    unittest.main()
