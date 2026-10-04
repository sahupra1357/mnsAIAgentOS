"""Phase 15c: per-feature validation — feature selection, coverage, app-backed scenarios."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_harness import cli
from agent_harness.bench import cli as bench_cli
from agent_harness.bench.core import BenchConfigError
from agent_harness.bench.features import parse_features, select_features, uncovered
from agent_harness.config import load_config
from agent_harness.evals import ScenarioError, load_scenarios, run_scenario

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "examples" / "validation.toml"
EVALS = ROOT / "examples" / "evals"
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise.toml",
}


def plan() -> dict:
    return tomllib.loads(PLAN.read_text(encoding="utf-8"))


class CoverageTests(unittest.TestCase):
    """Every check belongs to a feature, and every name a feature lists exists."""

    def setUp(self) -> None:
        self.features = parse_features(plan())

    def test_every_test_module_is_in_a_feature(self) -> None:
        modules = [path.stem for path in (ROOT / "tests").glob("test_*.py")]
        self.assertEqual(uncovered(self.features.values(), "tests", modules), [])

    def test_every_scenario_is_in_a_feature(self) -> None:
        # Features name scenario files (stems); live-only scenarios run with --live.
        stems = [s.path.stem for s in load_scenarios(EVALS) if s.scripted is not None and s.path]
        self.assertEqual(uncovered(self.features.values(), "scenarios", stems), [])

    def test_every_live_entry_is_in_a_feature(self) -> None:
        raw = plan()
        self.assertEqual(uncovered(self.features.values(), "smoke",
                                   [str(e.get("profile", "default")) for e in raw.get("smoke", [])]), [])
        self.assertEqual(uncovered(self.features.values(), "benchmarks",
                                   [str(e["name"]) for e in raw.get("benchmark", [])]), [])

    def test_every_benchmark_sets_its_own_allowed_drop(self) -> None:
        # The 0.03 default is less than one item for every benchmark here (one item is 100/n
        # points), so a single different answer would fail the gate.
        missing = [e["name"] for e in plan().get("benchmark", [])
                   if "max_accuracy_drop" not in e.get("thresholds", {})]
        self.assertEqual(missing, [])

    def test_every_live_benchmark_has_a_baseline_path(self) -> None:
        # Without `baseline`, --update-baselines has nowhere to save and gates never compare.
        missing = [e["name"] for e in plan().get("benchmark", []) if not e.get("baseline")]
        self.assertEqual(missing, [])

    def test_downloaded_datasets_come_from_the_fetch_script(self) -> None:
        script = (ROOT / "scripts" / "fetch_datasets.sh").read_text(encoding="utf-8")
        for entry in plan().get("benchmark", []):
            for key in ("data", "answers"):
                value = str(entry.get(key, ""))
                if value.startswith("../datasets/"):
                    self.assertIn(Path(value).name, script, (entry["name"], value))
                elif value:
                    self.assertTrue((PLAN.parent / value).is_file(), (entry["name"], value))

    def test_listed_names_exist(self) -> None:
        for feature in self.features.values():
            for module in feature.tests:
                self.assertTrue((ROOT / "tests" / f"{module}.py").is_file(), (feature.name, module))
            for scenario in feature.scenarios:
                self.assertTrue((EVALS / f"{scenario}.toml").is_file(), (feature.name, scenario))

    def test_plan_profiles_exist_in_its_config(self) -> None:
        raw = plan()
        with mock.patch.dict("os.environ", HERMETIC):
            config = load_config(PLAN.parent / raw["pipeline"]["config"], environ={})
        for entry in [*raw.get("smoke", []), *raw.get("benchmark", [])]:
            # No profile: the entry follows the config's default profile (Phase 16).
            self.assertIn(entry.get("profile", config.default_model_profile), config.models, entry)


class SelectionTests(unittest.TestCase):
    def test_union_in_order_and_unknown_names(self) -> None:
        features = parse_features(plan())
        selection = select_features(features, ["memory", "tools", "memory"])
        self.assertEqual(selection.names, ("memory", "tools"))
        self.assertEqual(selection.tests.count("test_phase14"), 1)
        self.assertIn("memory_roundtrip", selection.scenarios)
        with self.assertRaisesRegex(BenchConfigError, "unknown feature"):
            select_features(features, ["nope"])

    def test_invalid_entries(self) -> None:
        with self.assertRaisesRegex(BenchConfigError, "unknown smoke/benchmark"):
            parse_features({"feature": [{"name": "x", "benchmarks": ["missing"]}]})
        with self.assertRaisesRegex(BenchConfigError, "defined twice"):
            parse_features({"feature": [{"name": "x"}, {"name": "x"}]})
        with self.assertRaisesRegex(BenchConfigError, "unknown fields"):
            parse_features({"feature": [{"name": "x", "test": []}]})

    def test_features_command_lists_them(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(bench_cli.main(["features", str(PLAN)]), 0)
        self.assertIn("guardrails", out.getvalue())


class FeaturePipelineTests(unittest.TestCase):
    def test_feature_run_limits_unit_and_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "evals").mkdir()
            for name in ("a", "b"):
                (base / "evals" / f"{name}.toml").write_text(f'name = "{name}"\nprompt = "p"\n')
            (base / "models.toml").write_text(
                '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            (base / "plan.toml").write_text(
                '[pipeline]\nconfig = "models.toml"\nrepo = "."\noutput_dir = "out"\n'
                '[scenarios]\npath = "evals"\n'
                '[[smoke]]\nprofile = "p"\n'
                '[[feature]]\nname = "one"\ntests = ["test_x"]\nscenarios = ["a"]\n'
                '[[feature]]\nname = "two"\nsmoke = ["p"]\n')
            calls: list[list[str]] = []

            def fake(argv, *, cwd, env):
                calls.append(list(argv))
                return True, "ok"

            with mock.patch.object(bench_cli, "_run_command", side_effect=fake):
                results = asyncio.run(bench_cli.run_pipeline(
                    base / "plan.toml", stages=("static", "unit", "scenarios", "smoke"), features=["one"]))
        statuses = {r.stage: r.status for r in results}
        self.assertEqual(statuses["static"], "skip")
        self.assertEqual(statuses["unit:one"], "pass")
        self.assertEqual(statuses["scenarios:one"], "pass")
        self.assertEqual(statuses["smoke"], "skip")  # feature `one` lists no smoke profile
        unit = next(c for c in calls if "unittest" in c)
        self.assertEqual(unit[-1], "test_x")
        scenario = next(c for c in calls if "eval" in c)
        self.assertTrue(scenario[scenario.index("eval") + 3].endswith("evals/a.toml"))
        self.assertNotIn(str(base / "evals" / "b.toml"), scenario)

    def test_offline_flag_drops_live_stages(self) -> None:
        captured = {}

        async def fake_pipeline(plan_path, **kwargs):
            captured.update(kwargs)
            return []

        err = io.StringIO()
        with mock.patch.object(bench_cli, "run_pipeline", fake_pipeline), redirect_stderr(err):
            bench_cli.main(["pipeline", str(PLAN), "--offline", "--feature", "guardrails", "--quiet"])
        self.assertEqual(captured["stages"], ("static", "unit", "scenarios"))
        self.assertEqual(captured["features"], ["guardrails"])


class AppScenarioTests(unittest.TestCase):
    def test_bare_only_keys_are_rejected_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "s.toml")
            (Path(directory) / "c.toml").write_text("")
            path.write_text('prompt = "p"\nconfig = "c.toml"\nwait_seconds = 5\n')
            with self.assertRaisesRegex(ScenarioError, "put wait_seconds in"):
                load_scenarios(path)
            path.write_text('prompt = "p"\n[workspace]\n"../x" = "no"\n')
            with self.assertRaisesRegex(ScenarioError, "without '..'"):
                load_scenarios(path)

    def test_app_scenario_without_factory_is_skipped(self) -> None:
        scenario = load_scenarios(EVALS / "task_tracking.toml")[0]
        result = asyncio.run(run_scenario(scenario, tools=None))  # type: ignore[arg-type]
        self.assertEqual(result.outcome, "skip")

    def test_eval_command_runs_app_scenarios_offline(self) -> None:
        names = ["full_config_read_with_hooks", "permission_deny_rule", "subagent_delegation",
                 "memory_roundtrip", "dynamic_tool_loading", "plan_mode_then_edit"]
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", HERMETIC), \
                redirect_stdout(out), redirect_stderr(err):
            report = Path(directory, "report.json")
            code = cli.main(["eval", "--config", str(ROOT / "examples" / "models.toml"),
                             *[str(EVALS / f"{n}.toml") for n in names], "--report", str(report)])
            summary = json.loads(report.read_text())["summary"]
        self.assertEqual(code, 0, out.getvalue())
        self.assertEqual(summary, {"pass": len(names), "fail": 0, "skip": 0})


if __name__ == "__main__":
    unittest.main()
