"""The .env loader and the AGENT_HARNESS_MODEL_PROFILE switch."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from agent_harness import cli
from agent_harness.bench import cli as bench_cli
from agent_harness.config import load_config
from agent_harness.dotenv import load_dotenv, parse_dotenv

PROFILES = ('default_model_profile = "a"\n'
            '[models.a]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://a/v1"\n'
            '[models.b]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://b/v1"\n')


class ParseTests(unittest.TestCase):
    def test_formats(self) -> None:
        values = parse_dotenv(
            "# comment\n\nexport A=1\nB = two words # trailing comment\nC='{\"x\": 1} # kept'\n"
            'D="line\\nbreak \\"q\\""\nnot an assignment\n1BAD=x\nE=\n'
        )
        self.assertEqual(values, {"A": "1", "B": "two words", "C": '{"x": 1} # kept',
                                  "D": 'line\nbreak "q"', "E": ""})


class LoadTests(unittest.TestCase):
    def test_existing_variables_win_and_the_switch_turns_it_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, ".env")
            path.write_text("AGENT_HARNESS_MODEL_PROFILE=b\nSECRET=s\n")
            env = {"AGENT_HARNESS_MODEL_PROFILE": "a"}
            self.assertEqual(load_dotenv(path, env), ["SECRET"])
            self.assertEqual(env, {"AGENT_HARNESS_MODEL_PROFILE": "a", "SECRET": "s"})
            off = {"AGENT_HARNESS_DOTENV": "0"}
            self.assertEqual((load_dotenv(path, off), off), ([], {"AGENT_HARNESS_DOTENV": "0"}))
            named = {"AGENT_HARNESS_ENV_FILE": str(path)}
            load_dotenv(None, named)
            self.assertEqual(named["SECRET"], "s")
            self.assertEqual(load_dotenv(Path(directory, "missing"), {}), [])


class SwitchTests(unittest.TestCase):
    def test_new_name_wins_over_the_old_one_and_the_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(PROFILES)
            self.assertEqual(load_config(path, environ={}).default_model_profile, "a")
            self.assertEqual(load_config(path, environ={"AGENT_HARNESS_MODEL_PROFILE": "b"}).default_model_profile, "b")
            legacy = {"AGENT_HARNESS_DEFAULT_MODEL_PROFILE": "b"}
            self.assertEqual(load_config(path, environ=legacy).default_model_profile, "b")  # still read
            both = {"AGENT_HARNESS_MODEL_PROFILE": "a", "AGENT_HARNESS_DEFAULT_MODEL_PROFILE": "b"}
            self.assertEqual(load_config(path, environ=both).default_model_profile, "a")


class EntryPointTests(unittest.TestCase):
    def test_programs_read_dotenv_but_calls_with_argv_do_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "models.toml").write_text(PROFILES)
            Path(directory, ".env").write_text("AGENT_HARNESS_MODEL_PROFILE=b\n")
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                with mock.patch.dict(os.environ, {"AGENT_HARNESS_USER_SETTINGS": "/nonexistent",
                                                  "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent"}):
                    os.environ.pop("AGENT_HARNESS_MODEL_PROFILE", None)
                    os.environ.pop("AGENT_HARNESS_DOTENV", None)
                    out = io.StringIO()
                    with redirect_stdout(out):
                        cli.main(["profiles"])  # a test / library call: no .env
                    self.assertIn("a *:", out.getvalue())
                    self.assertNotIn("AGENT_HARNESS_MODEL_PROFILE", os.environ)
                    out = io.StringIO()
                    with redirect_stdout(out), mock.patch("sys.argv", ["agent-harness", "profiles"]):
                        cli.main()  # started as a program: ./.env is read
                    self.assertIn("b *:", out.getvalue())
            finally:
                os.chdir(cwd)

    def test_offline_pipeline_stages_never_reread_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "evals").mkdir()
            (base / "models.toml").write_text(PROFILES)
            (base / "plan.toml").write_text('[pipeline]\nconfig = "models.toml"\nrepo = "."\noutput_dir = "out"\n'
                                            '[scenarios]\npath = "evals"\n')
            seen: list[dict] = []

            def fake(argv, *, cwd, env):
                seen.append(dict(env))
                return True, "ok"

            with mock.patch.object(bench_cli, "_run_command", side_effect=fake), \
                    mock.patch.dict(os.environ, {"AGENT_HARNESS_MODEL_PROFILE": "b"}):
                asyncio.run(bench_cli.run_pipeline(base / "plan.toml", stages=("unit", "scenarios")))
        for env in seen:
            self.assertEqual(env["AGENT_HARNESS_DOTENV"], "0")
            self.assertNotIn("AGENT_HARNESS_MODEL_PROFILE", env)


if __name__ == "__main__":
    unittest.main()
