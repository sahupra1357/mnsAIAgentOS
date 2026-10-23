"""Phase 24: run agent-harness from any folder, like Claude Code (network-free)."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.application import build_application, default_database, state_directory
from agent_harness.config.loader import ConfigError, discover_config_path, load_config
from agent_harness.config.project import (
    absolutize_layer,
    find_project_root,
    project_slug,
    resolve_project_root,
)
from agent_harness.config.settings import load_layers
from agent_harness.domain import FinishReason, Message, ModelRequest, ModelResponse, Role, ToolCall

PROFILE = '[models.m]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
GLOBAL_CONFIG = 'default_model_profile = "m"\n' + PROFILE + """
[project]
root = "cwd"

[coding]
enabled = true
execution_runtime = "local"
allow_local_execution = true
approval_mode = "side_effects"
allowed_commands = ["git", "python"]
command_timeout_seconds = 60

[instructions]
enabled = true
files = ["GLOBAL.md"]

[skills]
enabled = true
paths = ["skills"]

[artifacts]
enabled = true
directory = ".agent-harness/artifacts"
"""


def _skill(folder: Path, name: str, description: str) -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nBody of {name}.\n")


class Layout:
    """A fake home with a global config folder, and a project somewhere else."""

    def __init__(self, root: Path) -> None:
        self.home = root / "home"
        self.global_dir = self.home / ".agent-harness"
        self.global_dir.mkdir(parents=True)
        self.config = self.global_dir / "models.toml"
        self.config.write_text(GLOBAL_CONFIG)
        (self.global_dir / "GLOBAL.md").write_text("Global rule.")
        _skill(self.global_dir / "skills", "global-skill", "From the global folder.")
        self.project = root / "work" / "VideoPipeline"
        (self.project / ".git").mkdir(parents=True)
        (self.project / "sub").mkdir()
        (self.project / "hello.txt").write_text("from the project\n")
        self.env = {
            "HOME": str(self.home),
            "AGENT_HARNESS_USER_SETTINGS": str(root / "none" / "user.toml"),
            "AGENT_HARNESS_ENTERPRISE_SETTINGS": str(root / "none" / "enterprise.toml"),
        }


class ProjectRootTests(unittest.TestCase):
    def test_nearest_marker_wins_and_home_is_never_a_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            home = root / "home"
            (home / ".agent-harness").mkdir(parents=True)
            repo = home / "dev" / "repo"
            (repo / ".git").mkdir(parents=True)
            (repo / "a" / "b").mkdir(parents=True)
            self.assertEqual(find_project_root(repo / "a" / "b", home=home), repo)
            loose = home / "dev" / "loose" / "deep"
            loose.mkdir(parents=True)
            # ~/.agent-harness exists, but home is not a project: the launch folder is.
            self.assertEqual(find_project_root(loose, home=home), loose)
            (home / "dev" / "loose" / ".agent-harness").mkdir()
            self.assertEqual(find_project_root(loose, home=home), home / "dev" / "loose")
            self.assertEqual(find_project_root(loose, (".hg",), home=home), loose)

    def test_slug(self) -> None:
        self.assertEqual(project_slug(Path("/Users/me/dev data/VideoPipeline")),
                         "-Users-me-dev-data-VideoPipeline")

    def test_mode_override_and_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            self.assertIsNone(resolve_project_root({}, {}, config_dir=None, cwd=here))
            self.assertEqual(
                resolve_project_root({}, {"AGENT_HARNESS_SET__PROJECT__ROOT": "cwd"}, config_dir=None,
                                     cwd=here), here.resolve())
            # Started in the config's own folder: not a project run, nothing changes.
            self.assertIsNone(resolve_project_root({"project": {"root": "cwd"}}, {}, config_dir=here, cwd=here))
            self.assertEqual(resolve_project_root({}, {}, config_dir=None, workspace=here), here.resolve())
            with self.assertRaisesRegex(ConfigError, "project.root"):
                resolve_project_root({"project": {"root": "anywhere"}}, {}, config_dir=None, cwd=here)

    def test_absolutize_layer(self) -> None:
        root = Path("/p")
        data = absolutize_layer({
            "skills": {"paths": ["skills", "/abs", "~/x"]},
            "coding": {"workspace": ".", "allowed_commands": ["ffmpeg"]},
            "agents": [{"name": "a", "prompt_file": "agents/a.md"}],
            "tools": [{"name": "t", "kind": "python", "module": "m:f", "path": "tools"}],
            "instructions": {"files": ["NOTES.md"]},
        }, root)
        self.assertEqual(data["skills"]["paths"][:2], ["/p/skills", "/abs"])
        self.assertTrue(Path(data["skills"]["paths"][2]).is_absolute())
        self.assertEqual(data["coding"], {"workspace": "/p", "allowed_commands": ["ffmpeg"]})
        self.assertEqual(data["agents"][0]["prompt_file"], "/p/agents/a.md")
        self.assertEqual(data["tools"][0]["path"], "/p/tools")
        self.assertEqual(data["instructions"]["files"], ["/p/NOTES.md"])


class ClaudeSkillFormatTests(unittest.TestCase):
    def test_folded_and_literal_descriptions(self) -> None:
        from agent_harness.extensions.instructions import discover_skills

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folded").mkdir()
            (root / "folded" / "SKILL.md").write_text(
                "---\nname: folded\ndescription: >\n  Mandatory entry point:\n  read this first.\n"
                "allowed-tools:\n  - Bash\n---\nBody\n")
            (root / "literal").mkdir()
            (root / "literal" / "SKILL.md").write_text(
                "---\nname: literal\ndescription: |\n  Line one.\n  Line two.\n---\nBody\n")
            skills = discover_skills([root])
            self.assertEqual(skills["folded"].description, "Mandatory entry point: read this first.")
            self.assertEqual(skills["literal"].description, "Line one.\nLine two.")


class ProjectConfigTests(unittest.TestCase):
    def test_project_layer_merges_over_the_global_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            (layout.project / ".agent-harness").mkdir()
            (layout.project / ".agent-harness" / "settings.toml").write_text(
                '[coding]\nallowed_commands = ["ffmpeg"]\ncommand_timeout_seconds = 1800\n'
                '[skills]\npaths = ["more-skills"]\n'
            )
            (layout.project / ".agent-harness" / "settings.local.toml").write_text(
                "[coding]\ncommand_timeout_seconds = 900\n"
            )
            config = load_config(layout.config, environ=layout.env, layered=True, cwd=layout.project / "sub")
            self.assertEqual(config.project_root, layout.project.resolve())
            self.assertEqual(config.coding.allowed_commands, ("git", "python", "ffmpeg"))
            self.assertEqual(config.coding.command_timeout_seconds, 900.0)
            self.assertEqual(config.skills.paths, ("skills", str(layout.project.resolve() / "more-skills")))
            settings = load_layers(layout.config, layout.env, cwd=layout.project)
            self.assertEqual([name for name, _ in settings.layers], ["project", "workspace", "workspace-local"])
            self.assertEqual(settings.provenance["coding.command_timeout_seconds"], "workspace-local")
            # [project] settings = false: the project's files are not read.
            layout.config.write_text(GLOBAL_CONFIG.replace('root = "cwd"', 'root = "cwd"\nsettings = false'))
            plain = load_config(layout.config, environ=layout.env, layered=True, cwd=layout.project)
            self.assertEqual(plain.coding.command_timeout_seconds, 60.0)

    def test_enterprise_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            (layout.project / ".agent-harness").mkdir()
            (layout.project / ".agent-harness" / "settings.toml").write_text('[permissions]\nmode = "bypass"\n')
            enterprise = Path(directory) / "enterprise.toml"
            enterprise.write_text('[permissions]\nmode = "default"\n')
            env = {**layout.env, "AGENT_HARNESS_ENTERPRISE_SETTINGS": str(enterprise)}
            config = load_config(layout.config, environ=env, layered=True, cwd=layout.project)
            self.assertEqual(config.permissions.mode, "default")

    def test_config_mode_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            layout.config.write_text(GLOBAL_CONFIG.replace('root = "cwd"', 'root = "config"'))
            config = load_config(layout.config, environ=layout.env, layered=True, cwd=layout.project)
            self.assertIsNone(config.project_root)
            self.assertEqual(default_database(config), "agent-harness.sqlite3")
            self.assertIsNone(state_directory(config))

    def test_symlinked_home_config_is_followed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / "models.toml").write_text(PROFILE)
            home = root / "home"
            (home / ".agent-harness").mkdir(parents=True)
            (home / ".agent-harness" / "models.toml").symlink_to(repo / "models.toml")
            found = discover_config_path(None, {"HOME": str(home)}, root)
            self.assertEqual(found, (repo / "models.toml").resolve())


class ProjectApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_run_started_in_a_project_works_on_that_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            (layout.project / "AGENTS.md").write_text("Project rule.")
            (layout.project / "CLAUDE.md").write_text("Claude rule.")
            _skill(layout.project / ".agent-harness" / "skills", "video", "Project video skill.")
            _skill(layout.project / ".claude" / "skills", "claude-project", "Project Claude skill.")
            _skill(layout.home / ".claude" / "skills", "claude-user", "User Claude skill.")
            _skill(layout.home / ".claude" / "skills", "video", "Shadowed by the project's.")
            model = _Model()
            cwd = os.getcwd()
            os.chdir(layout.project / "sub")
            try:
                with mock.patch.dict(os.environ, layout.env):
                    state = layout.home / ".agent-harness" / "projects" / project_slug(layout.project)
                    database = default_database(load_config(layout.config, layered=True))
                    self.assertEqual(Path(database), state / "runs.sqlite3")
                    async with build_application(config_path=layout.config, database_path=database,
                                                 model_resolver=model) as app:
                        self.assertEqual(app.config.project_root, layout.project.resolve())
                        self.assertEqual(app.orchestrator._snapshots.root, layout.project.resolve())
                        self.assertEqual(app.artifacts.root, state / "artifacts")
                        self.assertEqual(sorted(app.skills),
                                         ["claude-project", "claude-user", "global-skill", "video"])
                        self.assertEqual(app.skills["video"].description, "Project video skill.")
                        run = await app.orchestrator.create_run(
                            model_profile="m", messages=[Message(role=Role.USER, content="read hello.txt")]
                        )
                        done = await app.orchestrator.run(run.id)
            finally:
                os.chdir(cwd)
            self.assertEqual(done.status.value, "completed")
            self.assertTrue((state / "runs.sqlite3").is_file())
            self.assertFalse((layout.project / "agent-harness.sqlite3").exists())
            system = "\n".join(m.content for m in model.requests[0].messages if m.role is Role.SYSTEM)
            for text in ("Global rule.", "Project rule.", "Claude rule.", "Project video skill."):
                self.assertIn(text, system)
            self.assertLess(system.index("Global rule."), system.index("Project rule."))
            tool_result = model.requests[1].messages[-1].content
            self.assertIn("from the project", tool_result)

    async def test_claude_code_compatibility_can_be_turned_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            layout.config.write_text(GLOBAL_CONFIG.replace('root = "cwd"', 'root = "cwd"\nclaude_code = false'))
            (layout.project / "CLAUDE.md").write_text("Claude rule.")
            _skill(layout.home / ".claude" / "skills", "claude-user", "User Claude skill.")
            with mock.patch.dict(os.environ, layout.env):
                async with build_application(config_path=layout.config, database_path=":memory:",
                                             workspace=layout.project) as app:
                    self.assertEqual(sorted(app.skills), ["global-skill"])
                    context = app.orchestrator._system_context()
            self.assertNotIn("Claude rule.", context.text)

    async def test_claude_skills_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            layout.config.write_text(GLOBAL_CONFIG.replace(
                'root = "cwd"', 'root = "cwd"\nclaude_skills = ["hyperframes*", "aisystems-video"]'))
            for name in ("hyperframes", "hyperframes-audio", "aisystems-video", "browse"):
                _skill(layout.home / ".claude" / "skills", name, f"{name} skill.")
            with mock.patch.dict(os.environ, layout.env):
                async with build_application(config_path=layout.config, database_path=":memory:",
                                             workspace=layout.project) as app:
                    self.assertEqual(sorted(app.skills), ["aisystems-video", "global-skill", "hyperframes",
                                                          "hyperframes-audio"])


class _Model:
    """Reads hello.txt, then answers."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "_Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            call = ToolCall(id="c1", name="read_file", arguments={"path": "hello.txt"})
            return ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS)
        return ModelResponse(content="done", finish_reason=FinishReason.STOP)


class CliTests(unittest.TestCase):
    def _main(self, argv: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        previous = os.getcwd()
        os.chdir(cwd)
        try:
            with mock.patch.dict(os.environ, env), redirect_stdout(out), redirect_stderr(err):
                code = cli.main(argv)
        finally:
            os.chdir(previous)
        return code, out.getvalue(), err.getvalue()

    def test_init_project_and_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            repo = Path(directory) / "repo"
            repo.mkdir()
            (repo / "models.toml").write_text(GLOBAL_CONFIG)
            layout.config.unlink()
            code, out, _ = self._main(["init", "--global"], repo, layout.env)
            self.assertEqual(code, 0, out)
            link = layout.global_dir / "models.toml"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (repo / "models.toml").resolve())
            self.assertEqual(self._main(["init", "--global"], repo, layout.env)[0], 0)  # idempotent
            code, out, _ = self._main(["init"], layout.project, layout.env)
            self.assertEqual(code, 0)
            settings = layout.project / ".agent-harness" / "settings.toml"
            self.assertIn("allowed_commands", settings.read_text())
            code, _, err = self._main(["init"], layout.project, layout.env)
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)
            # From the project, config show finds the linked config, its folder, and the project.
            code, out, _ = self._main(["config", "show"], layout.project / "sub", layout.env)
            self.assertEqual(code, 0)
            self.assertIn(f"# project: {layout.project.resolve()}", out)
            self.assertIn("workspace=", out)

    def test_workspace_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = Layout(Path(directory))
            layout.config.write_text(GLOBAL_CONFIG.replace('root = "cwd"', 'root = "config"'))
            code, out, _ = self._main(["skills", "--config", str(layout.config), "--workspace",
                                       str(layout.project)], Path(directory), layout.env)
            self.assertEqual(code, 0)
            self.assertIn("global-skill", out)


if __name__ == "__main__":
    unittest.main()
