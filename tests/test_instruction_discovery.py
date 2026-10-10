"""Instruction files discovered in subdirectories (user -> project -> subdirectory)."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harness.application import build_application
from agent_harness.config.loader import ConfigError, load_config
from agent_harness.extensions.instructions import (
    SystemContextBuilder,
    discover_subdirectory_instructions,
    validate_instruction_filenames,
)

MODEL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/user-settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise-settings.toml",
}


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class DiscoveryTests(unittest.TestCase):
    def test_outermost_first_and_root_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "AGENTS.md", "root")
            write(root / "services" / "AGENTS.md", "services")
            write(root / "services" / "api" / "AGENTS.md", "api")
            write(root / "services" / "api" / "NOTES.md", "notes")
            write(root / "other" / "AGENTS.md", "sibling")
            found = discover_subdirectory_instructions(
                root, root / "services" / "api", ["AGENTS.md", "NOTES.md"]
            )
            at_root = discover_subdirectory_instructions(root, root, ["AGENTS.md"])
        self.assertEqual(
            found, ["services/AGENTS.md", "services/api/AGENTS.md", "services/api/NOTES.md"]
        )
        self.assertEqual(at_root, [])

    def test_start_outside_root_finds_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            write(Path(b) / "AGENTS.md", "elsewhere")
            self.assertEqual(discover_subdirectory_instructions(Path(a), Path(b), ["AGENTS.md"]), [])

    def test_symlink_escaping_the_root_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            root = Path(a)
            outside = write(Path(b) / "secret.md", "outside")
            (root / "sub").mkdir()
            try:
                os.symlink(outside, root / "sub" / "AGENTS.md")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            self.assertEqual(discover_subdirectory_instructions(root, root / "sub", ["AGENTS.md"]), [])

    def test_filenames_must_be_plain_names(self) -> None:
        for bad in ["", ".", "..", "../AGENTS.md", "a/b.md", "a\\b.md"]:
            with self.subTest(name=bad), self.assertRaises(ValueError):
                validate_instruction_filenames([bad])
        self.assertEqual(validate_instruction_filenames(["AGENTS.md"]), ("AGENTS.md",))


class SystemContextTests(unittest.TestCase):
    def test_subdirectory_files_follow_configured_files_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "AGENTS.md", "Project rule.")
            write(root / "pkg" / "AGENTS.md", "Package rule.")
            write(root / "pkg" / "mod" / "AGENTS.md", "Module rule.")
            builder = SystemContextBuilder(
                instruction_files=["AGENTS.md", "pkg/AGENTS.md"],
                base=root,
                subdirectory_filenames=("AGENTS.md",),
                working_directory=lambda: root / "pkg" / "mod",
            )
            context = builder()
            disabled = SystemContextBuilder(
                instruction_files=["AGENTS.md"],
                base=root,
                working_directory=lambda: root / "pkg" / "mod",
            )()
        text = context.text
        self.assertEqual(text.count("Package rule."), 1)
        self.assertLess(text.index("Project rule."), text.index("Package rule."))
        self.assertLess(text.index("Package rule."), text.index("Module rule."))
        self.assertIn("## Instructions from pkg/mod/AGENTS.md", text)
        self.assertEqual(len(context.sources), 3)
        self.assertNotIn("Module rule.", disabled.text)


class HeaderTests(unittest.TestCase):
    def test_configured_files_are_named_without_their_directory(self) -> None:
        # A configured path such as .agent-harness/instructions/CHAT.md made Qwen 3.6 read
        # "explain how agent-harness works" as a question about the .agent-harness folder.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / ".agent-harness" / "instructions" / "CHAT.md", "Answer concisely.")
            write(root / "pkg" / "AGENTS.md", "Package rule.")
            context = SystemContextBuilder(
                instruction_files=[".agent-harness/instructions/CHAT.md"],
                base=root,
                subdirectory_filenames=("AGENTS.md",),
                working_directory=lambda: root / "pkg",
            )()
        self.assertIn("## Instructions from CHAT.md\n\nAnswer concisely.", context.text)
        self.assertNotIn(".agent-harness", context.text)
        # A discovered subdirectory file keeps its path: its rules apply to that directory.
        self.assertIn("## Instructions from pkg/AGENTS.md", context.text)
        self.assertEqual(len(context.sources), 2)  # sources still record the full paths


class ConfigTests(unittest.TestCase):
    def test_defaults_parse_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = load_config(write(root / "a.toml", MODEL))
            parsed = load_config(write(
                root / "b.toml",
                MODEL + '[instructions]\nenabled = true\nsubdirectories = true\nfilenames = ["AGENTS.md", "AGENT.md"]\n',
            ))
            bad = write(
                root / "c.toml",
                MODEL + '[instructions]\nenabled = true\nsubdirectories = true\nfilenames = ["../x.md"]\n',
            )
            with mock.patch.dict(os.environ, HERMETIC), self.assertRaises(ConfigError):
                build_application(config_path=bad, database_path=root / "runs.sqlite3")
        self.assertFalse(default.instructions.subdirectories)
        self.assertEqual(default.instructions.filenames, ("AGENTS.md",))
        self.assertTrue(parsed.instructions.subdirectories)
        self.assertEqual(parsed.instructions.filenames, ("AGENTS.md", "AGENT.md"))

    def test_application_loads_subdirectory_instructions_from_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "AGENTS.md", "Project rule.")
            write(root / "app" / "AGENTS.md", "App rule.")
            config = write(
                root / "models.toml",
                MODEL + '[instructions]\nenabled = true\nfiles = ["AGENTS.md"]\nsubdirectories = true\n',
            )
            with mock.patch.dict(os.environ, HERMETIC):
                application = build_application(config_path=config, database_path=root / "runs.sqlite3")
            builder = application.orchestrator._system_context
            assert builder is not None
            with mock.patch("pathlib.Path.cwd", return_value=root / "app"):
                text = builder().text
            asyncio.run(application.close())
        self.assertIn("Project rule.", text)
        self.assertIn("App rule.", text)


if __name__ == "__main__":
    unittest.main()
