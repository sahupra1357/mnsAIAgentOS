"""The shipped example subagent configuration stays valid and offers the agent tool."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harness.application import build_application
from agent_harness.domain import Message, Role

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "subagents.toml"
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/user-settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise-settings.toml",
}


class ExampleSubagentsTests(unittest.TestCase):
    def test_example_defines_three_agents_and_top_level_only_nesting(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, HERMETIC):
            app = build_application(config_path=EXAMPLE, database_path=Path(directory) / "runs.sqlite3")

            async def offered() -> tuple[list[str], list[str]]:
                async with app:
                    top = await app.orchestrator.create_run(
                        model_profile="ollama", messages=[Message(Role.USER, "hi")]
                    )
                    child = await app.orchestrator.create_run(
                        model_profile="ollama", messages=[Message(Role.USER, "hi")], depth=1,
                        parent_run_id=top.id,
                    )
                    return (
                        [s.name for s in app.orchestrator._specs_for(top)],
                        [s.name for s in app.orchestrator._specs_for(child)],
                    )

            top_tools, child_tools = asyncio.run(offered())
            spec = app.orchestrator._agent_spec
        self.assertIn("agent", top_tools)
        self.assertNotIn("agent", child_tools)  # max_depth = 1 in the example
        assert spec is not None
        self.assertEqual(spec.input_schema["properties"]["agent"]["enum"],
                         ["researcher", "reviewer", "test-runner"])
        for name in ("researcher", "reviewer", "test-runner"):
            self.assertTrue((EXAMPLE.parent / "agents" / f"{name}.md").is_file())


if __name__ == "__main__":
    unittest.main()
