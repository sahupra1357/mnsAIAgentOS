from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent_harness.config import ConfigError, HarnessConfig, ModelProfile, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_named_profiles_and_default_parameters_from_toml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "harness.toml")
            path.write_text(
                """
default_model_profile = "local"

[models.local]
adapter = "openai_compatible"
model = "qwen-test"
endpoint = "http://127.0.0.1:8000/v1"
timeout_seconds = 12.5
max_retries = 1

[models.local.parameters]
top_p = 0.8

[models.hosted]
adapter = "openai_compatible"
model = "hosted-test"
endpoint = "https://models.invalid/v1"
api_key_env = "HOSTED_API_KEY"
""",
                encoding="utf-8",
            )

            config = load_config(path, environ={})

        self.assertEqual(config.default_model_profile, "local")
        self.assertEqual(config.model_profile().model, "qwen-test")
        self.assertEqual(config.models["local"].parameters["top_p"], 0.8)
        self.assertEqual(config.models["hosted"].api_key_env, "HOSTED_API_KEY")

    def test_environment_can_define_profile_without_file(self) -> None:
        config = load_config(
            environ={
                "AGENT_HARNESS_MODEL_PROFILE": "fast",
                "AGENT_HARNESS_MODEL__FAST__ADAPTER": "openai_compatible",
                "AGENT_HARNESS_MODEL__FAST__MODEL": "small-model",
                "AGENT_HARNESS_MODEL__FAST__ENDPOINT": "http://localhost:9000/v1",
                "AGENT_HARNESS_MODEL__FAST__MAX_RETRIES": "3",
                "AGENT_HARNESS_MODEL__FAST__PARAMETERS": '{"top_p": 0.9}',
            }
        )

        self.assertEqual(config.model_profile("fast").model, "small-model")
        self.assertEqual(config.model_profile("fast").max_retries, 3)
        self.assertEqual(config.model_profile("fast").parameters, {"top_p": 0.9})

    def test_environment_overrides_toml_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "harness.toml")
            path.write_text(
                """
default_model_profile = "primary"
[models.primary]
adapter = "openai_compatible"
model = "old-model"
endpoint = "http://localhost/v1"
""",
                encoding="utf-8",
            )
            config = load_config(
                path,
                environ={"AGENT_HARNESS_MODEL__PRIMARY__MODEL": "new-model"},
            )

        self.assertEqual(config.model_profile().model, "new-model")

    def test_single_profile_is_implicit_default(self) -> None:
        config = load_config(
            environ={
                "AGENT_HARNESS_MODEL__ONLY__ADAPTER": "openai_compatible",
                "AGENT_HARNESS_MODEL__ONLY__MODEL": "model",
                "AGENT_HARNESS_MODEL__ONLY__ENDPOINT": "http://localhost/v1",
            }
        )
        self.assertEqual(config.default_model_profile, "only")

    def test_rejects_unknown_default_profile(self) -> None:
        profile = ModelProfile(
            adapter="openai_compatible", model="model", endpoint="http://localhost/v1"
        )
        with self.assertRaisesRegex(ConfigError, "not configured"):
            HarnessConfig(models={"known": profile}, default_model_profile="missing")

    def test_rejects_invalid_environment_json(self) -> None:
        with self.assertRaisesRegex(ConfigError, "PARAMETERS"):
            load_config(
                environ={
                    "AGENT_HARNESS_MODEL__LOCAL__PARAMETERS": "[]",
                }
            )

    def test_configuration_does_not_resolve_secret(self) -> None:
        config = load_config(
            environ={
                "AGENT_HARNESS_MODEL__HOSTED__ADAPTER": "openai_compatible",
                "AGENT_HARNESS_MODEL__HOSTED__MODEL": "model",
                "AGENT_HARNESS_MODEL__HOSTED__ENDPOINT": "https://models.invalid/v1",
                "AGENT_HARNESS_MODEL__HOSTED__API_KEY_ENV": "A_SECRET_NAME",
                "A_SECRET_NAME": "must-not-appear-in-config",
            }
        )
        profile = config.model_profile()
        self.assertEqual(profile.api_key_env, "A_SECRET_NAME")
        self.assertNotIn("must-not-appear", repr(config))

    def test_loads_coding_sandbox_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "harness.toml")
            path.write_text(
                """
[models.local]
adapter = "openai_compatible"
model = "model"
endpoint = "http://localhost/v1"

[coding]
enabled = true
workspace = "."
execution_runtime = "docker"
approval_mode = "side_effects"
command_timeout_seconds = 45

[coding.docker]
image = "python:3.12-slim"
network_enabled = false
memory = "512m"
""",
                encoding="utf-8",
            )
            config = load_config(path, environ={})

        self.assertTrue(config.coding.enabled)
        self.assertEqual(config.coding.execution_runtime, "docker")
        self.assertEqual(config.coding.docker.memory, "512m")
        self.assertEqual(config.coding.command_timeout_seconds, 45)

    def test_local_coding_execution_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "harness.toml")
            path.write_text(
                """
[models.local]
adapter = "openai_compatible"
model = "model"
endpoint = "http://localhost/v1"

[coding]
enabled = true
execution_runtime = "local"
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "allow_local_execution"):
                load_config(path, environ={})


if __name__ == "__main__":
    unittest.main()
