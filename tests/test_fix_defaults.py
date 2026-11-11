"""Phases 30-31: a fix replaces the behaviour; only features and settings are options.

The rule (CLAUDE.md, "Fix or feature"): a change is a *fix* when nobody would knowingly
choose the old behaviour. A fix has **no switch**: the old code path is deleted and the
tests that described it are rewritten. Only a *feature* (it needs something the code
cannot supply, or it is a mode the user picks) or a *setting* (a value that depends on
the model or the workload) is an option, and every option says here why it is one.

``REMOVED`` lists the switches Phases 25-29 added for fixes and Phase 31 deleted; they
must not come back. ``OPTIONS`` is the register of what is allowed to be configurable:
a new field in one of these config sections that is in neither list fails the suite.
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
import tomllib
import unittest
from pathlib import Path

from agent_harness.config.loader import (
    ConfigError,
    ContextConfig,
    MemoryConfig,
    OrchestratorConfig,
    StorageConfig,
    SummarizationConfig,
    load_config,
)
from agent_harness.domain import Message, Role, ToolCall
from agent_harness.orchestration.context import (
    STATE_HEADER,
    BudgetedContextBuilder,
    ContextLimits,
    ContextPolicy,
    ContextWindowExceededError,
    fit_to_tokens,
)
from agent_harness.orchestration.orchestrator import OrchestratorSettings

ROOT = Path(__file__).resolve().parents[1]

# Switches that guarded a fix. The behaviour is now unconditional.
REMOVED: dict[tuple[str, str], str] = {
    ("orchestrator", "resurface_repeated_results"): "a repeated read gets the earlier result",
    ("orchestrator", "phase_dispatch"): "the loop uses the stored phase",
    ("orchestrator", "close_orphan_tool_calls"): "a budget stop answers the calls it did not run",
    ("orchestrator", "mid_phase_checks"): "cancel, steer and budgets are checked between tool calls",
    ("context", "layout"): "one layout: the first request is sent as stored",
    ("context", "recovery_hints"): "a truncation marker says how to get the rest",
    ("context", "ledger"): "the calls no longer shown are listed",
    ("context", "calibrate"): "the estimate follows the provider's count",
    ("summarization", "trigger"): "history is summarized before it is dropped",
    ("summarization", "chunked"): "a long range is summarized in parts",
    ("memory", "scopes"): "notes are scoped to the user and the project",
    ("memory", "guidance"): "the system prompt explains the memory tools",
    ("storage", "message_table"): "a checkpoint writes only the messages that changed",
    ("storage", "off_loop"): "database work runs off the event loop",
    ("orchestrator", "state_block"): "each request carries the current plan, tasks and low budget",
}

# Options added since Phase 25, each with the reason it is a feature or a setting.
OPTIONS: dict[tuple[str, str], str] = {
    ("orchestrator", "verify_phase"): "test aid: raises instead of continuing",
    ("orchestrator", "max_tool_calls_per_run"): "a limit only the operator can choose",
    ("orchestrator", "cancel_poll_seconds"): "depends on deployment: queue workers already poll",
    ("context", "default_window"): "depends on the model served",
    ("context", "stale_tool_output_chars"): "tuning value",
    ("context", "keep_recent_units"): "tuning value",
    ("context", "min_fitted_output_chars"): "tuning value",
    ("context", "ledger_max_entries"): "tuning value",
    ("context", "ledger_max_chars"): "tuning value",
    ("summarization", "trigger_ratio"): "tuning value",
    ("summarization", "keep_ratio"): "tuning value",
    ("summarization", "style"): "a choice of summary format",
    ("summarization", "max_message_chars"): "tuning value",
    ("memory", "recall"): "on; off for a benchmark that must not see earlier notes",
    ("memory", "recall_limit"): "tuning value",
    ("memory", "recall_max_chars"): "tuning value",
    ("memory", "auto_save"): "on; off for a benchmark that must not write notes",
    ("memory", "auto_save_profile"): "a cheaper model for note extraction is the operator's choice",
    ("memory", "auto_save_max_notes"): "tuning value",
    ("memory", "auto_save_max_output_tokens"): "tuning value",
    ("memory", "auto_save_parameters"): "depends on the model served (e.g. thinking off)",
}

SECTIONS = {"orchestrator": OrchestratorConfig, "context": ContextConfig, "summarization": SummarizationConfig,
            "memory": MemoryConfig, "storage": StorageConfig}

# Options from before Phase 25 are not classified here.
BEFORE_PHASE_25 = {
    "orchestrator": {
        "max_model_attempts", "max_tool_attempts", "retry_base_delay_seconds", "max_identical_tool_calls",
        "max_refused_repeats", "max_tool_calls_per_turn", "max_output_rejections", "context_overflow_retries",
        "context_overflow_shrink", "context_overflow_patterns", "quota_check_seconds", "current_date_note",
        "final_answer_on_last_step", "final_answer_reserve_seconds", "max_answer_continuations",
        "max_cut_off_retries", "prefetch_with_hooks", "reserve_tokens_per_run", "turn_max_steps",
        "turn_max_wall_seconds",
    },
    "context": {"chars_per_token", "high_water_ratio", "low_water_ratio", "message_overhead_tokens",
                "max_tool_output_chars"},
    "summarization": {"enabled", "profile", "max_output_tokens", "max_input_chars", "parameters"},
    "memory": {"enabled", "namespace", "database", "backend", "url_env", "search", "embedding_profile", "vectors"},
    "storage": {"url_env"},
}

MINIMAL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'


def names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


class RegisterTests(unittest.TestCase):
    def test_a_fix_has_no_switch(self) -> None:
        runtime = {"orchestrator": OrchestratorSettings, "context": ContextPolicy}
        for (section, key), behaviour in REMOVED.items():
            with self.subTest(option=f"{section}.{key}", now=behaviour):
                self.assertNotIn(key, names(SECTIONS[section]))
                if section in runtime:
                    self.assertNotIn(key, names(runtime[section]))

    def test_the_root_config_does_not_name_a_removed_switch(self) -> None:
        root = tomllib.loads((ROOT / "models.toml").read_text(encoding="utf-8"))
        for section, key in REMOVED:
            with self.subTest(option=f"{section}.{key}"):
                self.assertNotIn(key, root.get(section, {}))

    def test_a_removed_switch_in_a_config_is_an_error(self) -> None:
        # Not ignored silently: a config that still tries to switch a fix off says so.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(MINIMAL + '[context]\nlayout = "legacy"\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_every_new_option_is_classified(self) -> None:
        for section, cls in SECTIONS.items():
            for name in names(cls) - BEFORE_PHASE_25[section]:
                with self.subTest(option=f"{section}.{name}"):
                    self.assertIn((section, name), OPTIONS,
                                  "a fix needs no option; a feature or setting goes in OPTIONS with its reason")

    def test_memory_and_recall_work_without_being_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text(MINIMAL, encoding="utf-8")
            config = load_config(path)
        self.assertTrue(config.memory.enabled and config.memory.recall and config.memory.auto_save)


def turn(index: int, text: str) -> list[Message]:
    call = ToolCall(id=f"c{index}", name="echo", arguments={"text": text})
    return [Message(role=Role.ASSISTANT, content="", tool_calls=(call,)),
            Message(role=Role.TOOL, content=text, tool_call_id=call.id, name="echo")]


class TightBudgetTests(unittest.TestCase):
    """The state message must never cost more than the turns it stands for."""

    def test_overflow_retry_can_shrink_a_small_history(self) -> None:
        # Small turns: the full state message (header, note, ledger) is larger than one of
        # them, so the layout falls back to shorter forms instead of failing the retry.
        messages = [Message(role=Role.USER, content="Echo two words, then give the status in one line."),
                    *turn(0, "alpha " * 10), *turn(1, "beta " * 10)]
        policy = ContextPolicy()
        full = BudgetedContextBuilder(lambda _: ContextLimits(1_000_000, 0), policy=policy).build(
            messages, (), model_profile="p")
        self.assertEqual(full.omitted_messages, 0)
        cap, fitted = math.floor(full.estimated_tokens * 0.5), None
        while cap < full.estimated_tokens and fitted is None:
            try:
                fitted = fit_to_tokens(full, (), max_input_tokens=cap, model_profile="p", policy=policy)
            except ContextWindowExceededError:
                cap += 1
        assert fitted is not None, "no cap below the full size fits: an overflow could not be retried"
        self.assertEqual(fitted.omitted_messages, 2)
        self.assertEqual(fitted.messages[0], messages[0])  # the request is sent as stored
        note = fitted.messages[1].content
        self.assertIn("2 earlier messages", note)  # never dropped silently
        self.assertLessEqual(fitted.estimated_tokens, cap)

    def test_with_room_the_full_state_message_is_sent(self) -> None:
        messages = [Message(role=Role.USER, content="Read the files."),
                    *[m for i in range(12) for m in turn(i, "line of output " * 40)]]
        view = BudgetedContextBuilder(lambda _: ContextLimits(2_000, 200), policy=ContextPolicy()).build(
            messages, (), model_profile="p")
        self.assertGreater(view.omitted_messages, 0)
        state = view.messages[1].content
        self.assertTrue(state.startswith(STATE_HEADER))
        self.assertIn("were omitted to fit", state)
        self.assertIn("echo", state)  # the ledger of the calls no longer shown


if __name__ == "__main__":
    unittest.main()
