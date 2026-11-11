"""Phase 30: fixes are the code default; features are switched on by the root config.

The rule (CLAUDE.md, "Fix or feature"): a change is a *fix* when nobody would knowingly
choose the old behaviour. A fix is on in code, so the harness is correct with any config
or none, and it has no key in the root ``models.toml``. A *feature* needs something
outside the harness, costs extra model calls, or changes a stored format; it is off in
code and the root config turns it on.

``FIXES`` and ``FEATURES`` below are the register. A new option must be added to one of
them, or ``test_every_new_option_is_classified`` fails.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
import unittest
from pathlib import Path
from typing import Any

from agent_harness.config.loader import (
    ContextConfig,
    OrchestratorConfig,
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

# (config section, key, the value that is the fix). On in code; not set in the root config.
FIXES: tuple[tuple[str, str, Any], ...] = (
    ("orchestrator", "resurface_repeated_results", True),  # a repeated read was refused
    ("orchestrator", "phase_dispatch", True),              # the loop guessed its state
    ("orchestrator", "close_orphan_tool_calls", True),     # a budget stop left calls unanswered
    ("orchestrator", "mid_phase_checks", True),            # cancel and steer waited for every call
    ("context", "layout", "anchored"),                     # the first request was rewritten
    ("context", "recovery_hints", True),                   # shortened output did not say how to get the rest
    ("context", "ledger", True),                           # the model forgot what it had read
    ("context", "calibrate", True),                        # the estimate drifted below the real count
    ("summarization", "trigger", "threshold"),             # history was dropped before it was summarized
    ("summarization", "chunked", True),                    # a long range lost its middle
)

# (config section, key): why it is a feature or a setting, so off (or neutral) in code.
FEATURES: dict[tuple[str, str], str] = {
    ("orchestrator", "verify_phase"): "test aid: raises instead of continuing",
    ("orchestrator", "max_tool_calls_per_run"): "a limit only the operator can choose",
    ("orchestrator", "cancel_poll_seconds"): "depends on deployment: queue workers already poll",
    ("orchestrator", "state_block"): "adds a message to every request",
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
}

SECTIONS = {"orchestrator": OrchestratorConfig, "context": ContextConfig, "summarization": SummarizationConfig}

# Options that existed before Phase 25 keep the older rule and are not classified here.
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
}


def default_of(cls: type, name: str) -> Any:
    return {f.name: f.default for f in dataclasses.fields(cls)}[name]


class RegisterTests(unittest.TestCase):
    def test_fixes_are_on_in_code(self) -> None:
        for section, key, value in FIXES:
            with self.subTest(option=f"{section}.{key}"):
                self.assertEqual(default_of(SECTIONS[section], key), value)

    def test_runtime_defaults_match_the_config_defaults(self) -> None:
        # Code that builds an orchestrator or a builder without a config gets the fixes too.
        for section, key, value in FIXES:
            runtime = {"orchestrator": OrchestratorSettings, "context": ContextPolicy}.get(section)
            if runtime is None or key not in {f.name for f in dataclasses.fields(runtime)}:
                continue
            with self.subTest(option=f"{section}.{key}"):
                self.assertEqual(default_of(runtime, key), value)

    def test_the_root_config_does_not_set_a_fix(self) -> None:
        # A fix needs no key: a key there would suggest it is optional.
        root = tomllib.loads((ROOT / "models.toml").read_text(encoding="utf-8"))
        for section, key, _ in FIXES:
            with self.subTest(option=f"{section}.{key}"):
                self.assertNotIn(key, root.get(section, {}))

    def test_every_new_option_is_classified(self) -> None:
        fixes = {(section, key) for section, key, _ in FIXES}
        self.assertFalse(fixes & set(FEATURES), "an option cannot be both a fix and a feature")
        for section, cls in SECTIONS.items():
            for field in dataclasses.fields(cls):
                if field.name in BEFORE_PHASE_25[section]:
                    continue
                with self.subTest(option=f"{section}.{field.name}"):
                    self.assertIn((section, field.name), fixes | set(FEATURES),
                                  "add it to FIXES (on in code) or FEATURES (off in code, with the reason)")

    def test_a_config_with_only_a_profile_gets_every_fix(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n',
                            encoding="utf-8")
            config = load_config(path)
        for section, key, value in FIXES:
            with self.subTest(option=f"{section}.{key}"):
                self.assertEqual(getattr(getattr(config, section), key), value)


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
