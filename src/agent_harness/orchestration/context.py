"""Context-window management: bounded, deterministic, prefix-cache-friendly views.

``BudgetedContextBuilder`` never mutates stored messages. It groups history into
units that must stay together (an assistant tool-call message plus its tool
results), then replays the history unit by unit with hysteresis: when the
estimate crosses a high-water mark, the oldest units are omitted until it falls
to a low-water mark. Between trims every view extends the previous one, so
self-hosted servers keep reusing their KV cache, and because the replay depends
only on the messages, a resumed run rebuilds exactly the same view.

With a persisted ``ContextSummary`` the view is the pinned head (with the
summary appended to the first user message) followed by the history after
``summary.covers``; the same hysteresis then applies to that remainder.
``ContextSummarizer`` builds the model request that writes such a summary; the
orchestrator makes the call and persists the result.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from agent_harness.domain import (
    ContextSummary,
    ContextView,
    ContextWindowExceededError,
    Message,
    ModelRequest,
    Role,
    ToolSpec,
)


# Images are counted as a fixed number of tokens (vision encoders vary: ~250-1500).
IMAGE_TOKENS = 1000


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Token limits for one model profile."""

    context_window: int
    output_reserve_tokens: int = 1024
    # Whether assistant reasoning is sent back (reasoning_history = "keep"), so it
    # must be counted against the window.
    counts_reasoning: bool = False

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window must be greater than zero")
        if self.output_reserve_tokens < 0:
            raise ValueError("output_reserve_tokens must not be negative")


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Estimation and trimming parameters.

    The character-based estimate is deliberately conservative (3.5 characters
    per token undercounts characters for most tokenizers) and is not calibrated
    from reported usage: calibration would change past trim decisions and break
    both resume determinism and prefix stability.
    """

    high_water_ratio: float = 0.85
    low_water_ratio: float = 0.60
    chars_per_token: float = 3.5
    message_overhead_tokens: int = 8
    max_tool_output_chars: int | None = 16_000

    def __post_init__(self) -> None:
        if not 0 < self.low_water_ratio < self.high_water_ratio <= 1:
            raise ValueError("require 0 < low_water_ratio < high_water_ratio <= 1")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be greater than zero")
        if self.message_overhead_tokens < 0:
            raise ValueError("message_overhead_tokens must not be negative")
        if self.max_tool_output_chars is not None and self.max_tool_output_chars < 200:
            raise ValueError("max_tool_output_chars must be at least 200")


class FullHistoryContextBuilder:
    """Send the complete history unchanged (the default)."""

    def build(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        model_profile: str,
        summary: ContextSummary | None = None,
    ) -> ContextView:
        # The full history is still stored, so a summary is never needed here.
        return ContextView(messages=tuple(messages))


LimitsLookup = Callable[[str], ContextLimits | None]


class BudgetedContextBuilder:
    """Fit history into each profile's context window.

    Profiles without limits get the full, unmodified history.
    """

    def __init__(
        self,
        limits: Mapping[str, ContextLimits] | LimitsLookup,
        *,
        policy: ContextPolicy | None = None,
    ) -> None:
        if callable(limits):
            self._lookup: LimitsLookup = limits
        else:
            table = dict(limits)
            self._lookup = table.get
        self._policy = policy or ContextPolicy()

    def build(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        model_profile: str,
        summary: ContextSummary | None = None,
    ) -> ContextView:
        limits = self._lookup(model_profile)
        if limits is None:
            # Without a window the full history fits by definition; no summary needed.
            return ContextView(messages=tuple(messages))

        prepared: list[Message] = []
        trimmed_outputs = 0
        for message in messages:
            shortened = self._trim_tool_output(message)
            trimmed_outputs += shortened is not message
            prepared.append(shortened)

        head_end = _pinned_head_end(prepared)
        head = prepared[:head_end]
        if summary is not None and (not head or not _summary_applies(summary, prepared, head_end)):
            summary = None  # e.g. the history was rewound behind it
        rest_start = summary.covers if summary is not None else head_end
        units = _group_units(prepared[rest_start:])
        summarized = rest_start - head_end

        budget = limits.context_window - limits.output_reserve_tokens - self._tools_tokens(tools)
        if summary is not None:
            block = f"\n\n[Summary of {summarized} earlier messages]\n{summary.content.strip()}"
            head[-1] = replace(head[-1], content=head[-1].content + block)
        reasoning = limits.counts_reasoning
        head_tokens = sum(self._message_tokens(message, reasoning) for message in head)
        if budget <= 0 or head_tokens > budget:
            raise ContextWindowExceededError(
                f"context window {limits.context_window} cannot hold the output reserve, "
                f"tool definitions, and pinned instructions{' and summary' if summary else ''} "
                f"for profile {model_profile!r}"
            )
        high_water = math.floor(budget * self._policy.high_water_ratio)
        low_water = math.floor(budget * self._policy.low_water_ratio)

        unit_tokens = [sum(self._message_tokens(m, reasoning) for m in unit) for unit in units]
        total = head_tokens
        cut = 0
        for index, tokens in enumerate(unit_tokens):
            total += tokens
            if total > high_water:
                # Never omit the newest unit: the model must see the latest turn.
                while cut < index and total > low_water:
                    total -= unit_tokens[cut]
                    cut += 1

        omitted = sum(len(unit) for unit in units[:cut])
        if omitted and head:
            notice = (
                f"\n\n[Context note: {omitted} earlier messages were omitted "
                "to fit the model's context window.]"
            )
            head[-1] = replace(head[-1], content=head[-1].content + notice)
            total += self._text_tokens(notice)
        if total > budget:
            raise ContextWindowExceededError(
                f"estimated {total} tokens exceed the {budget}-token input budget for "
                f"profile {model_profile!r} even after omitting older turns"
            )

        kept = [message for unit in units[cut:] for message in unit]
        return ContextView(
            messages=tuple(head + kept),
            estimated_tokens=total,
            omitted_messages=omitted,
            trimmed_tool_outputs=trimmed_outputs,
            summarized_messages=summarized,
            omitted_from=rest_start if omitted else None,
            omitted_through=rest_start + omitted if omitted else None,
        )

    def _trim_tool_output(self, message: Message) -> Message:
        limit = self._policy.max_tool_output_chars
        if limit is None or message.role is not Role.TOOL or len(message.content) <= limit:
            return message
        removed = len(message.content) - limit
        marker = f"\n[... {removed} characters truncated ...]\n"
        keep_head = limit // 2
        keep_tail = limit - keep_head
        content = message.content[:keep_head] + marker + message.content[-keep_tail:]
        return replace(message, content=content)

    def _text_tokens(self, text: str) -> int:
        return math.ceil(len(text) / self._policy.chars_per_token)

    def _message_tokens(self, message: Message, reasoning: bool = False) -> int:
        tokens = self._policy.message_overhead_tokens + self._text_tokens(message.content)
        tokens += IMAGE_TOKENS * len(message.parts)  # a rough, fixed allowance per image
        if reasoning and message.reasoning:
            tokens += self._text_tokens(message.reasoning)
        for call in message.tool_calls:
            arguments = json.dumps(call.arguments, separators=(",", ":"), sort_keys=True)
            tokens += self._text_tokens(call.name) + self._text_tokens(arguments)
        return tokens

    def _tools_tokens(self, tools: Sequence[ToolSpec]) -> int:
        if not tools:
            return 0
        serialized = json.dumps(
            [
                {"name": t.name, "description": t.description, "parameters": t.input_schema}
                for t in tools
            ],
            separators=(",", ":"),
            sort_keys=True,
        )
        return self._text_tokens(serialized)


def estimate_tokens(
    messages: Sequence[Message], tools: Sequence[ToolSpec], policy: ContextPolicy | None = None
) -> int:
    """The builder's token estimate for a request: messages plus tool definitions."""

    builder = BudgetedContextBuilder({}, policy=policy)
    return sum(builder._message_tokens(m, True) for m in messages) + builder._tools_tokens(tools)


def fit_to_tokens(
    view: ContextView,
    tools: Sequence[ToolSpec],
    *,
    max_input_tokens: int,
    model_profile: str,
    policy: ContextPolicy | None = None,
) -> ContextView:
    """Shrink a view to ``max_input_tokens`` (after a provider reported an overflow).

    The same unit-wise trimming applies, against a fixed cap, so later views stay
    append-only between trims. Tool outputs are also capped at a quarter of the cap.
    """

    selected = policy or ContextPolicy()
    quarter = math.floor(max_input_tokens * selected.chars_per_token / 4)
    current = selected.max_tool_output_chars
    tight = replace(
        selected, max_tool_output_chars=max(200, quarter if current is None else min(current, quarter))
    )
    limits = ContextLimits(context_window=max_input_tokens, output_reserve_tokens=0)
    fitted = BudgetedContextBuilder(lambda _: limits, policy=tight).build(
        view.messages, tools, model_profile=model_profile
    )
    return replace(
        fitted,
        omitted_messages=view.omitted_messages + fitted.omitted_messages,
        trimmed_tool_outputs=view.trimmed_tool_outputs + fitted.trimmed_tool_outputs,
        summarized_messages=view.summarized_messages,
        omitted_from=view.omitted_from,
        omitted_through=view.omitted_through,
    )


def _summary_applies(summary: ContextSummary, messages: Sequence[Message], head_end: int) -> bool:
    """A summary must end inside the history and at a unit boundary."""

    if not head_end <= summary.covers <= len(messages):
        return False
    return summary.covers == len(messages) or messages[summary.covers].role is not Role.TOOL


def _pinned_head_end(messages: Sequence[Message]) -> int:
    """Index after the leading system messages and the first user message."""

    for index, message in enumerate(messages):
        if message.role is Role.USER:
            return index + 1
        if message.role is not Role.SYSTEM:
            return index
    return len(messages)


def _group_units(messages: Sequence[Message]) -> list[list[Message]]:
    """Group messages so a tool-call message is never split from its results."""

    units: list[list[Message]] = []
    for message in messages:
        # Tool results never start a unit, so a cut can never orphan one.
        if message.role is Role.TOOL and units:
            units[-1].append(message)
        else:
            units.append([message])
    return units


def context_policy_from_config(config: object) -> ContextPolicy:
    """``[context]`` (``HarnessConfig.context``) as a policy; 0 tool-output characters = never shorten."""

    section = getattr(config, "context", None)
    if section is None:
        return ContextPolicy()
    return ContextPolicy(
        high_water_ratio=section.high_water_ratio,
        low_water_ratio=section.low_water_ratio,
        chars_per_token=section.chars_per_token,
        message_overhead_tokens=section.message_overhead_tokens,
        max_tool_output_chars=section.max_tool_output_chars or None,
    )


def limits_from_profiles(
    profiles: Mapping[str, object],
    *,
    default_output_reserve_tokens: int = 1024,
) -> dict[str, ContextLimits]:
    """Derive limits from model profiles that declare ``context_window``.

    The output reserve is the profile's ``parameters.max_tokens`` when set.
    """

    limits: dict[str, ContextLimits] = {}
    for name, profile in profiles.items():
        window = getattr(profile, "context_window", None)
        if window is None:
            continue
        parameters = getattr(profile, "parameters", {}) or {}
        reserve = parameters.get("max_tokens")
        if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve <= 0:
            reserve = default_output_reserve_tokens
        limits[name] = ContextLimits(
            context_window=window,
            output_reserve_tokens=reserve,
            counts_reasoning=getattr(profile, "reasoning_history", "drop") == "keep",
        )
    return limits


SUMMARY_INSTRUCTIONS = (
    "You compress the earlier part of an agent's conversation so the agent can keep "
    "working with a smaller context. Write a concise, factual summary that preserves: "
    "the user's goals and constraints, decisions made, facts and results learned from "
    "tools (file paths, identifiers, numbers, errors), work completed, and open "
    "questions or next steps. Merge the previous summary, if given, with the new "
    "messages into one summary. Do not invent anything, do not address the user, and "
    "do not call tools. Reply with the summary text only."
)


class ContextSummarizer:
    """Build the model request that summarizes omitted history.

    ``profile`` selects the summarizing model (None: the run's own profile).
    The transcript is bounded by ``input_chars(profile)`` characters (or
    ``max_input_chars`` when no lookup is given): each message is capped, and
    if the whole transcript is still too long its middle is dropped.
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        max_output_tokens: int = 1024,
        max_input_chars: int = 48_000,
        max_message_chars: int = 4_000,
        input_chars: Callable[[str], int | None] | None = None,
        instructions: str = SUMMARY_INSTRUCTIONS,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if max_input_chars < 1_000:
            raise ValueError("max_input_chars must be at least 1000")
        if max_message_chars < 200:
            raise ValueError("max_message_chars must be at least 200")
        self.profile = profile
        self.max_output_tokens = max_output_tokens
        self._max_input_chars = max_input_chars
        self._max_message_chars = max_message_chars
        self._input_chars = input_chars
        self._instructions = instructions

    def input_limit(self, profile: str) -> int:
        derived = self._input_chars(profile) if self._input_chars is not None else None
        if derived is None:
            return self._max_input_chars
        return max(1_000, min(self._max_input_chars, derived))

    def request(
        self,
        previous: str | None,
        messages: Sequence[Message],
        *,
        profile: str,
        metadata: Mapping[str, str] | None = None,
    ) -> ModelRequest:
        transcript = "\n\n".join(self._render(message) for message in messages)
        parts = []
        if previous:
            parts.append("Previous summary:\n" + previous.strip())
        parts.append("New messages:\n" + transcript)
        text = _shorten("\n\n".join(parts), self.input_limit(profile))
        return ModelRequest(
            messages=(
                Message(role=Role.SYSTEM, content=self._instructions),
                Message(role=Role.USER, content=text),
            ),
            max_output_tokens=self.max_output_tokens,
            metadata=dict(metadata or {}),
        )

    def _render(self, message: Message) -> str:
        label = message.role.value.upper()
        if message.role is Role.TOOL and message.name:
            label += f" ({message.name})"
        lines = [f"{label}: {_shorten(message.content, self._max_message_chars)}"]
        if message.parts:
            lines.append(f"  [{len(message.parts)} image(s) attached]")
        for call in message.tool_calls:
            arguments = json.dumps(call.arguments, separators=(",", ":"), sort_keys=True)
            lines.append(f"  -> {call.name}({_shorten(arguments, 500)})")
        return "\n".join(lines)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit} characters omitted ...]\n"
    keep_head = limit // 2
    return text[:keep_head] + marker + text[-(limit - keep_head):]


def summary_input_chars(
    limits: Mapping[str, ContextLimits], policy: ContextPolicy | None = None
) -> Callable[[str], int | None]:
    """Transcript size that fits a profile's window after the summary's own output."""

    selected = policy or ContextPolicy()

    def lookup(profile: str) -> int | None:
        found = limits.get(profile)
        if found is None:
            return None
        usable = found.context_window - found.output_reserve_tokens - 256
        return math.floor(usable * selected.chars_per_token * 0.75) if usable > 0 else None

    return lookup
