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
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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
    # The view's layout (Phase 26; the only one since Phase 31): the first user message is
    # never changed; the summary, the omission note and the ledger of tool calls no longer
    # shown are one message of their own after it, and the user's latest request, when the
    # summary or a trim covers it, follows as a message of its own. A truncation marker
    # says how to get the rest (the artifact, or calling again).
    # Before whole turns are dropped, tool outputs of turns older than the newest
    # ``keep_recent_units`` are shortened to this many characters (None: off).
    stale_tool_output_chars: int | None = None
    keep_recent_units: int = 4
    # The newest turn's tool outputs are never shortened below this.
    min_fitted_output_chars: int = 200
    ledger_max_entries: int = 40
    ledger_max_chars: int = 4_000

    def __post_init__(self) -> None:
        if self.stale_tool_output_chars is not None and self.stale_tool_output_chars < 200:
            raise ValueError("stale_tool_output_chars must be at least 200")
        if self.keep_recent_units < 1:
            raise ValueError("keep_recent_units must be at least 1")
        if self.min_fitted_output_chars < 200:
            raise ValueError("min_fitted_output_chars must be at least 200")
        if self.ledger_max_entries < 1 or self.ledger_max_chars < 200:
            raise ValueError("ledger_max_entries must be at least 1 and ledger_max_chars at least 200")
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

# The newest turn's tool outputs are never shortened below this (characters).
_MIN_FITTED_OUTPUT = 200


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
        max_input_tokens: int | None = None,
        scale: float = 1.0,
    ) -> ContextView:
        """``max_input_tokens`` (a cap a provider overflow set) and ``scale`` (how far the
        estimate is under the provider's count) are passed by the orchestrator to builders
        with ``native_caps`` (this one); other builders' views are capped afterwards with
        :func:`fit_to_tokens`."""

        limits = self._lookup(model_profile)
        if limits is None:
            # Without a window the full history fits by definition; no summary needed.
            return ContextView(messages=tuple(messages))
        if scale > 1.0:
            # Count every character as more tokens; the rest of the logic is unchanged.
            scaled = replace(self._policy, chars_per_token=self._policy.chars_per_token / scale)
            return BudgetedContextBuilder(self._lookup, policy=scaled).build(
                messages, tools, model_profile=model_profile, summary=summary,
                max_input_tokens=max_input_tokens)

        prepared: list[Message] = []
        trimmed_outputs = 0
        for message in messages:
            shortened = self._trim_tool_output(message)
            trimmed_outputs += shortened is not message
            prepared.append(shortened)

        return self._build_anchored(prepared, tools, limits, summary, trimmed_outputs,
                                    model_profile, max_input_tokens)

    @property
    def native_caps(self) -> bool:
        """Whether ``build`` takes ``max_input_tokens`` and ``scale`` itself."""

        return True

    @property
    def policy(self) -> ContextPolicy:
        return self._policy

    def _build_anchored(
        self,
        prepared: list[Message],
        tools: Sequence[ToolSpec],
        limits: ContextLimits,
        summary: ContextSummary | None,
        trimmed_outputs: int,
        model_profile: str,
        max_input_tokens: int | None,
    ) -> ContextView:
        """The Phase 26 layout: pinned head, one state message, the latest request, kept turns.

        The first user message is sent exactly as stored. What stands in for older turns
        (the summary, how many messages were omitted, and the ledger of their tool calls)
        is one message after the head. When the user's latest request lies in that older
        part, it follows as a message of its own, so the model answers it and not the
        first one. Trimming is the same replay with hysteresis; before whole turns are
        dropped, the tool outputs of older turns are shortened.
        """

        policy = self._policy
        head_end = _pinned_head_end(prepared)
        head = prepared[:head_end]
        if summary is not None and (not head or not _summary_applies(summary, prepared, head_end)):
            summary = None  # e.g. the history was rewound behind it
        rest_start = summary.covers if summary is not None else head_end
        units = _group_units(prepared[rest_start:])
        latest = latest_request_index(prepared, start=head_end)
        reasoning = limits.counts_reasoning

        tools_tokens = self._tools_tokens(tools)
        budget = limits.context_window - limits.output_reserve_tokens - tools_tokens
        if max_input_tokens is not None:
            budget = min(budget, max_input_tokens - tools_tokens)
        head_tokens = sum(self._message_tokens(message, reasoning) for message in head)
        if budget <= 0 or head_tokens > budget:
            raise ContextWindowExceededError(
                f"context window {limits.context_window} cannot hold the output reserve, "
                f"tool definitions, and pinned instructions for profile {model_profile!r}"
            )
        high_water = math.floor(budget * policy.high_water_ratio)
        low_water = math.floor(budget * policy.low_water_ratio)
        summary_text = ""
        if summary is not None:
            summary_text = (f"[Summary of {rest_start - head_end} earlier messages]\n"
                            f"{summary.content.strip()}")
        fixed = head_tokens + (self._text_tokens(summary_text) if summary_text else 0)

        unit_tokens = [sum(self._message_tokens(m, reasoning) for m in unit) for unit in units]
        total = fixed
        cut = 0
        shrunk = 0  # units[:shrunk] were considered for shortening
        stale = policy.stale_tool_output_chars
        for index, tokens in enumerate(unit_tokens):
            total += tokens
            if total <= high_water:
                continue
            if stale is not None:
                # First shorten what older turns printed; their calls stay visible.
                while shrunk <= index - policy.keep_recent_units and total > low_water:
                    if shrunk >= cut:
                        shorter = [self._trim_tool_output(m, stale) for m in units[shrunk]]
                        changed = sum(new is not old for new, old in zip(shorter, units[shrunk]))
                        if changed:
                            saved = unit_tokens[shrunk] - sum(
                                self._message_tokens(m, reasoning) for m in shorter)
                            units[shrunk] = shorter
                            unit_tokens[shrunk] -= saved
                            total -= saved
                            trimmed_outputs += changed
                    shrunk += 1
            # Never omit the newest unit: the model must see the latest turn.
            while cut < index and total > low_water:
                total -= unit_tokens[cut]
                cut += 1

        def state(cut: int, detail: int = 2) -> tuple[list[Message], int]:
            """The state message and the carried request for this cut, and their tokens.

            ``detail`` 2 is the full message; 1 leaves the ledger out; 0 also leaves the
            header out. The smaller forms are for a budget so tight that the state message
            would cost more than the turns it stands for (Phase 30).
            """

            kept_from = rest_start + sum(len(unit) for unit in units[:cut])
            parts = [summary_text] if summary_text else []
            omitted = kept_from - rest_start
            if omitted and detail:
                parts.append(f"[Context note: {omitted} earlier messages were omitted "
                             "to fit the model's context window.]")
            elif omitted:
                parts.append(f"[Context note: {omitted} earlier messages omitted.]")
            if detail >= 2 and kept_from > head_end:
                room = min(policy.ledger_max_chars,
                           max(200, math.floor(budget * 0.1 * policy.chars_per_token)))
                ledger = tool_ledger(prepared[head_end:kept_from],
                                     max_entries=policy.ledger_max_entries, max_chars=room)
                if ledger:
                    parts.append(ledger)
            extra: list[Message] = []
            if parts:
                body = "\n\n".join(parts)
                extra.append(Message(role=Role.USER, content=STATE_HEADER + "\n\n" + body if detail else body))
            if latest is not None and latest < kept_from:
                extra.append(prepared[latest])  # the request being worked on, in the user's words
            return extra, sum(self._message_tokens(m, reasoning) for m in extra)

        kept_tokens = sum(unit_tokens[cut:])
        extra, extra_tokens = state(cut)
        while fixed - (self._text_tokens(summary_text) if summary_text else 0) + extra_tokens \
                + kept_tokens > budget and cut < len(units) - 1:
            kept_tokens -= unit_tokens[cut]
            cut += 1
            extra, extra_tokens = state(cut)
        total = head_tokens + extra_tokens + kept_tokens
        for detail in (1, 0):
            if total <= budget:
                break
            extra, extra_tokens = state(cut, detail)
            total = head_tokens + extra_tokens + kept_tokens
        if total > budget and units:
            newest, fitted_outputs = self._fit_newest_unit(units[-1], total - budget, reasoning)
            if fitted_outputs:
                units[-1] = newest
                trimmed_outputs += fitted_outputs
                kept_tokens = sum(unit_tokens[cut:-1]) + sum(
                    self._message_tokens(m, reasoning) for m in newest)
                total = head_tokens + extra_tokens + kept_tokens
        if total > budget:
            raise ContextWindowExceededError(
                f"estimated {total} tokens exceed the {budget}-token input budget for "
                f"profile {model_profile!r} even after omitting older turns"
            )

        omitted = sum(len(unit) for unit in units[:cut])
        kept = [message for unit in units[cut:] for message in unit]
        return ContextView(
            messages=tuple(head + extra + kept),
            estimated_tokens=total,
            omitted_messages=omitted,
            trimmed_tool_outputs=trimmed_outputs,
            summarized_messages=rest_start - head_end,
            omitted_from=rest_start if omitted else None,
            omitted_through=rest_start + omitted if omitted else None,
            budget_tokens=budget,
        )

    def unit_tokens(self, unit: Sequence[Message], reasoning: bool = False) -> int:
        """The estimate for one unit as the view would show it (tool outputs capped)."""

        return sum(self._message_tokens(self._trim_tool_output(m), reasoning) for m in unit)

    def _fit_newest_unit(
        self, unit: list[Message], excess_tokens: int, reasoning: bool
    ) -> tuple[list[Message], int]:
        """Shorten the unit's tool outputs by about ``excess_tokens``; returns (unit, count)."""

        _MIN_FITTED_OUTPUT = self._policy.min_fitted_output_chars
        sizes = [len(m.content) if m.role is Role.TOOL else 0 for m in unit]
        available = sum(size - _MIN_FITTED_OUTPUT for size in sizes if size > _MIN_FITTED_OUTPUT)
        if available <= 0:
            return unit, 0
        before = sum(self._message_tokens(m, reasoning) for m in unit)
        fitted = unit
        for _ in range(3):  # the markers add a little; a retry or two absorbs it
            # Room for each marker, so the estimate lands under the budget.
            remove = math.ceil(excess_tokens * self._policy.chars_per_token) + 64 * len(unit)
            ratio = max(0.0, 1 - remove / available)
            fitted = [
                self._trim_tool_output(message, max(
                    _MIN_FITTED_OUTPUT,
                    _MIN_FITTED_OUTPUT + math.floor((size - _MIN_FITTED_OUTPUT) * ratio),
                ))
                if size > _MIN_FITTED_OUTPUT else message
                for message, size in zip(unit, sizes)
            ]
            after = sum(self._message_tokens(m, reasoning) for m in fitted)
            if before - after >= excess_tokens or ratio == 0.0:
                break
            excess_tokens += excess_tokens - (before - after)
        shortened = sum(new is not old for new, old in zip(fitted, unit))
        return fitted, shortened

    def _trim_tool_output(self, message: Message, limit: int | None = None) -> Message:
        if limit is None:
            limit = self._policy.max_tool_output_chars
        if limit is None or message.role is not Role.TOOL or len(message.content) <= limit:
            return message
        removed = len(message.content) - limit
        stored = _ARTIFACT_ID.search(message.content)
        hint = (f"; the full output is artifact {stored.group(1)}, read the part you need "
                "with read_artifact" if stored else
                "; call the tool again (a narrower range or query) if you need the rest")
        marker = f"\n[... {removed} characters truncated{hint} ...]\n"
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


STATE_HEADER = "[Working state - written by the harness, not by the user]"
"""First line of the anchored layout's state message (summary, omission note, ledger)."""

_ARTIFACT_ID = re.compile(r"stored as artifact ([0-9a-f]{16,64})")


def tool_ledger(messages: Sequence[Message], *, max_entries: int = 40, max_chars: int = 4_000) -> str:
    """One line per tool call in ``messages``: the tool, its arguments, and how it ended.

    Written without a model, so it is exact: a summary may leave out which files were read
    or which commands ran, and this cannot. Identical calls are listed once with a count;
    when there are too many, the oldest are left out and counted.
    """

    calls: dict[str, tuple[str, str]] = {}
    lines: dict[str, int] = {}
    for message in messages:
        for call in message.tool_calls:
            arguments = json.dumps(call.arguments, separators=(",", ":"), sort_keys=True)
            calls[call.id] = (call.name, _shorten_line(arguments, 160))
        if message.role is Role.TOOL and message.tool_call_id in calls:
            name, arguments = calls.pop(message.tool_call_id)
            line = f"- {name} {arguments} -> {_result_status(message.content)}"
            lines[line] = lines.pop(line, 0) + 1  # re-inserted: the latest occurrence orders it
    for name, arguments in calls.values():
        line = f"- {name} {arguments} -> no result"
        lines[line] = lines.pop(line, 0) + 1
    if not lines:
        return ""
    entries = [line + (f" (x{count})" if count > 1 else "") for line, count in lines.items()]
    header = ("[Earlier tool calls - their output is no longer shown; call a tool again if "
              "you need its output]")
    kept = entries[-max_entries:]
    while len(kept) > 1 and sum(len(entry) + 1 for entry in kept) > max_chars:
        kept = kept[1:]
    left_out = len(entries) - len(kept)
    if left_out:
        kept.insert(0, f"- ({left_out} older calls not listed)")
    return header + "\n" + "\n".join(kept)


def _shorten_line(text: str, limit: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _result_status(content: str) -> str:
    """How a tool message says it ended: its ``status`` field, or what its text shows."""

    text = content.lstrip()
    if text.startswith("{"):
        try:
            status = json.loads(text).get("status")
        except (ValueError, AttributeError):
            status = None
        if isinstance(status, str) and status:
            return status
        found = re.search(r'"status"\s*:\s*"([a-z_]+)"', text[-200:])
        if found:
            return found.group(1)
    if text.startswith("Refused") or text.startswith("Denied"):
        return "refused"
    return "returned"


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


ORCHESTRATOR_NOTE_PREFIXES = ("[Plan]", "[Budget]", "[Memory]")
"""Prefixes of the user messages the orchestrator injects; they are not the user's requests."""


def latest_request_index(messages: Sequence[Message], *, start: int = 0) -> int | None:
    """Index of the newest user message from ``start`` on that the user wrote, or None."""

    for index in range(len(messages) - 1, start - 1, -1):
        message = messages[index]
        if (message.role is Role.USER and message.content.strip()
                and not message.content.startswith(ORCHESTRATOR_NOTE_PREFIXES)):
            return index
    return None


def pinned_head_end(messages: Sequence[Message]) -> int:
    """Index after the leading system messages and the first user message (always kept)."""

    return _pinned_head_end(messages)


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
        stale_tool_output_chars=getattr(section, "stale_tool_output_chars", 0) or None,
        keep_recent_units=getattr(section, "keep_recent_units", 4),
        min_fitted_output_chars=getattr(section, "min_fitted_output_chars", 200),
        ledger_max_entries=getattr(section, "ledger_max_entries", 40),
        ledger_max_chars=getattr(section, "ledger_max_chars", 4_000),
    )


def limits_from_profiles(
    profiles: Mapping[str, object],
    *,
    default_output_reserve_tokens: int = 1024,
    default_window: int | None = None,
) -> dict[str, ContextLimits]:
    """Derive limits from model profiles that declare ``context_window``.

    The output reserve is the profile's ``parameters.max_tokens`` when set.
    ``default_window`` (``[context] default_window``) bounds the profiles that declare
    none; without it they are sent the full history.
    """

    limits: dict[str, ContextLimits] = {}
    for name, profile in profiles.items():
        window = getattr(profile, "context_window", None)
        if window is None:
            window = default_window
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


STRUCTURED_SUMMARY_INSTRUCTIONS = (
    "You compress the earlier part of an agent's conversation so the agent can keep "
    "working with a smaller context. Reply with these sections, in this order, and "
    "nothing else. Write 'none' under a section that has nothing.\n"
    "## Goal\nWhat the user wants, including the latest request.\n"
    "## Constraints\nRules, preferences and limits the user or the instructions set.\n"
    "## Decisions\nChoices made so far and why.\n"
    "## Files and commands\nPaths read or changed and commands run, with what each showed.\n"
    "## Facts\nExact identifiers, numbers, names and error messages learned from tools.\n"
    "## Done\nWork that is finished.\n"
    "## Next\nWhat remains, in order, and any open question.\n"
    "Merge the previous summary, if given, with the new messages into one summary. Copy "
    "identifiers, paths, numbers and error text exactly. Do not invent anything, do not "
    "address the user, and do not call tools."
)

SUMMARY_STYLES = {"prose": SUMMARY_INSTRUCTIONS, "structured": STRUCTURED_SUMMARY_INSTRUCTIONS}


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
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if max_input_chars < 1_000:
            raise ValueError("max_input_chars must be at least 1000")
        if max_message_chars < 200:
            raise ValueError("max_message_chars must be at least 200")
        self.profile = profile
        self.max_output_tokens = max_output_tokens
        self.parameters: Mapping[str, Any] = dict(parameters or {})
        self._max_input_chars = max_input_chars
        self._max_message_chars = max_message_chars
        self._input_chars = input_chars
        self._instructions = instructions

    def input_limit(self, profile: str) -> int:
        derived = self._input_chars(profile) if self._input_chars is not None else None
        if derived is None:
            return self._max_input_chars
        return max(1_000, min(self._max_input_chars, derived))

    def chunks(
        self, messages: Sequence[Message], start: int, end: int, *, profile: str
    ) -> list[tuple[int, int]]:
        """Split ``[start, end)`` at unit boundaries so each part's transcript fits one request.

        Room is left for the user's request, the previous summary and the labels. A single
        unit longer than the room is still one part (its messages are capped as before).
        """

        room = max(1_000, self.input_limit(profile) - self._max_message_chars
                   - self.max_output_tokens * 4 - 200)
        parts: list[tuple[int, int]] = []
        part_start, size, index = start, 0, start
        for unit in _group_units(messages[start:end]):
            length = sum(len(self._render(message)) + 2 for message in unit)
            if size and size + length > room:
                parts.append((part_start, index))
                part_start, size = index, 0
            size += length
            index += len(unit)
        if index > part_start:
            parts.append((part_start, index))
        return parts

    def request(
        self,
        previous: str | None,
        messages: Sequence[Message],
        *,
        profile: str,
        metadata: Mapping[str, str] | None = None,
        task: str | None = None,
    ) -> ModelRequest:
        """``task`` is the user's request, which stays in the context unsummarized: without
        it the summarizer cannot tell what matters and reports that there is no goal."""

        transcript = "\n\n".join(self._render(message) for message in messages)
        parts = []
        if task:
            parts.append(
                "The user's request (kept in full elsewhere; do not repeat it, use it to "
                "decide what matters):\n" + _shorten(task.strip(), self._max_message_chars)
            )
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
            # e.g. chat_template_kwargs = {enable_thinking = false}: a thinking model would
            # otherwise spend the output allowance thinking and return an empty summary.
            extra_body=dict(self.parameters) or None,
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
