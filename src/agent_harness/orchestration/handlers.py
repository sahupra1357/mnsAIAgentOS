"""Orchestrator-owned tool handlers (Phase 23b).

Some tools are not run by the tool runtime: the orchestrator handles them itself
(``exit_plan``, ``ask_user``, ``todo_write``, ``submit_result``, the dynamic-loading
meta-tools, the plan tools, ``agent``). Each is a :class:`ToolHandler` in a
:class:`ToolHandlerRegistry`, so adding a control primitive is one registration
instead of another branch in ``_execute_tool``; the prefetch pass and the offered
tool list read the same registry.

Order in ``_execute_tool`` (unchanged from the earlier ``if`` chain):

1. ``pre_tool`` hook and permission decision, ``TOOL_REQUESTED`` recorded;
2. handlers with ``before_approval = True`` (they pause or decide approval themselves);
3. the generic approval pause for a permission rule's ``ask``;
4. the other handlers;
5. generic execution through the tool runtime.

A handler whose ``applies`` returns False for this call falls through to the next
step, as the old conditions did (``submit_result`` without an output schema, meta-tools
without dynamic loading, plan tools outside a plan).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from agent_harness.domain import RunEvent, RunRecord, ToolCall, ToolSpec


@dataclass(frozen=True, slots=True)
class ToolCallContext:
    """What a handler gets: the run, its events, the (hook-final) call, and its decision state."""

    run: RunRecord
    events: list[RunEvent]
    call: ToolCall
    spec: ToolSpec
    attempt: int
    approved: bool


ToolHandle = Callable[[ToolCallContext], Awaitable[None]]


def _always(run: RunRecord, call: ToolCall) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class ToolHandler:
    """One orchestrator-owned tool (or family of tools sharing a handler).

    ``handle`` must record the call's result (or pause the run); the generic path is
    skipped once it returns. ``parallel_safe`` lets the prefetch pass run the call through
    the tool runtime concurrently; orchestrator-owned tools normally are not.
    ``offer`` adds a spec to the tools offered to the model (None: offered elsewhere, as
    the built-in handlers are); it is placed after the built-in specs, so existing tool
    lists and their prefix-cache hashes do not change.
    """

    names: frozenset[str]
    handle: ToolHandle
    applies: Callable[[RunRecord, ToolCall], bool] = _always
    before_approval: bool = False
    parallel_safe: bool = False
    offer: Callable[[RunRecord], ToolSpec | None] | None = None


class ToolHandlerRegistry:
    """Handlers by tool name; one handler per name."""

    def __init__(self, handlers: Iterable[ToolHandler] = ()) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._order: list[ToolHandler] = []
        for handler in handlers:
            self.register(handler)

    def register(self, handler: ToolHandler, *, replace: bool = False) -> None:
        if not handler.names:
            raise ValueError("a tool handler needs at least one tool name")
        taken = sorted(name for name in handler.names if name in self._handlers)
        if taken and not replace:
            raise ValueError(f"tool handler already registered for: {', '.join(taken)}")
        for name in taken:
            old = self._handlers[name]
            if old in self._order:
                self._order.remove(old)
        for name in handler.names:
            self._handlers[name] = handler
        self._order.append(handler)

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    def names(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def claims(self, name: str) -> bool:
        """True if the orchestrator owns this tool and it must not be prefetched."""

        handler = self._handlers.get(name)
        return handler is not None and not handler.parallel_safe

    def lookup(self, run: RunRecord, call: ToolCall, *, before_approval: bool) -> ToolHandler | None:
        handler = self._handlers.get(call.name)
        if handler is None or handler.before_approval is not before_approval:
            return None
        return handler if handler.applies(run, call) else None

    def offered(self, run: RunRecord) -> tuple[ToolSpec, ...]:
        specs = []
        for handler in self._order:
            if handler.offer is not None and (spec := handler.offer(run)) is not None:
                specs.append(spec)
        return tuple(specs)
