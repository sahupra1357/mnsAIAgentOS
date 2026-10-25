"""The message skeleton derived from a run's events (Phase 23d).

Events deliberately do not hold message text (tool output, answers, and reasoning stay
in the run snapshot, which keeps the event log small and redactable), so the stored
messages cannot be rebuilt from events. Their *shape* can: the role of each message, the
tool-call ids an assistant message asked for, and the call id a tool message answers.
:func:`derive_message_skeleton` replays that shape from the events, and a test checks it
equals :func:`message_skeleton` of ``run.messages`` at every checkpoint, so a new code
path that changes the history without an event is caught (the first step towards event
sourcing; see CLAUDE.md, Phase 23).

Which event adds which message:

* ``run_created``: the initial messages (``message_roles``);
* ``user_message_added``, ``steer_applied``: one user message each;
* ``hook_decided``: ``session_start`` with context, or ``stop`` with ``continue``: a user message;
* ``plan_step_updated`` with ``message``, ``output_rejected``: a user message (an orchestrator note);
* ``model_responded`` (``ok``): an assistant message with ``tool_call_ids``; a
  ``continuation`` reply instead removes the note before it and joins the previous answer;
* ``guardrail_refused``: the refusal message (assistant);
* ``tool_responded``: a tool message, except a result that will be retried (``retrying``);
* ``rewound``: removes the last ``archived_messages`` messages;
* ``reply_discarded``: removes the last ``archived_messages`` messages (an oversized reply)
  and adds a user message (the note asking again).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from agent_harness.domain import EventType, Message, Role, RunEvent


@dataclass(frozen=True, slots=True)
class MessageShape:
    role: str
    tool_call_ids: tuple[str, ...] = ()
    tool_call_id: str | None = None


def message_skeleton(messages: Iterable[Message]) -> list[MessageShape]:
    return [
        MessageShape(
            role=message.role.value,
            tool_call_ids=tuple(call.id for call in message.tool_calls),
            tool_call_id=message.tool_call_id if message.role is Role.TOOL else None,
        )
        for message in messages
    ]


class SkeletonUnavailable(ValueError):
    """The events predate the fields the skeleton needs (runs created before Phase 23)."""


def derive_message_skeleton(events: Sequence[RunEvent]) -> list[MessageShape]:
    shapes: list[MessageShape] = []
    user = MessageShape(Role.USER.value)
    for event in events:
        payload = event.payload
        kind = event.type
        if kind is EventType.RUN_CREATED:
            roles = payload.get("message_roles")
            if not isinstance(roles, list):
                raise SkeletonUnavailable("run_created has no message_roles")
            shapes = [MessageShape(str(role)) for role in roles]
        elif kind in {EventType.USER_MESSAGE_ADDED, EventType.STEER_APPLIED, EventType.OUTPUT_REJECTED}:
            shapes.append(user)
        elif kind is EventType.PLAN_STEP_UPDATED:
            if payload.get("message") is True:
                shapes.append(user)
        elif kind is EventType.HOOK_DECIDED:
            hook_event = payload.get("hook_event")
            decision = payload.get("decision")
            if (hook_event == "session_start" and payload.get("context") and decision != "deny") or (
                hook_event == "stop" and decision == "continue"
            ):
                shapes.append(user)
        elif kind is EventType.MODEL_RESPONDED:
            if payload.get("ok") is not True:
                continue
            if "continuation" in payload:
                shapes.pop()  # the note asking to continue; the reply joins the answer before it
                continue
            ids = payload.get("tool_call_ids")
            if ids is None and payload.get("tool_call_count"):
                raise SkeletonUnavailable("model_responded has no tool_call_ids")
            shapes.append(MessageShape(Role.ASSISTANT.value, tuple(str(i) for i in ids or ())))
        elif kind is EventType.GUARDRAIL_REFUSED:
            shapes.append(MessageShape(Role.ASSISTANT.value))
        elif kind is EventType.TOOL_RESPONDED:
            if not payload.get("retrying"):
                shapes.append(MessageShape(Role.TOOL.value, tool_call_id=str(payload.get("call_id"))))
        elif kind is EventType.REWOUND:
            archived = int(payload.get("archived_messages", 0))
            if archived:
                del shapes[-archived:]
        elif kind is EventType.REPLY_DISCARDED:
            archived = int(payload.get("archived_messages", 0))
            if archived:
                del shapes[-archived:]
            shapes.append(user)
    return shapes
