"""Model calls the harness makes for itself (a summary, the notes of a turn).

Such a call wants a short answer, so its output is capped. The server counts a model's
thinking against the same cap, and some deployments cannot turn thinking off: the answer
then comes back cut off or empty. ``complete_whole`` asks again with more room until the
reply is whole, and remembers per profile what room that took.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from agent_harness.domain import FinishReason, ModelRequest, ModelResponse, Usage

MAX_ROOM_ATTEMPTS = 4
ROOM_FACTOR = 8  # the ceiling when the profile's own output limit is not known


@dataclass(slots=True)
class OutputRoom:
    """The output room that last returned a whole reply, per (profile, purpose)."""

    _known: dict[tuple[str, str], int] = field(default_factory=dict)

    def start(self, key: tuple[str, str], answer_tokens: int) -> int:
        return max(answer_tokens, self._known.get(key, 0))

    def learned(self, key: tuple[str, str], room: int) -> None:
        self._known[key] = room


@dataclass(frozen=True, slots=True)
class SideReply:
    response: ModelResponse
    usages: tuple[Usage, ...]  # one per request made
    room: int  # max_tokens of the last request
    cut_off: bool  # still cut off at the ceiling: the content is not whole

    @property
    def attempts(self) -> int:
        return len(self.usages)


def cut_off(response: ModelResponse) -> bool:
    return response.finish_reason is FinishReason.LENGTH


async def complete_whole(
    client: Any,
    request: ModelRequest,
    *,
    answer_tokens: int,
    ceiling: int | None,
    room: OutputRoom,
    key: tuple[str, str],
) -> SideReply:
    """Send ``request`` with room for ``answer_tokens`` of answer, and more if it is cut off.

    A reply that ends at the output limit is asked for again with twice the room (four
    times when it held no answer at all: the thinking alone did not fit), up to
    ``ceiling`` (None: ``ROOM_FACTOR`` times the answer). Every request's usage is returned.
    """

    limit = max(answer_tokens, ceiling if ceiling else answer_tokens * ROOM_FACTOR)
    now = min(limit, room.start(key, answer_tokens))
    usages: list[Usage] = []
    while True:
        response = await client.complete(replace(request, max_output_tokens=now))
        usages.append(response.usage)
        if not cut_off(response):
            if len(usages) > 1:
                room.learned(key, now)
            return SideReply(response, tuple(usages), now, False)
        if now >= limit or len(usages) >= MAX_ROOM_ATTEMPTS:
            # Start the next call here too: it will not fit in less, and finding that out
            # again costs a request each time.
            room.learned(key, now)
            return SideReply(response, tuple(usages), now, True)
        now = min(limit, now * (2 if response.content.strip() else 4))
