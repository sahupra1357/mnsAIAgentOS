"""Send one system message: merge the system messages that open the conversation.

A run's history can start with several system messages: instruction files (AGENTS.md),
the system prompt, the plan-and-execute instructions, the guardrails canary. Many chat
templates accept exactly one system message, first; Qwen 3.x templates (llama.cpp
``--jinja``, vLLM, SGLang) raise "System message must be at the beginning" otherwise.
Every client merges the leading system messages into one before sending, joined by a
blank line in their original order, so the text is identical every turn (prefix cache).
The stored history keeps them separate. System messages after the first non-system
message are left alone; the harness never adds any there.
"""

from __future__ import annotations

import dataclasses

from agent_harness.domain import Message, ModelRequest, Role


def single_system_message(request: ModelRequest) -> ModelRequest:
    messages = request.messages
    leading = 0
    while leading < len(messages) and messages[leading].role is Role.SYSTEM:
        leading += 1
    if leading < 2:
        return request
    text = "\n\n".join(m.content for m in messages[:leading] if m.content)
    merged = Message(role=Role.SYSTEM, content=text)
    return dataclasses.replace(request, messages=(merged, *messages[leading:]))
