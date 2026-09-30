"""The ``ask_user`` tool (Phase 11d): the model asks the user a clarifying question.

The run pauses in ``awaiting_input`` through the approval path (an approval
request with ``kind = "question"``). ``AgentOrchestrator.answer`` returns the
user's reply to the model as the tool result; declining returns a refusal.
"""

from __future__ import annotations

from agent_harness.domain import ToolRisk, ToolSpec

ASK_USER_TOOL = "ask_user"
MAX_QUESTION_CHARS = 2_000
MAX_OPTIONS = 10

ASK_USER_SPEC = ToolSpec(
    name=ASK_USER_TOOL,
    description=(
        "Ask the user a clarifying question and wait for the answer. Use it only when the "
        "request is ambiguous or you need a decision or information you cannot find with "
        "your other tools. Ask one concise question. Optionally list suggested answers in "
        "options; the user may still answer freely."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1, "maxLength": MAX_QUESTION_CHARS},
            "options": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
                "maxItems": MAX_OPTIONS,
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)
