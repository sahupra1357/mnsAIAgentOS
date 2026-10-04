"""In-process Python tool source, loaded by the root models.toml (`text-tools`).

Copied from examples/tools/example_tools.py, which stays as reference.

A tool module exposes one function that receives a registrar and registers
(ToolSpec, handler) pairs. Handlers are async, take the validated arguments,
and return JSON-compatible data.
"""

from agent_harness.domain import ToolRisk, ToolSpec


async def word_count(arguments):
    text = arguments["text"]
    return {"words": len(text.split()), "characters": len(text)}


def register(registrar):
    registrar.register(
        ToolSpec(
            name="word_count",
            description="Count the words and characters in a piece of text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 100000}},
                "required": ["text"],
                "additionalProperties": False,
            },
            risk=ToolRisk.READ_ONLY,
        ),
        word_count,
    )
