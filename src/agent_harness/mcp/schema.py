"""Reduce MCP tool input schemas to the JSON Schema subset the harness validates.

MCP servers publish full JSON Schema (``$schema``, ``default``, ``format``,
``anyOf``, type lists...). The harness validator supports a deliberate subset,
so schemas are normalized: supported keywords are kept, nullable unions become
their non-null type, and constraints that cannot be expressed are dropped
(making the schema *more permissive*, never stricter than the server's).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agent_harness.tools.validation import SchemaDefinitionError, check_schema

_KEEP = {
    "type", "properties", "required", "additionalProperties", "items", "enum",
    "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
    "pattern", "description", "title",
}
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


class SchemaNormalizationError(ValueError):
    """The schema could not be reduced to a valid supported schema."""


def normalize_input_schema(schema: Any) -> dict[str, Any]:
    """Return a supported object schema, or raise ``SchemaNormalizationError``."""

    if schema is None:
        schema = {"type": "object"}
    if not isinstance(schema, Mapping):
        raise SchemaNormalizationError("inputSchema must be an object")
    normalized = _normalize(schema)
    if normalized.get("type") is None and "properties" in normalized:
        normalized["type"] = "object"
    if normalized.get("type") != "object":
        raise SchemaNormalizationError("inputSchema must describe an object")
    try:
        check_schema(normalized)
    except SchemaDefinitionError as exc:
        raise SchemaNormalizationError(str(exc)) from exc
    return normalized


def _normalize(schema: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(schema)
    # A nullable union collapses to its single non-null member.
    for keyword in ("anyOf", "oneOf"):
        options = source.get(keyword)
        if isinstance(options, list):
            non_null = [o for o in options if isinstance(o, Mapping) and o.get("type") != "null"]
            if len(non_null) == 1:
                merged = {k: v for k, v in source.items() if k != keyword}
                merged.update(non_null[0])
                return _normalize(merged)
    if "const" in source and "enum" not in source:
        source["enum"] = [source["const"]]

    result: dict[str, Any] = {}
    schema_type = source.get("type")
    if isinstance(schema_type, list):
        candidates = [t for t in schema_type if t != "null"]
        schema_type = candidates[0] if len(candidates) == 1 else None
    if isinstance(schema_type, str) and schema_type in _TYPES:
        result["type"] = schema_type
    elif "properties" in source:
        result["type"] = "object"

    for keyword in ("description", "title"):
        if isinstance(source.get(keyword), str):
            result[keyword] = source[keyword]
    enum = source.get("enum")
    if isinstance(enum, list) and enum:
        result["enum"] = list(enum)

    kind = result.get("type")
    if kind == "object":
        properties = source.get("properties")
        if isinstance(properties, Mapping):
            result["properties"] = {
                str(name): _normalize(child) if isinstance(child, Mapping) else {}
                for name, child in properties.items()
            }
            required = source.get("required")
            if isinstance(required, list):
                names = [n for n in dict.fromkeys(required) if isinstance(n, str) and n in result["properties"]]
                if names:
                    result["required"] = names
        if isinstance(source.get("additionalProperties"), bool):
            result["additionalProperties"] = source["additionalProperties"]
    elif kind == "array":
        items = source.get("items")
        if isinstance(items, Mapping):
            result["items"] = _normalize(items)
        _copy_counts(source, result, ("minItems", "maxItems"))
    elif kind == "string":
        _copy_counts(source, result, ("minLength", "maxLength"))
        pattern = source.get("pattern")
        if isinstance(pattern, str) and _compiles(pattern):
            result["pattern"] = pattern  # patterns Python cannot compile are dropped
    elif kind in {"integer", "number"}:
        for keyword in ("minimum", "maximum"):
            value = source.get(keyword)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[keyword] = value
    return {k: v for k, v in result.items() if k in _KEEP}


def _compiles(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def _copy_counts(source: Mapping[str, Any], result: dict[str, Any], keywords: tuple[str, ...]) -> None:
    for keyword in keywords:
        value = source.get(keyword)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[keyword] = value
