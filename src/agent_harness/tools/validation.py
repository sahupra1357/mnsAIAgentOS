"""Small, dependency-free validation for the JSON Schema subset used by tools."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


class SchemaDefinitionError(ValueError):
    """Raised when a tool declares an unsupported or malformed schema."""


class ArgumentValidationError(ValueError):
    """Raised when tool arguments do not match the declared schema."""


_SUPPORTED_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_SUPPORTED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "description",
    "title",
}


def check_schema(schema: Mapping[str, Any], *, path: str = "$") -> None:
    """Validate the supported schema vocabulary before a tool is registered."""

    if not isinstance(schema, Mapping):
        raise SchemaDefinitionError(f"{path}: schema must be an object")
    unknown = set(schema) - _SUPPORTED_KEYWORDS
    if unknown:
        keyword = sorted(unknown)[0]
        raise SchemaDefinitionError(f"{path}: unsupported schema keyword {keyword!r}")

    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in _SUPPORTED_TYPES:
        raise SchemaDefinitionError(f"{path}: unsupported type {schema_type!r}")

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise SchemaDefinitionError(f"{path}: enum must be a non-empty array")

    properties = schema.get("properties")
    if properties is not None:
        if schema_type != "object" or not isinstance(properties, Mapping):
            raise SchemaDefinitionError(f"{path}: properties requires an object schema")
        for name, child in properties.items():
            if not isinstance(name, str):
                raise SchemaDefinitionError(f"{path}: property names must be strings")
            check_schema(child, path=f"{path}.properties.{name}")

    required = schema.get("required")
    if required is not None:
        if schema_type != "object" or not _is_string_sequence(required):
            raise SchemaDefinitionError(f"{path}: required must be an array of strings")
        if len(set(required)) != len(required):
            raise SchemaDefinitionError(f"{path}: required contains duplicate names")
        if properties is not None and any(name not in properties for name in required):
            raise SchemaDefinitionError(f"{path}: required names must exist in properties")

    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise SchemaDefinitionError(f"{path}: additionalProperties must be a boolean")

    items = schema.get("items")
    if items is not None:
        if schema_type != "array":
            raise SchemaDefinitionError(f"{path}: items requires an array schema")
        check_schema(items, path=f"{path}.items")

    for keyword in ("minimum", "maximum"):
        value = schema.get(keyword)
        if value is not None and (not _is_number(value)):
            raise SchemaDefinitionError(f"{path}: {keyword} must be a number")
    for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
        value = schema.get(keyword)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise SchemaDefinitionError(f"{path}: {keyword} must be a non-negative integer")
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise SchemaDefinitionError(f"{path}: pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SchemaDefinitionError(f"{path}: invalid pattern") from exc


def validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate arguments, raising a concise error safe to expose to a model."""

    _validate(arguments, schema, path="$")


def validate_value(value: Any, schema: Mapping[str, Any]) -> None:
    """Validate any JSON value (not only an argument object) against a schema."""

    _validate(value, schema, path="$")


def _validate(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise ArgumentValidationError(f"{path}: value is not in the allowed enum")

    schema_type = schema.get("type")
    if schema_type is not None and not _matches_type(value, schema_type):
        raise ArgumentValidationError(f"{path}: expected {schema_type}")

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ArgumentValidationError(f"{path}.{name}: required property is missing")
        if schema.get("additionalProperties", True) is False:
            extra = set(value) - set(properties)
            if extra:
                name = sorted(extra)[0]
                raise ArgumentValidationError(f"{path}.{name}: additional property is not allowed")
        for name, child in properties.items():
            if name in value:
                _validate(value[name], child, path=f"{path}.{name}")
    elif schema_type == "array":
        _check_bound(value, schema, "minItems", len, path)
        _check_bound(value, schema, "maxItems", len, path, maximum=True)
        child = schema.get("items")
        if child is not None:
            for index, item in enumerate(value):
                _validate(item, child, path=f"{path}[{index}]")
    elif schema_type == "string":
        _check_bound(value, schema, "minLength", len, path)
        _check_bound(value, schema, "maxLength", len, path, maximum=True)
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise ArgumentValidationError(f"{path}: string does not match the required pattern")
    elif schema_type in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ArgumentValidationError(f"{path}: value is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ArgumentValidationError(f"{path}: value is above maximum")


def _matches_type(value: Any, schema_type: str) -> bool:
    return {
        "object": lambda: isinstance(value, Mapping),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: _is_number(value),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }[schema_type]()


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _is_string_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, str) for item in value)
    )


def _check_bound(
    value: Any,
    schema: Mapping[str, Any],
    keyword: str,
    measure: Any,
    path: str,
    *,
    maximum: bool = False,
) -> None:
    if keyword not in schema:
        return
    actual = measure(value)
    limit = schema[keyword]
    if (maximum and actual > limit) or (not maximum and actual < limit):
        qualifier = "above maximum" if maximum else "below minimum"
        raise ArgumentValidationError(f"{path}: length is {qualifier}")
