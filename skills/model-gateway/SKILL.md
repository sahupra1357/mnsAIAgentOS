---
name: model-gateway
description: Create or change provider-neutral model contracts, configurable model routing, and provider adapters. Use when adding a model provider, endpoint, capability route, normalized response, or model retry behavior.
---

# Model Gateway

Make model selection a configuration concern. Orchestration code must depend on a small normalized interface and must not branch on provider names.

## Normalized Contract

A model request should carry messages, available tool schemas, sampling controls, a timeout, and trace metadata. A normalized response should represent assistant text, zero or more tool calls, usage, provider request ID, and finish reason.

`ModelRequest.response_schema` (a JSON Schema) and `ModelRequest.tool_choice` (`auto`,
`none`, `required`, or one tool name) default to `None`, and adapters must then leave the
payload unchanged. The OpenAI-compatible adapter maps them to `response_format`
(`json_schema`, strict) and `tool_choice` (a named tool becomes
`{"type": "function", "function": {"name": ...}}`); native Ollama maps the schema to
`format` and has no `tool_choice`. The orchestrator uses them for runs with an
`output_schema` (see `structured_output.py`) and validates results itself, because servers
enforce both loosely.

Preserve provider-only fields inside adapter metadata. Do not leak provider SDK types into shared domain objects.

## Configuration

Support named model profiles. A profile identifies an adapter, model name, endpoint, credential environment-variable name, timeout, and optional default parameters. Resolve secrets at runtime; configuration files contain only environment-variable references.

OpenAI-compatible endpoints are the baseline adapter because they cover hosted and many local models. Keep the adapter injectable so tests use a fake transport and never access the network.

## Reliability

- Validate normalized output before returning it.
- Classify authentication, throttling, timeout, malformed-response, and provider errors.
- Retry only explicit transient failures with bounded backoff.
- Do not silently route to a different model unless the selected routing policy permits fallback.
- Return usage when provided; mark it unknown rather than inventing counts.

## Handoff

Document how callers select a profile and how another adapter can be registered. Provide tests for text responses, tool calls, malformed responses, timeout classification, and missing credentials.

