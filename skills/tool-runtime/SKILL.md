---
name: tool-runtime
description: Build or modify typed agent tools, the tool registry, validation, execution policies, permissions, and result normalization. Use for tool definitions, tool invocation, sandbox boundaries, or approval rules.
---

# Tool Runtime

Treat tools as untrusted capability boundaries, even when the initial implementation runs in-process.

## Tool Contract

Each tool declares a stable name, description, JSON-compatible input schema, timeout, risk level, and whether it is idempotent. Execution returns a normalized result that distinguishes success, validation failure, policy denial, timeout, retryable failure, and permanent failure.

Reject duplicate tool names and unknown tool calls. Validate arguments before executing implementation code. Keep registration explicit; importing a module must not silently grant capabilities.

Tools can also come from config-declared sources (Python modules, HTTP, web search, MCP
servers) with a persistent catalog and dynamic loading; see `skills/tool-platform/SKILL.md`.
`ToolSpec.trust = untrusted` marks output that the orchestrator wraps as data.

## Execution Policy

- Read-only tools may run automatically when enabled.
- Side-effecting tools require an idempotency key.
- Approval-required tools must return an approval-required result without invoking the handler; the orchestrator persists an approval request and pauses the run.
- High-risk or destructive tools require an approval decision supplied by the application boundary.
- Enforce per-call timeout and output-size limits.
- Return sanitized error information to the model while keeping diagnostic detail in internal events.
- Do not build shell command strings from model arguments.

For coding tools, use the project coding skill as well. Its command guardrails are independent of approval: an approved command can still be rejected as dangerous.

The execution interface should allow a future out-of-process sandbox without changing the orchestrator.

## Handoff

Include simple deterministic tools for examples and tests. Test schema rejection, unknown tools, timeout, policy denial, duplicate registration, and idempotency-key enforcement.
