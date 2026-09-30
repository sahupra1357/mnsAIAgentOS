# Agent Harness Collaboration Guide

## Mission

Build a provider-neutral, asynchronous agent harness whose model, storage, and tool implementations can be replaced without changing the orchestration loop.

The first usable version must support:

- Selecting a model through configuration rather than application code.
- OpenAI-compatible chat-completion endpoints, including locally hosted endpoints.
- A deterministic agent state machine with explicit stopping limits.
- Durable run, event, and message state through a storage interface.
- Typed tools with validation, timeouts, idempotency metadata, and risk classification.
- Resuming interrupted runs from checkpoints.
- Tests that use fake models and tools and never require network access.

## Architecture Boundaries

Keep dependencies directed inward:

```text
API / CLI → Application services → Domain contracts
                    │                    ▲
                    ├── Model gateway ───┤
                    ├── Tool runtime ────┤
                    └── Persistence ─────┘
```

- Domain objects and protocols must not import provider SDKs, web frameworks, or database implementations.
- The orchestrator may use only the model, tool, and run-store protocols.
- Provider-specific request and response shapes stay inside provider adapters.
- Tool implementations stay outside the orchestration loop.
- SQLite is a local durable implementation, not part of the domain contract.
- All externally supplied identifiers and arguments must be validated at a boundary.

## Project Layout

```text
src/agent_harness/
  domain.py              Shared domain values and protocols
  orchestrator.py        Durable execution state machine
  config.py              Environment/config-file loading
  model_gateway/         Model routing and provider adapters
  tools/                 Tool contracts, registry, policies, implementations
  coding/                Workspace-safe coding tools and execution backends
  persistence/           Run/event store implementations
  api.py                 Optional HTTP entrypoint
  cli.py                 Local operator entrypoint
tests/                    Network-free unit and integration tests
examples/                 Small runnable configurations and examples
```

## Project-Local Skills

Read the skill that matches the component before modifying it:

- Orchestration or persistence: `skills/durable-agent-orchestration/SKILL.md`
- Models, providers, or routing: `skills/model-gateway/SKILL.md`
- Tools, permissions, or execution: `skills/tool-runtime/SKILL.md`
- Coding-agent tools or sandbox execution: `skills/coding-agent/SKILL.md`
- Tests, evaluations, or release checks: `skills/harness-verification/SKILL.md`
- Tool catalog, tool sources, MCP, or dynamic tool loading: `skills/tool-platform/SKILL.md`
- Planning, delegation, or parallel execution: `skills/planning-delegation/SKILL.md`
- Storage migrations, Postgres, tenancy, security, or deployment: `skills/production-hardening/SKILL.md`

Read more than one when a change crosses boundaries. These project skills supplement, but do not override, user instructions or repository-wide safety rules.

## Multi-Agent Working Rules

- Respect the paths assigned by the coordinating agent. Do not edit another agent's owned paths unless asked.
- Shared contracts belong in `src/agent_harness/domain.py`; coordinate before changing them after initial creation.
- Prefer small protocol surfaces and dependency injection over global registries or provider conditionals.
- Keep commits and patches focused. Do not reformat unrelated files.
- Report created files, public interfaces, test coverage, and any integration assumptions to the coordinating agent.
- Never place real credentials, tokens, or production endpoints in source, tests, examples, or logs.

## Implementation Standards

- Target Python 3.12 or newer and use type annotations throughout public interfaces.
- Prefer standard-library components unless a dependency materially improves a boundary, validation, or server interface.
- Use `async` interfaces for model and tool execution.
- Use UTC timestamps and stable string identifiers.
- Represent run status, event type, finish reason, and tool risk with enums.
- Preserve raw provider payloads only when explicitly enabled; sanitized normalized data is the default.
- Every loop must have step, token/cost, and wall-clock stopping conditions.
- Every side-effecting tool call must carry an idempotency key.
- Retries must be bounded and limited to errors classified as retryable.
- Coding commands must use structured argv, an allowlist, and a configured sandbox; never construct shell command strings from model input.
- Approval-required actions must pause the persisted run and resume only after an explicit application-level decision.
- Dangerous commands remain denied even after approval; approval is not a bypass for guardrails.

## Verification

Run the narrowest relevant tests while developing, then run the complete suite before handoff:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
```

Tests must cover success, malformed model output, unknown tools, tool failure, retry exhaustion, budget exhaustion, cancellation, and resume behavior where relevant.
