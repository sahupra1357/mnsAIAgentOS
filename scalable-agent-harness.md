# Designing a Scalable Agent Harness

A scalable agent harness is the runtime around an AI model that manages tasks, tools, memory, safety, retries, and observability. Treat the model as an unreliable decision-making component—not as the application itself.

## High-Level Architecture

```text
Client / API
     │
     ▼
Task Service ──► Durable Queue
                     │
                     ▼
              Agent Orchestrator
              (state machine)
               │    │     │
               ▼    ▼     ▼
           Model  Tools  Memory
          Gateway       / Context
               │
               ▼
        Events, Traces, Evals
```

## 1. Represent Execution as a State Machine

Each agent run should move through explicit states:

```text
queued → planning → model_call → tool_call → validating
                    ▲              │
                    └──────────────┘
                         loop
                              → completed / failed / awaiting_input
```

Persist the state after every meaningful step. If a worker crashes, another worker can resume the run rather than restarting it.

Avoid implementing the core loop as one long synchronous request.

## 2. Separate Logical Runs from Worker Processes

A run is durable data. A worker is temporary compute.

Store:

- Run ID and status
- Conversation and instructions
- Current step
- Tool calls and results
- Token, time, and cost budgets
- Retry history
- Produced artifacts
- Approval or user-input requirements

This allows workers to scale horizontally and process runs from a queue.

## 3. Use an Event-Driven Execution Model

Record events such as:

```text
RunCreated
ModelRequested
ModelResponded
ToolRequested
ToolCompleted
CheckpointCreated
RunCompleted
```

Events provide auditability, debugging, streaming updates, and the ability to reconstruct what happened. Large tool results should go into object storage, with references stored in the event log.

## 4. Put Models Behind a Gateway

Do not let every agent call providers directly. A model gateway should handle:

- Provider and model selection
- Authentication
- Timeouts and retries
- Rate limiting
- Structured-output validation
- Token accounting
- Caching where appropriate
- Fallback models
- Cost and latency routing

The orchestrator asks for capabilities—such as “fast classifier” or “strong reasoning model”—instead of depending heavily on one provider.

## 5. Standardize Tools

Give every tool a clear contract:

```text
name
description
input schema
output schema
timeout
permissions
idempotency behavior
side-effect classification
```

Separate tools into risk levels:

- Read-only
- Reversible writes
- External side effects
- Destructive or high-risk actions

Sensitive actions should require policy checks and possibly human approval. Run untrusted tools in isolated containers or sandboxes with restricted filesystem, network, credentials, CPU, memory, and execution time.

## 6. Make Retries Safe

Failures are normal: APIs time out, models produce invalid output, and workers restart.

Use:

- Idempotency keys for side-effecting operations
- Exponential backoff with jitter
- Maximum retry limits
- Dead-letter queues
- Checkpoints before expensive actions
- Compensation logic for partially completed workflows

Never blindly retry actions such as payments, emails, deployments, or deletions.

## 7. Control Context Explicitly

Do not continually append everything to the prompt. Maintain different kinds of state:

- Working context: information needed for the current step
- Run state: authoritative structured data
- Conversation history: user-visible interaction
- Long-term memory: selectively retrieved knowledge
- Artifacts: files and large tool outputs

Use summarization and retrieval to construct a bounded prompt for each model call. The database—not the model’s prompt—should be the source of truth.

## 8. Enforce Budgets and Stopping Conditions

Every run needs limits:

- Maximum steps
- Maximum model calls
- Token and monetary budget
- Wall-clock deadline
- Tool timeout
- Repeated-action detection
- Maximum parallel branches

Without these controls, agents can loop, duplicate actions, or consume unlimited resources.

## 9. Scale Different Components Independently

Model calls, browser automation, code execution, and lightweight tools have different resource profiles. Give them separate queues and worker pools.

For example:

```text
model queue   → network-oriented workers
browser queue → browser containers
code queue    → isolated compute sandboxes
quick tools   → lightweight workers
```

Use per-tenant quotas and fair scheduling so one large customer or runaway run cannot starve the system.

## 10. Build Observability Around Runs and Steps

For every step, capture:

- Run, step, tenant, and trace IDs
- Model and prompt version
- Tool name and sanitized arguments
- Latency, tokens, and cost
- Retry and error category
- Policy decisions
- Final outcome

Track product-level metrics too: completion rate, human intervention rate, task quality, cost per successful run, and failure categories. Raw model accuracy alone is not enough.

## 11. Treat Prompts and Policies Like Versioned Code

Version:

- System instructions
- Tool definitions
- Routing policies
- Context-building logic
- Output schemas
- Safety rules
- Model configuration

Store the version used by each run. This makes failures reproducible and enables controlled rollouts and rollback.

## 12. Test with Evaluation Scenarios

Create a suite containing:

- Normal tasks
- Ambiguous requests
- Tool failures
- Malformed model responses
- Prompt-injection attempts
- Duplicate delivery
- Worker crashes
- Long-running tasks
- Permission violations

Run these evaluations whenever prompts, models, tools, or orchestration logic change. Use shadow traffic and canary releases before full rollout.

## Core Design Principle

> Keep orchestration deterministic and durable; use the model only for decisions that genuinely require intelligence.

A good first implementation is usually a modular monolith with PostgreSQL, a durable job queue, object storage, stateless workers, and an isolated tool runner. Split it into separate services only when load, security boundaries, or team ownership justify the added complexity.

## Coding-Agent Workflow

The harness can be specialized into a coding agent by registering workspace-safe tools:

```text
user request
  → model creates a plan
  → reads and searches the repository
  → proposes an exact patch
  → awaits user approval
  → applies the patch
  → awaits approval before code execution
  → runs allowlisted tests in Docker
  → inspects Git status and diff
  → awaits approval before commit or pull request
  → returns a summary
```

Coding tools should enforce a workspace boundary, reject path traversal and credential-like files, use structured `argv` with `shell=False`, and apply command allowlists before a process is created. Always-forbidden commands remain blocked even after approval. The Docker execution backend should mount only the workspace, disable networking by default, drop capabilities, set `no-new-privileges`, apply CPU/memory/PID limits, and use a read-only root filesystem with a bounded temporary filesystem.

Approval is a durable state transition, not a prompt hidden inside a tool handler. The tool returns `approval_required`, the orchestrator persists an approval event and moves the run to `awaiting_input`, and the application explicitly calls approve or deny. An approval denial is sent back to the model as a tool result so it can explain the limitation or choose another path.
