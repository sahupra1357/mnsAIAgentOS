The harness has two entry points:

  1. CLI entry point: agent-harness run ...
  2. Python entry point: build_application(...)

  ## 1. CLI entry point

  Example:

  agent-harness run \
    --config models.toml \
    --model-profile local \
    "Inspect the repository and explain the failing tests"

  The call flow is:

  main()
    └── _run_command()
          ├── build_application()
          ├── create_run()
          └── orchestrator.run()

  Relevant file: src/agent_harness/cli.py

  ## 2. Application construction

  build_application() creates the runtime components:

  Configuration
      ↓
  ModelGateway
      ↓
  ToolRuntime
      ↓
  SQLiteRunStore
      ↓
  AgentOrchestrator

  Relevant file: src/agent_harness/application.py

  It performs these actions:

  - Loads models.toml
  - Selects the configured model provider
  - Creates the tool registry and policies
  - Creates coding tools if coding mode is enabled
  - Opens the SQLite persistence layer
  - Creates the orchestrator

  The orchestration loop does not know whether the model is OpenAI, local, or another provider.

  ## 3. Run creation

  The CLI converts the user prompt into a message:

  Message(
      role=Role.USER,
      content="Inspect the repository and explain the failing tests",
  )

  Then it creates a durable run:

  run = await app.orchestrator.create_run(
      model_profile="local",
      messages=[message],
      budget=RunBudget(...),
  )

  The run is persisted before contacting the model:

  RunStatus.QUEUED
      ↓
  RUN_CREATED event

  This gives every run a stable run_id.

  ## 4. Run starts

  orchestrator.run(run_id) loads the run from SQLite.

  If the run is valid and not already finished:

  QUEUED
    ↓
  RUNNING
    ↓
  RUN_STARTED event

  The orchestrator then enters its execution loop.

  Relevant file: src/agent_harness/orchestrator.py

  ## 5. Model request

  The orchestrator resolves the configured model profile:

  model = await self.models.resolve(run.model_profile)

  It sends:

  - Conversation messages
  - Tool specifications
  - Run metadata
  - Current step information

  The provider adapter converts the normalized request into the provider-specific format.

  For OpenAI-compatible models, this becomes a /chat/completions request.

  MODEL_REQUESTED event
      ↓
  Model provider
      ↓
  Normalized ModelResponse
      ↓
  MODEL_RESPONDED event

  Provider-specific code is isolated in:

  src/agent_harness/model_gateway

  ## 6. Model decision

  The model response has two possible paths.

  ### Final answer path

  If the assistant returns text without tool calls:

  Assistant response
      ↓
  Persist response
      ↓
  RUN_COMPLETED
      ↓
  Return final answer

  The final answer is printed to stdout.

  ### Tool-call path

  If the model requests a tool:

  Assistant tool call
      ↓
  Validate tool name and arguments
      ↓
  Check policy and risk
      ↓
  Execute or request approval

  The tool call is not executed blindly.

  ## 7. Tool validation and policy

  The tool runtime validates:

  - Tool name
  - Argument schema
  - Argument types
  - Required fields
  - Timeout settings
  - Idempotency key
  - Risk classification
  - Approval policy

  For coding tools, additional controls apply:

  - Workspace boundary checks
  - Protected file checks
  - Command allowlists
  - Dangerous command rejection
  - Structured argv
  - Docker sandbox execution

  Relevant directories:

  - src/agent_harness/tools
  - src/agent_harness/coding

  ## 8. Read-only tool execution

  A read-only tool such as read_file, search_files, or git_diff can execute immediately if it passes validation.

  TOOL_REQUESTED
      ↓
  Tool policy check
      ↓
  Tool handler
      ↓
  Tool result
      ↓
  TOOL_RESPONDED

  The tool result is appended to the conversation as a tool message.

  Then the model receives the updated conversation and decides what to do next.

  Tool result
      ↓
  Next model request

  ## 9. Approval-required execution

  A side-effecting or high-risk tool may require user approval.

  Example tools:

  - apply_patch
  - run_tests
  - run_command
  - git_commit
  - create_pull_request

  The handler is not executed before approval.

  Tool requires approval
      ↓
  APPROVAL_REQUESTED event
      ↓
  RunStatus.AWAITING_INPUT
      ↓
  Persist checkpoint
      ↓
  Process exits

  The CLI prints an approval request such as:

  {
    "run_id": "run-123",
    "call_id": "call-456",
    "tool_name": "apply_patch",
    "risk": "side_effect"
  }

  At this point, the run is safely paused in SQLite.

  ## 10. Resume after approval

  Approve the action:

  agent-harness approve \
    --config models.toml \
    --database agent-harness.sqlite3 \
    RUN_ID \
    --allow

  Deny the action:

  agent-harness approve \
    --config models.toml \
    --database agent-harness.sqlite3 \
    RUN_ID \
    --deny

  The approval flow is:

  approve command
      ↓
  pending_approval()
      ↓
  resolve_approval()
      ↓
  APPROVAL_RESOLVED event
      ↓
  resume(run_id)
      ↓
  Continue orchestration

  If approved, the tool runs.

  If denied, a policy-denied tool result is sent back to the model, allowing it to explain the denial or choose another approach.

  Approval does not bypass dangerous-command guardrails. Commands such as rm, sudo, shell execution, unrestricted Docker access, and unsafe Git operations remain blocked.

  ## 11. Loop continuation

  After a successful tool call:

  Tool result
      ↓
  Append tool message
      ↓
  Save checkpoint
      ↓
  Call model again

  The normal loop is:

  Model
    ↓
  Tool decision
    ↓
  Validate
    ↓
  Approve if needed
    ↓
  Execute tool
    ↓
  Persist result
    ↓
  Model again

  This continues until the model returns a final answer or the run reaches a stopping condition.

  ## 12. Stopping conditions

  Every run is bounded by:

  - Maximum number of steps
  - Maximum wall-clock duration
  - Maximum token usage
  - Maximum estimated cost
  - Maximum tool attempts
  - Cancellation
  - Permanent model error
  - Permanent tool error

  Possible persisted statuses are:

  QUEUED
  RUNNING
  AWAITING_INPUT
  COMPLETED
  FAILED
  CANCELLED
  BUDGET_EXHAUSTED

  Because every event and checkpoint is stored, an interrupted run can be resumed without losing its previous messages or tool results.

  ## 13. Programmatic usage

  You can use the harness directly from Python:

  import asyncio

  from agent_harness.application import build_application
  from agent_harness.domain import Message, Role, RunStatus


  async def main() -> None:
      async with build_application(
          config_path="models.toml",
          database_path="agent-harness.sqlite3",
      ) as app:
          run = await app.orchestrator.create_run(
              model_profile="local",
              messages=[
                  Message(
                      role=Role.USER,
                      content="Inspect the repository and summarize the architecture.",
                  )
              ],
          )

          result = await app.orchestrator.run(run.id)

          if result.status is RunStatus.AWAITING_INPUT:
              approval = await app.orchestrator.pending_approval(run.id)
              print("Approval required:", approval)
              return

          print("Run status:", result.status)
          print("Messages:", result.messages)


  asyncio.run(main())

  For an approval-aware application:

  result = await app.orchestrator.run(run.id)

  if result.status is RunStatus.AWAITING_INPUT:
      approval = await app.orchestrator.pending_approval(run.id)

      user_allowed = ask_user(approval)

      await app.orchestrator.resolve_approval(
          run.id,
          approval.call_id,
          approved=user_allowed,
      )

      result = await app.orchestrator.resume(run.id)

  ## 14. Process exit codes

  The CLI distinguishes run results from process errors:

   Condition                                      Exit code
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━
   Run completed successfully                             0
  ─────────────────────────────────────────────  ───────────
   Run failed, cancelled, or budget exhausted             1
  ─────────────────────────────────────────────  ───────────
   Invalid configuration or command-line input            2
  ─────────────────────────────────────────────  ───────────
   Approval is required                                   3

  The final assistant response is written to stdout. Operational information such as run_id, status, and errors is written to stderr.

  ## Complete flow

  User command
      ↓
  CLI main()
      ↓
  Build application
      ↓
  Load configuration
      ↓
  Create model gateway, tools, and store
      ↓
  Create durable run
      ↓
  Start orchestrator
      ↓
  Call model
      ↓
  Model returns text or tool call
      ├── Text
      │     ↓
      │   Complete run
      │     ↓
      │   Print final answer
      │
      └── Tool call
            ↓
          Validate arguments
            ↓
          Check policy and guardrails
            ├── Denied
            │     ↓
            │   Return policy error to model
            │
            ├── Approval required
            │     ↓
            │   Persist paused run
            │     ↓
            │   Exit with code 3
            │
            └── Allowed
                  ↓
                Execute tool
                  ↓
                Persist result
                  ↓
                Call model again
 