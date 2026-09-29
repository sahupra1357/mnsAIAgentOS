# Architecture diagrams

Each diagram comes in two formats:
- `.svg`: opens in any browser or on GitHub.
- `.excalidraw`: editable. Open it at <https://excalidraw.com> (menu → Open) or with
  the VS Code Excalidraw extension.

| Diagram | What it shows |
|---|---|
| [01-end-to-end](01-end-to-end.svg) | Entry points, composition root, the orchestrator loop, model gateway and servers, durable state, observability |
| [02-permissions-approval](02-permissions-approval.svg) | How a tool call is decided (hooks, deny/mode/ask/allow rules, risk policy), human approval, plan mode |
| [03-subagents](03-subagents.svg) | The `agent` tool: limits, child run, pause and resume, result and usage roll-up |
| [04-summarization](04-summarization.svg) | Context trimming and persisted summarization, the `pre_compact` hook, failure fallback |
