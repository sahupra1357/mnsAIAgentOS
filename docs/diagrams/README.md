# Architecture diagrams

Each diagram comes in three formats:
- `.svg`: a static image. Opens in any browser or on GitHub.
- `.animated.svg`: a self-running walkthrough. A token follows a run through the diagram, the
  current step is outlined, and a caption explains it. It loops and needs no script, so it
  plays in browsers and on GitHub.
- `.excalidraw`: editable. Open it at <https://excalidraw.com> (menu → Open) or with the
  VS Code Excalidraw extension.

| Diagram | Static | Animated | What it shows |
|---|---|---|---|
| End-to-end | [svg](01-end-to-end.svg) | [animated](01-end-to-end.animated.svg) (23 steps) | Entry points, composition root, the orchestrator loop, model gateway and servers, durable state, observability |
| Permissions and approval | [svg](02-permissions-approval.svg) | [animated](02-permissions-approval.animated.svg) (18 steps) | How a tool call is decided (hooks, deny/mode/ask/allow rules, risk policy), human approval, plan mode |
| Subagents | [svg](03-subagents.svg) | [animated](03-subagents.animated.svg) (11 steps) | The `agent` tool: limits, child run, pause and resume, result and usage roll-up |
| Context summarization | [svg](04-summarization.svg) | [animated](04-summarization.animated.svg) (13 steps) | Context trimming and persisted summarization, the `pre_compact` hook, failure fallback |
| Tool platform | [svg](05-tool-platform.svg) | [animated](05-tool-platform.animated.svg) (13 steps) | Tool sources (Python, HTTP, web search, MCP), the persistent catalog, operator switches, dynamic loading, untrusted output |
| Plan and execute | [svg](06-plan-and-execute.svg) | [animated](06-plan-and-execute.animated.svg) (16 steps) | The orchestrator-owned plan: validation, parallel delegated steps, main-model steps, replanning, finishing |
| API access and tenancy | [svg](07-api-access.svg) | [animated](07-api-access.animated.svg) (12 steps) | Keys, principals, roles, rate limit, tenant isolation, tenant and user quotas, queue and workers, redacted store, health |
| Context overflow | [svg](08-context-overflow.svg) | [animated](08-context-overflow.animated.svg) (12 steps) | When a server rejects a prompt as too long: detection, a smaller persisted cap, retry, or failure |
| Limits and deadlines | [svg](09-limits.svg) | [animated](09-limits.animated.svg) (14 steps) | Where each limit applies: rate limit, quotas, step checks, output cap, tool calls per reply, approval and question deadlines |

## Regenerating

All nine diagrams are defined in `tools/generate.py`: boxes, arrows, tour steps, and
captions. Edit a diagram function there, then run this from the repository root:

```bash
python docs/diagrams/tools/generate.py docs/diagrams build/tours.json docs/diagrams/tools/page_template.html
```

This rewrites the `.svg`, `.animated.svg`, and `.excalidraw` files. It also writes
`build/tours.html`, the interactive walkthrough page, with Play, Step, and speed
controls.

