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
