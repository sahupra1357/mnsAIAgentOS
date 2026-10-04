---
name: tool-platform
description: The pluggable, persistent tool platform — the durable tool catalog, config-driven tool sources (Python modules, declarative HTTP tools, the built-in web search), the stdlib MCP client (stdio and streamable HTTP), MCP schema normalization and trust, untrusted-output handling, and dynamic tool loading (search_tools / load_tools) for large catalogs. Use for any Phase 7 item or when adding a tool source.
---

# Tool Platform

Tools come from *sources* declared in configuration. The catalog of what each source
provides is **persisted**; handlers are code, so they are re-attached from their source at
startup. A run records the exact tool versions it used.

## Model

- **ToolSource** (`tools/sources.py`): a named provider of tools.
  - `python`: `module = "pkg.mod:register"`. The function receives a `ToolRegistrar`
    and registers `(ToolSpec, handler)` pairs.
  - `http`: a declarative JSON-over-HTTP tool: URL, method, headers from env or secrets,
    input schema. The response is returned as the tool output.
  - `web_search`: the built-in Tavily search. The key comes from `api_key_env`, never
    from config text.
  - `mcp`: see below.
- **ToolCatalog** (`tools/catalog.py`): the table `tool_catalog` in the run store
  (SQLite and Postgres). It holds one row per tool: `name, source, source_kind, spec_json,
  spec_sha256, trust, enabled, first_seen, last_seen`.
  - **Startup sync:** `sync()` upserts what the sources provide.
  - **Missing tools:** a tool that disappears from its source is marked unavailable. Its
    row is never deleted, so old runs still resolve their tool names.
  - **Enabled flag:** the operator's `enabled` flag (CLI `tools enable|disable`) persists
    across restarts and wins over config.
- **Registration order:** built-ins first (example or coding tools, memory, artifacts,
  `load_skill`), then sources in config order. Duplicate names are a `ConfigError`.
- **Per-run record:** `RUN_CREATED.versions["tools"]` maps each offered tool to its
  `spec_sha256`. `tools_sha256` is unchanged.

## MCP (`mcp/`, stdlib only)

- **Transports:**
  - `stdio`: an `asyncio` subprocess, one JSON-RPC message per line.
  - `http`: streamable HTTP. It POSTs JSON-RPC and accepts either a JSON response or an
    SSE stream, and echoes the `Mcp-Session-Id` header.
  - Both go through a fakeable boundary.
- **Handshake:** `initialize` (protocol version `2025-06-18`), then the
  `notifications/initialized` notification. `tools/list` follows `nextCursor`, and
  `tools/call` returns content blocks: text is joined, other block kinds are summarized,
  and `isError` maps to a permanent error.
- **Naming:** tools are registered as `<server>__<tool>`, since the name may use only
  letters, digits, `_`, `-`, and `.`.
- **Schema normalization:** the result is reduced to the validator's subset. It keeps
  `type`, `properties`, `required`, `items`, `enum`, bounds, `pattern`, `description`, and
  `title`, and drops `$schema`, `default`, `format`, `examples`, and similar keywords.
  - `anyOf`/`oneOf` of a type and `null` becomes that type.
  - A tool whose schema still fails is skipped and reported, never registered half-valid.
- **Risk:** it defaults to `high_risk`, so every call asks for approval.
  - `read_only = [...]` lists tools that run freely.
  - `trust_annotations = true` accepts the server's `readOnlyHint`.
- **Output:** MCP output is `untrusted` (below).
- **Lifecycle:** servers start lazily on the first call, so `build_application` stays
  synchronous and the catalog provides the specs.
  - `await app.sync_tools()` (CLI `tools sync`, and at `serve`/`worker` start) refreshes
    the catalog from live servers.
  - `app.close()` stops them.
- **Secrets:** `env = ["NAME"]` passes named environment variables through; values never
  appear in config or events.

## Untrusted output

`ToolSpec.trust` is `"trusted"` or `"untrusted"`; the default is trusted. Untrusted results
(web search, HTTP tools, MCP) are wrapped when appended to the conversation:
`<untrusted-tool-output tool="…">…</untrusted-tool-output>`. The system context then
states once that content inside these tags is data, never instructions. Wrapping happens
in the orchestrator, where the tool message is built, so every source is covered.

## Dynamic loading (`[tool_loading]`)

- **Modes:** `mode = "all"` (the default, today's behaviour) or `"dynamic"`. In `dynamic`,
  the model sees:
  - the `core` tools, which are always offered;
  - the tools loaded so far in this run, in load order (append-only, so prefix caches
    keep hitting);
  - the meta-tools `search_tools(query, limit)` and `load_tools(names)`.
- **Search:** `search_tools` ranks catalog entries with SQLite FTS5 (BM25) over name and
  description, falling back to token overlap when FTS5 is unavailable. It returns names,
  descriptions, and risk, never the full schemas.
- **Loading:** `load_tools` validates the names, enforces `max_loaded`, and appends them to
  `RunRecord.loaded_tools`, recorded with a `TOOLS_LOADED` event. Resume therefore offers
  exactly the same tools.
- **Unloaded tools:** a call to a tool that exists but is not loaded returns an error
  telling the model to load it first. The tool is never auto-loaded, which keeps
  behaviour explicit.
- **Scale:** it is designed for hundreds of tools. The context cost is the core tools plus
  the loaded ones, not the whole catalog.

## Testing

- **MCP:** a fake MCP server is a small Python script run over stdio with
  `sys.executable`. That is a local subprocess, not the network. The HTTP transport is
  tested with a fake transport.
- **Tavily and HTTP tools:** use a fake `HttpTransport`.
- **Catalog:** tests cover persistence across two app instances on the same database, the
  enable/disable flag surviving a restart, a disappeared tool marked unavailable, and
  dynamic loading surviving resume.

## Sandboxed Python tools (Phase 10)

- **Config:** `[[tools]] kind = "python", sandbox = "container", path = "tools"` (the directory
  is required, and is mounted read-only).
- **Container:**
  - the module's directory and the `agent_harness` package are mounted read-only;
  - no network unless `network = true`;
  - read-only root, dropped capabilities, and CPU, memory, and process limits;
  - the host user id;
  - `env = [...]` passes named variables.
- **Discovery:** `tools sync` runs `python -m agent_harness.sandbox.runner list MOD:FUNC` in a
  one-shot container, and the specs are cached in the catalog (like MCP).
- **Calls:** each call runs `sandbox_runner call MOD:FUNC TOOL` in the **run's warm container**,
  with the arguments on stdin and `{"ok": ...}` or `{"error": ...}` on stdout.
- **Timeouts:** a timeout discards the container.
- **Testing:** use a fake docker invoker that runs the inner command locally (see
  `tests/test_sandbox_containers.py`).

## Phase 14: MCP resources, prompts, list_changed; web_fetch

- **Config:** `McpServerConfig.resources` / `.prompts` (default off).
- **Tools:** when the server announces the capability, `extra_specs` adds read-only, untrusted
  `NAME__mcp_list_resources`, `NAME__mcp_read_resource`, `NAME__mcp_list_prompts`, and
  `NAME__mcp_get_prompt`.
- **Discovery:** skips `tools/list` for servers without the tools capability.
- **Notifications:** connections deliver them through `on_notification` (stdio read loop; HTTP
  messages without an id in a response). `*/list_changed` sets `source.stale` and calls
  `on_tools_changed`, which the application sets to `sync_source`.
- **`web_fetch`** (`WebFetchSource`):
  - follows redirects itself, re-checking `_enforce_host` and the private-address rule per
    hop;
  - reads at most `max_bytes`, accepts only text, HTML, JSON, or XML, and extracts HTML text
    with `html.parser`;
  - tests inject `fetch` and `resolve`. The DNS check runs before connecting, so DNS
    rebinding is not covered.
