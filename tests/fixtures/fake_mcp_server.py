"""A tiny MCP server over stdio for tests (no network, stdlib only).

Tools: ``echo`` (read-only hint), ``fail`` (isError), ``structured``,
``ping_me`` (the server pings the client before answering), and
``weird_schema`` (JSON Schema features the harness must normalize). tools/list
is paginated. Set FAKE_MCP_EXTRA_TOOL=1 to add a tool, FAKE_MCP_CRASH_ON_CALL=1
to exit on the first tools/call.
"""

import json
import os
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the text back.",
        "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                        "properties": {"text": {"type": "string", "default": "hi"}},
                        "required": ["text"]},
        "annotations": {"readOnlyHint": True},
    },
    {"name": "fail", "description": "Always fails.", "inputSchema": {"type": "object"}},
    {"name": "structured", "description": "Returns structured content.", "inputSchema": {"type": "object"}},
    {"name": "ping_me", "description": "Pings the client first.", "inputSchema": {"type": "object"}},
    {
        "name": "weird_schema",
        "description": "Uses anyOf, type lists and formats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "when": {"type": "string", "format": "date-time"},
                "limit": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                "mode": {"const": "fast"},
            },
            "required": ["when", "missing_from_properties"],
        },
    },
]
if os.environ.get("FAKE_MCP_EXTRA_TOOL") == "1":
    TOOLS.append({"name": "extra", "description": "An extra tool.", "inputSchema": {"type": "object"}})


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main():
    print("fake server starting", file=sys.stderr)
    sys.stdout.write("this line is not JSON-RPC and must be ignored\n")
    sys.stdout.flush()
    pending_call = None
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if method is None:  # a response from the client (to our ping)
            if pending_call is not None:
                send({"jsonrpc": "2.0", "id": pending_call,
                      "result": {"content": [{"type": "text", "text": "pong received"}]}})
                pending_call = None
            continue
        if "id" not in message:
            continue  # notification
        mid = message["id"]
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"},
            }})
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            page = TOOLS[:2] if cursor is None else TOOLS[2:]
            result = {"tools": page}
            if cursor is None:
                result["nextCursor"] = "page2"
            send({"jsonrpc": "2.0", "id": mid, "result": result})
        elif method == "tools/call":
            if os.environ.get("FAKE_MCP_CRASH_ON_CALL") == "1":
                sys.exit(3)
            name = message["params"]["name"]
            args = message["params"].get("arguments") or {}
            if name == "echo":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"echo: {args.get('text')}"}]}})
            elif name == "fail":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "isError": True, "content": [{"type": "text", "text": "it broke"}]}})
            elif name == "structured":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "ok"}, {"type": "image", "mimeType": "image/png", "data": "x"}],
                    "structuredContent": {"answer": 42}}})
            elif name == "ping_me":
                pending_call = mid
                send({"jsonrpc": "2.0", "id": "srv-1", "method": "ping"})
            else:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "unknown tool"}})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "no such method"}})


if __name__ == "__main__":
    main()
