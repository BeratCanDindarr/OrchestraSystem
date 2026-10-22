"""A read-only window on any MCP server for Orchestra's agents: a proxy that offers only what is allowed.

An MCP server's tools often read and write (UnityMCP's manage_scene does get_hierarchy and
save), and Claude's permissions work per tool. The proxy starts the server, offers only the
allowed tools, narrows a tool allowed for some actions to those actions, and refuses every
other call before it reaches the server.

    [mcp_read.memory]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-memory"]
    allow = ["read_graph", "search_nodes", "open_nodes"]   # "tool", or "tool:action1|action2"

Started by the Claude CLI as `python -m orchestra.mcp_read --server memory` (the server is read
from config, so a key in its env never appears on a command line), or with a JSON spec
`{"upstream": {...}, "allow": [...]}` (tests, and the Unity proxy).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Optional

from orchestra import config

Allow = dict[str, Optional[frozenset]]   # tool -> None (every call) or the actions it may do


def parse_allow(rules: list[str]) -> Allow:
    allow: Allow = {}
    for rule in rules:
        tool, _, actions = str(rule).partition(":")
        allow[tool.strip()] = frozenset(a.strip() for a in actions.split("|") if a.strip()) if actions else None
    return allow


def configured_servers() -> dict[str, dict]:
    """The `[mcp_read.<name>]` servers that can be given: a command and something allowed."""
    return {name: spec for name, spec in (config.section("mcp_read") or {}).items()
            if isinstance(spec, dict) and (spec.get("command") or spec.get("url")) and spec.get("allow")}


def spec_for(argv: list[str]) -> tuple[dict, Allow]:
    """(upstream, allow) from `--server <name>` or from one JSON spec."""
    if argv[:1] == ["--server"] and len(argv) == 2:
        spec = configured_servers().get(argv[1])
        if spec is None:
            raise SystemExit(f"[mcp_read.{argv[1]}] is not configured (it needs command and allow)")
        if spec.get("url"):   # a remote server (streamable HTTP)
            return ({"url": str(spec["url"]), "headers": {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}},
                    parse_allow(spec["allow"]))
        upstream = {"command": str(spec["command"]), "args": [str(a) for a in spec.get("args", [])],
                    "env": {str(k): str(v) for k, v in (spec.get("env") or {}).items()}}
        return upstream, parse_allow(spec["allow"])
    if len(argv) == 1:
        spec = json.loads(argv[0])
        return spec["upstream"], parse_allow(spec["allow"])
    raise SystemExit("usage: python -m orchestra.mcp_read --server <name> | '<{\"upstream\": ..., \"allow\": [...]}>'")


def allowed(name: str, arguments: dict, allow: Allow) -> bool:
    if name not in allow:
        return False
    actions = allow[name]
    return actions is None or str(arguments.get("action", "")) in actions


def narrowed(tool, allow: Allow):
    """The tool as the agents see it, or None when it is not allowed."""
    if tool.name not in allow:
        return None
    actions = allow[tool.name]
    if actions is None:
        return tool
    schema = json.loads(json.dumps(tool.inputSchema or {}))
    action = schema.setdefault("properties", {}).setdefault("action", {"type": "string"})
    offered = set(action.get("enum") or actions)   # only what this server version has
    action["enum"] = sorted(actions & offered)
    return tool.model_copy(update={"inputSchema": schema, "description": f"Read-only (through Orchestra): {tool.description or ''}"})


async def serve(upstream: dict, allow: Allow, *, label: str = "orchestra-read") -> None:
    import mcp.types as types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    if upstream.get("url"):
        from mcp.client.streamable_http import streamablehttp_client

        transport = streamablehttp_client(upstream["url"], headers=upstream.get("headers") or None)
    else:
        transport = stdio_client(StdioServerParameters(command=upstream["command"], args=upstream.get("args", []),
                                                       env={**os.environ, **upstream.get("env", {})}))
    async with transport as streams, ClientSession(streams[0], streams[1]) as server_side:
        await server_side.initialize()
        tools = [tool for tool in (narrowed(t, allow) for t in (await server_side.list_tools()).tools) if tool is not None]
        server = Server(label)

        @server.list_tools()
        async def _list_tools() -> list:
            return tools

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict | None):
            arguments = arguments or {}
            if not allowed(name, arguments, allow):
                return types.CallToolResult(isError=True, content=[types.TextContent(
                    type="text", text=f"{name} {arguments.get('action', '')} is not allowed: Orchestra gives "
                                      "agents read-only access to this server.")])
            return await server_side.call_tool(name, arguments)

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


def main() -> None:
    upstream, allow = spec_for(sys.argv[1:])
    asyncio.run(serve(upstream, allow))


if __name__ == "__main__":
    main()
