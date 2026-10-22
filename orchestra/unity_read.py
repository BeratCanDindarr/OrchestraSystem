"""A read-only window on the Unity editor for Orchestra's agents: a proxy of UnityMCP.

UnityMCP's tools mostly read and write through one tool and an `action` (manage_scene does
get_hierarchy and save), and Claude's permissions work per tool: allowing manage_scene would
allow deleting a scene. This proxy starts UnityMCP, offers only the reading actions listed
here (everything else is refused before it reaches Unity) and narrows each schema to them.

Started by the Claude CLI as `python -m orchestra.unity_read '<upstream json>'`.
Switched on per project with `[unity] enabled = true`.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

# tool -> the actions that only read. A tool not here is not offered at all.
READ_ACTIONS: dict[str, frozenset[str]] = {
    "manage_scene": frozenset({"get_hierarchy", "get_active", "get_build_settings", "get_loaded_scenes"}),
    "manage_gameobject": frozenset({"find", "get_components", "get_component"}),
    "manage_asset": frozenset({"search", "get_info", "get_components"}),
    "manage_prefabs": frozenset({"get_info", "get_hierarchy"}),
    "manage_script": frozenset({"read"}),
    "read_console": frozenset({"get"}),
}
# Tools that only read whatever their arguments.
READ_ONLY_TOOLS = frozenset({"unity_docs", "unity_reflect", "validate_script", "get_sha",
                             "manage_script_capabilities", "get_test_job"})


def allowed(name: str, arguments: dict) -> bool:
    if name in READ_ONLY_TOOLS:
        return True
    return name in READ_ACTIONS and str(arguments.get("action", "")) in READ_ACTIONS[name]


def narrowed(tool):
    """The tool as the agents see it, or None when it does not read."""
    if tool.name in READ_ONLY_TOOLS:
        return tool
    if tool.name not in READ_ACTIONS:
        return None
    schema = json.loads(json.dumps(tool.inputSchema or {}))
    action = schema.setdefault("properties", {}).setdefault("action", {"type": "string"})
    offered = set(action.get("enum") or READ_ACTIONS[tool.name])   # only what this UnityMCP version has
    action["enum"] = sorted(READ_ACTIONS[tool.name] & offered)
    description = f"Read-only (through Orchestra): {tool.description or ''}"
    return tool.model_copy(update={"inputSchema": schema, "description": description})


def upstream_config(project_root: Optional[Path] = None) -> Optional[dict]:
    """How to start UnityMCP for this project, or None when Unity access is off or not found.

    `[unity] command` / `args` when set, else the project's UnityMCP entry in ~/.claude.json
    (each Unity project pins its own server version there).
    """
    from orchestra import config

    section = config.section("unity") if hasattr(config, "section") else {}
    if not section.get("enabled", False):
        return None
    if section.get("command"):
        return {"command": str(section["command"]), "args": [str(a) for a in section.get("args", [])]}
    root = str((project_root or config.project_root()).resolve())
    try:
        projects = json.loads((Path.home() / ".claude.json").read_text(encoding="utf-8")).get("projects") or {}
    except (OSError, ValueError):
        return None
    servers = (projects.get(root) or {}).get("mcpServers") or {}
    entry = next((srv for name, srv in servers.items() if "unity" in name.lower() and srv.get("command")), None)
    return {"command": entry["command"], "args": list(entry.get("args") or [])} if entry else None


async def _serve(upstream: dict) -> None:
    import mcp.types as types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    params = StdioServerParameters(command=upstream["command"], args=upstream.get("args", []),
                                   env={**os.environ, **upstream.get("env", {})})
    async with stdio_client(params) as (up_read, up_write), ClientSession(up_read, up_write) as unity:
        await unity.initialize()
        tools = [tool for tool in map(narrowed, (await unity.list_tools()).tools) if tool is not None]
        server = Server("orchestra-unity-read")

        @server.list_tools()
        async def _list_tools() -> list:
            return tools

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict | None):
            arguments = arguments or {}
            if not allowed(name, arguments):
                return types.CallToolResult(isError=True, content=[types.TextContent(
                    type="text", text=f"{name} {arguments.get('action', '')} is not allowed: Orchestra's Unity "
                                      "access is read-only (scene, objects, components, assets, scripts, console).")])
            return await unity.call_tool(name, arguments)

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m orchestra.unity_read '<upstream json: {\"command\": ..., \"args\": [...]}>'")
    asyncio.run(_serve(json.loads(sys.argv[1])))


if __name__ == "__main__":
    main()
