"""A stand-in for UnityMCP over stdio: one tool that reads and writes, one that only writes."""
from __future__ import annotations

import asyncio

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server("fake-unity")
SCENE_ACTIONS = ["create", "save", "get_hierarchy", "get_active", "close_scene"]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name="manage_scene", description="Scenes.", inputSchema={
            "type": "object", "required": ["action"],
            "properties": {"action": {"type": "string", "enum": SCENE_ACTIONS}, "name": {"type": "string"}}}),
        types.Tool(name="delete_script", description="Deletes a script.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string"}}}),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=f"upstream did {name}:{arguments.get('action', '')}")]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
