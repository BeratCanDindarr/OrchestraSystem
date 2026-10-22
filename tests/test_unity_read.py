"""Agents see Unity through a read-only proxy of UnityMCP: reads pass, every write is refused."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FAKE_UPSTREAM = str(Path(__file__).parent / "fixtures" / "fake_unity_mcp.py")


def _proxy_params() -> StdioServerParameters:
    upstream = json.dumps({"command": sys.executable, "args": [FAKE_UPSTREAM]})
    return StdioServerParameters(command=sys.executable, args=["-m", "orchestra.unity_read", upstream])


@pytest.mark.asyncio
async def test_only_reading_actions_are_offered():
    async with stdio_client(_proxy_params()) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"manage_scene"}   # delete_script only writes: not offered
    assert tools["manage_scene"].inputSchema["properties"]["action"]["enum"] == ["get_active", "get_hierarchy"]


@pytest.mark.asyncio
async def test_a_read_is_forwarded_and_a_write_is_refused_without_reaching_unity():
    async with stdio_client(_proxy_params()) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        hierarchy = await client.call_tool("manage_scene", {"action": "get_hierarchy"})
        save = await client.call_tool("manage_scene", {"action": "save"})
        delete = await client.call_tool("delete_script", {"path": "Assets/A.cs"})

    assert not hierarchy.isError and "upstream did manage_scene:get_hierarchy" in hierarchy.content[0].text
    assert save.isError and "upstream" not in save.content[0].text   # refused by the narrowed schema itself
    assert delete.isError and "upstream" not in delete.content[0].text


def test_a_reading_claude_agent_gets_unity_and_a_writing_one_does_not(monkeypatch):
    from orchestra.providers import claude

    profiles = {"claude": {"tools": ["Read", "Grep", "Glob"]}, "claude_write": {"skip_permissions": True}}
    monkeypatch.setattr(claude.config, "provider_config", lambda key: dict(profiles.get(key, {})))
    monkeypatch.setattr(claude, "unity_mcp_config", lambda: {"command": "uvx", "args": ["mcp-for-unity"]})

    reader = claude.ClaudeProvider().build_command("x", "sonnet")
    writer = claude.ClaudeProvider(config_key="claude_write").build_command("x", "opus")

    config_arg = json.loads(reader[reader.index("--mcp-config") + 1])
    assert config_arg["mcpServers"]["unity"]["args"][:2] == ["-m", "orchestra.unity_read"]
    assert "mcp__unity" in reader[reader.index("--allowed-tools") + 1]
    assert "--mcp-config" not in writer


def test_no_unity_unless_it_is_switched_on(monkeypatch):
    from orchestra.providers import claude

    monkeypatch.setattr(claude.config, "provider_config", lambda key: {"tools": ["Read"]})
    monkeypatch.setattr(claude, "unity_mcp_config", lambda: None)

    assert "--mcp-config" not in claude.ClaudeProvider().build_command("x", "sonnet")
