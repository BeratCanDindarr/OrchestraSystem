"""Any MCP server can be given to reading agents through a proxy that offers only what is allowed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

FAKE_UPSTREAM = str(Path(__file__).parent / "fixtures" / "fake_unity_mcp.py")


def _proxy(allow: list[str]) -> StdioServerParameters:
    spec = json.dumps({"upstream": {"command": sys.executable, "args": [FAKE_UPSTREAM]}, "allow": allow})
    return StdioServerParameters(command=sys.executable, args=["-m", "orchestra.mcp_read", spec])


def test_allow_rules_are_read_from_config_strings():
    from orchestra.mcp_read import parse_allow

    assert parse_allow(["read_graph", "manage_scene:get_active|get_hierarchy"]) == {
        "read_graph": None, "manage_scene": frozenset({"get_active", "get_hierarchy"})}


@pytest.mark.asyncio
async def test_only_allowed_tools_and_actions_are_offered_and_the_rest_is_refused():
    async with stdio_client(_proxy(["manage_scene:get_active"])) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        allowed = await client.call_tool("manage_scene", {"action": "get_active"})
        written = await client.call_tool("manage_scene", {"action": "save"})
        deleted = await client.call_tool("delete_script", {"path": "Assets/A.cs"})

    assert set(tools) == {"manage_scene"}
    assert tools["manage_scene"].inputSchema["properties"]["action"]["enum"] == ["get_active"]
    assert not allowed.isError and "upstream did manage_scene:get_active" in allowed.content[0].text
    assert written.isError and "upstream" not in written.content[0].text
    assert deleted.isError and "upstream" not in deleted.content[0].text


@pytest.mark.asyncio
async def test_a_tool_allowed_by_name_is_offered_whole():
    async with stdio_client(_proxy(["delete_script"])) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        tools = {tool.name for tool in (await client.list_tools()).tools}
        answer = await client.call_tool("delete_script", {"path": "x"})

    assert tools == {"delete_script"} and not answer.isError


def test_a_reading_claude_agent_gets_the_configured_servers_and_a_writer_does_not(monkeypatch):
    from orchestra.providers import claude

    servers = {"memory": {"command": "npx", "args": ["-y", "server-memory"], "allow": ["read_graph", "search_nodes"]}}
    profiles = {"claude": {"tools": ["Read"]}, "claude_write": {"skip_permissions": True}}
    monkeypatch.setattr(claude.config, "provider_config", lambda key: dict(profiles.get(key, {})))
    monkeypatch.setattr(claude.config, "section", lambda name: servers if name == "mcp_read" else {})
    monkeypatch.setattr(claude, "unity_mcp_config", lambda: None)

    reader = claude.ClaudeProvider().build_command("x", "sonnet")
    writer = claude.ClaudeProvider(config_key="claude_write").build_command("x", "opus")

    proxy = json.loads(reader[reader.index("--mcp-config") + 1])["mcpServers"]["memory"]
    # by name: the proxy reads the server (its env may hold a key) from config, not from argv
    assert proxy["args"] == ["-m", "orchestra.mcp_read", "--server", "memory"]
    assert "mcp__memory" in reader[reader.index("--allowed-tools") + 1].split(",")
    assert "--mcp-config" not in writer


def test_the_proxy_finds_a_named_server_in_config(monkeypatch):
    from orchestra import mcp_read

    monkeypatch.setattr(mcp_read.config, "section", lambda name: {"memory": {
        "command": "npx", "args": ["-y", "server-memory"], "env": {"KEY": "v"}, "allow": ["read_graph"]}} if name == "mcp_read" else {})

    assert mcp_read.spec_for(["--server", "memory"]) == (
        {"command": "npx", "args": ["-y", "server-memory"], "env": {"KEY": "v"}}, {"read_graph": None})
    with pytest.raises(SystemExit):
        mcp_read.spec_for(["--server", "unknown"])


def test_a_server_without_an_allow_list_is_not_given(monkeypatch):
    # Nothing allowed means nothing offered: a server is never handed over whole by omission.
    from orchestra.providers import claude

    monkeypatch.setattr(claude.config, "provider_config", lambda key: {"tools": ["Read"]})
    monkeypatch.setattr(claude.config, "section", lambda name: {"memory": {"command": "npx"}} if name == "mcp_read" else {})
    monkeypatch.setattr(claude, "unity_mcp_config", lambda: None)

    assert "--mcp-config" not in claude.ClaudeProvider().build_command("x", "sonnet")


def test_a_remote_server_is_given_by_url(monkeypatch):
    from orchestra import mcp_read

    monkeypatch.setattr(mcp_read.config, "section", lambda name: {"exa": {
        "url": "https://mcp.exa.ai/mcp", "allow": ["web_search_exa"]}} if name == "mcp_read" else {})

    assert "exa" in mcp_read.configured_servers()
    assert mcp_read.spec_for(["--server", "exa"]) == ({"url": "https://mcp.exa.ai/mcp", "headers": {}}, {"web_search_exa": None})

