"""Tests for the external MCP client (transport-injected JSON-RPC)."""
import pytest

from orchestra.engine.mcp_client import McpClient, McpClientError


class FakeTransport:
    """Records sent messages and replays a scripted recv sequence."""

    def __init__(self, recv_script):
        self.sent = []
        self._recv = list(recv_script)
        self.closed = False

    def send(self, obj):
        self.sent.append(obj)

    def recv(self):
        return self._recv.pop(0) if self._recv else None

    def close(self):
        self.closed = True


def test_list_tools_returns_tools():
    t = FakeTransport([{"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "t"}]}}])
    client = McpClient(t)
    tools = client.list_tools()
    assert tools == [{"name": "t"}]
    assert t.sent[0]["method"] == "tools/list"
    assert t.sent[0]["id"] == 1


def test_call_tool_sends_params_and_returns_result():
    t = FakeTransport([{"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}])
    client = McpClient(t)
    result = client.call_tool("t", {"a": 1})
    assert result == {"ok": True}
    assert t.sent[0]["method"] == "tools/call"
    assert t.sent[0]["params"] == {"name": "t", "arguments": {"a": 1}}


def test_ignores_notifications_and_mismatched_ids():
    t = FakeTransport([
        {"jsonrpc": "2.0", "method": "notify"},          # notification (no id) → skip
        {"jsonrpc": "2.0", "id": 99, "result": {"x": 1}},  # wrong id → skip
        {"jsonrpc": "2.0", "id": 1, "result": {"ok": 1}},  # our response
    ])
    assert McpClient(t).call_tool("t") == {"ok": 1}


def test_error_response_raises():
    t = FakeTransport([{"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "boom"}}])
    with pytest.raises(McpClientError):
        McpClient(t).list_tools()


def test_closed_connection_raises():
    t = FakeTransport([])  # recv immediately returns None
    with pytest.raises(McpClientError):
        McpClient(t).list_tools()


def test_context_manager_closes():
    t = FakeTransport([{"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}])
    with McpClient(t) as client:
        client.list_tools()
    assert t.closed is True
