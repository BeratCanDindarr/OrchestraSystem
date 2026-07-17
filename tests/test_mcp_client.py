"""Tests for the external MCP client (transport-injected JSON-RPC)."""
import pytest

from orchestra.engine import mcp_client
from orchestra.engine.mcp_client import McpClient, McpClientError, StdioTransport


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _FakeProc:
    def __init__(self, stdout_lines):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class FakeTransport:
    """Records sent messages and replays a scripted recv sequence."""

    def __init__(self, recv_script):
        self.sent = []
        self._recv = list(recv_script)
        self.closed = False

    def send(self, obj):
        self.sent.append(obj)

    def recv(self, timeout=None):
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


def test_stdio_transport_send_and_recv(monkeypatch):
    proc = _FakeProc(['{"id": 1, "result": {"ok": true}}\n'])
    monkeypatch.setattr(mcp_client.subprocess, "Popen", lambda *a, **k: proc)
    t = StdioTransport(["server"])
    t.start()
    t.send({"a": 1})
    assert proc.stdin.written[0] == '{"a": 1}\n'
    assert t.recv() == {"id": 1, "result": {"ok": True}}


def test_stdio_transport_recv_eof(monkeypatch):
    proc = _FakeProc([])  # readline() -> "" (EOF)
    monkeypatch.setattr(mcp_client.subprocess, "Popen", lambda *a, **k: proc)
    t = StdioTransport(["server"])
    t.start()
    assert t.recv() is None


def test_from_command_wires_stdio_transport(monkeypatch):
    proc = _FakeProc(['{"id": 1, "result": {"tools": [{"name": "ext"}]}}\n'])
    monkeypatch.setattr(mcp_client.subprocess, "Popen", lambda *a, **k: proc)
    client = McpClient.from_command(["server", "--flag"])
    assert client.list_tools() == [{"name": "ext"}]


def test_connect_from_config_uses_configured_command(monkeypatch):
    from orchestra import config
    monkeypatch.setattr(config, "mcp_client_servers", lambda: {"fs": ["cmd", "--x"]})
    seen = {}
    monkeypatch.setattr(McpClient, "from_command", classmethod(lambda cls, command: seen.setdefault("cmd", command)))
    mcp_client.connect_from_config("fs")
    assert seen["cmd"] == ["cmd", "--x"]


def test_connect_from_config_unknown_name_raises(monkeypatch):
    from orchestra import config
    monkeypatch.setattr(config, "mcp_client_servers", lambda: {})
    with pytest.raises(McpClientError):
        mcp_client.connect_from_config("nope")


def test_config_default_empty():
    from orchestra import config
    assert config.mcp_client_servers() == {}


def test_recv_timeout_raises_instead_of_deadlocking():
    """Regression: a subprocess that accepts a request but never answers must NOT
    hang the caller forever — recv is bounded by a timeout."""
    import sys
    transport = StdioTransport([sys.executable, "-c", "import time; time.sleep(30)"])
    transport.start()
    try:
        with pytest.raises(McpClientError):
            transport.recv(timeout=0.3)
    finally:
        transport.close()


def test_client_request_timeout_propagates():
    class _HangingTransport:
        def send(self, obj):
            pass

        def recv(self, timeout=None):
            raise McpClientError("timed out")

        def close(self):
            pass

    with pytest.raises(McpClientError):
        McpClient(_HangingTransport(), timeout=0.1).list_tools()


def test_close_reaps_with_terminate_then_wait(monkeypatch):
    proc = _FakeProc([])
    monkeypatch.setattr(mcp_client.subprocess, "Popen", lambda *a, **k: proc)
    t = StdioTransport(["server"])
    t.start()
    t.close()
    assert proc.terminated is True


def test_from_command_wraps_start_failure():
    def _boom(*a, **k):
        raise FileNotFoundError("no such command")

    import unittest.mock as m
    with m.patch.object(mcp_client.subprocess, "Popen", _boom):
        with pytest.raises(McpClientError):
            McpClient.from_command(["does-not-exist"])
