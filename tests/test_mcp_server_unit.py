import io
import json
import logging
import pytest
from unittest.mock import MagicMock, patch

from orchestra.mcp_server import _handle_tool_call, _handle_request
from orchestra.engine.mcp_registry import McpRegistry, _auto_wrap

# ---------------------------------------------------------------------------
# Pre-existing tests (kept intact)
# ---------------------------------------------------------------------------

def test_handle_request_initialize():
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    res = _handle_request(req)
    assert res["protocolVersion"] == "2024-11-05"
    assert res["serverInfo"]["name"] == "orchestra-mcp"


@patch("orchestra.mcp_server.REGISTRY.invoke")
def test_handle_tool_call_success(mock_invoke):
    mock_invoke.return_value = {"content": [{"type": "text", "text": "ok"}], "isError": False}

    with patch("orchestra.engine.policy_engine.PolicyEngine.evaluate_tool") as mock_eval:
        mock_eval.return_value = MagicMock(action="allow")
        res = _handle_tool_call("some_tool", {"arg": 1})
        assert res["content"][0]["text"] == "ok"
        mock_invoke.assert_called_with("some_tool", {"arg": 1})


def test_handle_tool_call_unknown():
    with patch("orchestra.engine.policy_engine.PolicyEngine.evaluate_tool") as mock_eval:
        mock_eval.return_value = MagicMock(action="allow")
        # Registry will return None for unknown tool
        res = _handle_tool_call("unknown_tool", {})
        assert "Unknown tool" in res["content"][0]["text"]


# ---------------------------------------------------------------------------
# Task 25: New tests (TDD — written before implementation)
# ---------------------------------------------------------------------------

# 1. tools/list: returns ALL registered tools (>= 11); 7 core names present;
#    every descriptor has name, non-empty description, inputSchema with type:object
def test_tools_list_advertises_all_registered_tools():
    import orchestra.mcp_server  # ensure side-effect registrations run  # noqa: F401
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    res = _handle_request(req)
    tools = res["tools"]
    assert isinstance(tools, list)
    assert len(tools) >= 11, f"Expected >= 11 tools, got {len(tools)}"
    names = {t["name"] for t in tools}
    core = {
        "orchestra_run",
        "orchestra_submit_run",
        "orchestra_get_run",
        "orchestra_list_runs",
        "orchestra_get_logs",
        "orchestra_wait_for_run",
        "orchestra_cancel_run",
    }
    assert core <= names, f"Missing core tools: {core - names}"
    for t in tools:
        assert "name" in t
        assert t.get("description"), f"Tool {t['name']} has empty description"
        schema = t.get("inputSchema", {})
        assert schema.get("type") == "object", f"Tool {t['name']} inputSchema type != object"


# 2. Explicit schema accuracy: orchestra_submit_run requires "task"; orchestra_get_run requires "run_id"
def test_explicit_schema_accuracy():
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
    res = _handle_request(req)
    tools_by_name = {t["name"]: t for t in res["tools"]}

    submit_schema = tools_by_name["orchestra_submit_run"]["inputSchema"]
    assert "task" in submit_schema.get("required", []), "orchestra_submit_run should require 'task'"

    get_schema = tools_by_name["orchestra_get_run"]["inputSchema"]
    assert "run_id" in get_schema.get("required", []), "orchestra_get_run should require 'run_id'"


# 3. Unknown method WITH id -> error response code -32601
def test_unknown_method_with_id_returns_error():
    import orchestra.mcp_server as mcp
    msg_in = {"jsonrpc": "2.0", "id": 99, "method": "resources/list", "params": {}}
    out_buf = io.StringIO()
    in_buf = io.StringIO(json.dumps(msg_in) + "\n")

    with patch.object(mcp, "_recv_message", side_effect=[msg_in, None]):
        written: list[dict] = []

        def fake_send(message: dict) -> None:
            written.append(message)

        with patch.object(mcp, "_send_message", side_effect=fake_send):
            mcp.main()

    assert len(written) == 1, f"Expected exactly 1 message, got {len(written)}: {written}"
    response = written[0]
    assert "error" in response
    assert response["error"]["code"] == -32601
    assert response["id"] == 99


# 4. Notification (no id, unknown method) -> NO response written
def test_notification_no_response():
    import orchestra.mcp_server as mcp
    msg_in = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}

    with patch.object(mcp, "_recv_message", side_effect=[msg_in, None]):
        written: list[dict] = []

        def fake_send(message: dict) -> None:
            written.append(message)

        with patch.object(mcp, "_send_message", side_effect=fake_send):
            mcp.main()

    assert len(written) == 0, f"Expected no messages for notification, got {len(written)}: {written}"


# 5. initialize capabilities contain "tools" and NOT "resources"
def test_initialize_capabilities_no_resources():
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    res = _handle_request(req)
    caps = res["capabilities"]
    assert "tools" in caps, "capabilities should include 'tools'"
    assert "resources" not in caps, "capabilities should NOT include 'resources'"


# 6. Auto-derived schema from raw fn: def f(a: str, b: int = 3) -> properties + required
def test_auto_derived_schema_from_fn():
    def f(a: str, b: int = 3) -> dict:
        return {"a": a, "b": b}

    reg = McpRegistry()
    reg.register("test_f", f, description="test fn")
    descriptors = reg.list_tools()
    assert len(descriptors) == 1
    schema = descriptors[0]["inputSchema"]
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["a"]["type"] == "string"
    assert props["b"]["type"] == "integer"
    assert "a" in schema["required"]
    assert "b" not in schema["required"]


# 7. _auto_wrap unknown-arg warning: extra key -> warning logged, call succeeds, key ignored
def test_auto_wrap_unknown_arg_warning(caplog):
    def f(x: str) -> dict:
        return {"x": x}

    wrapped = _auto_wrap(f)
    with caplog.at_level(logging.WARNING, logger="orchestra.engine.mcp_registry"):
        result = wrapped({"x": "hello", "unknown_key": "should_warn"})
    assert result == {"x": "hello"}
    warnings = [r for r in caplog.records if "unknown" in r.message.lower() or "ignoring" in r.message.lower()]
    assert len(warnings) >= 1, f"Expected warning about unknown args, got records: {caplog.records}"
