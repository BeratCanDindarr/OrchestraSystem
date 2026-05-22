import pytest
import json
from unittest.mock import MagicMock, patch
from orchestra.mcp_server import _handle_tool_call, _handle_request

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
