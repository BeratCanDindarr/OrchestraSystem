"""Tests for the MCP Fast Lane dispatcher."""
from __future__ import annotations

from unittest.mock import patch

from orchestra.engine.mcp_fastlane import FastLaneDispatcher
from orchestra.engine.mcp_registry import McpRegistry


class TestFastLaneDispatcher:
    def test_is_available_when_registered(self) -> None:
        reg = McpRegistry()
        reg.register("test_tool", lambda args: {"ok": True}, fastlane=True)
        fl = FastLaneDispatcher(registry=reg)
        assert fl.is_available("test_tool") is True

    def test_is_available_denied_by_policy(self) -> None:
        reg = McpRegistry()
        reg.register("test_tool", lambda args: {"ok": True}, fastlane=True)
        fl = FastLaneDispatcher(registry=reg)
        with patch(
            "orchestra.engine.mcp_fastlane.PolicyEngine.from_default"
        ) as mock_pe:
            from orchestra.engine.policy_engine import PolicyDecision
            mock_pe.return_value.evaluate_tool.return_value = PolicyDecision(
                "deny", "denied", "test"
            )
            assert fl.is_available("test_tool") is False

    def test_dispatch_success(self) -> None:
        reg = McpRegistry()
        reg.register("test_tool", lambda args: {"ok": True}, fastlane=True)
        fl = FastLaneDispatcher(registry=reg)
        result = fl.dispatch("test_tool", {})
        assert result["isError"] is False
        assert "ok" in result["structuredContent"]

    def test_dispatch_unknown_tool(self) -> None:
        reg = McpRegistry()
        fl = FastLaneDispatcher(registry=reg)
        result = fl.dispatch("missing", {})
        assert result["isError"] is True
        assert "not available" in result["content"][0]["text"]

    def test_dispatch_fastlane_false(self) -> None:
        reg = McpRegistry()
        reg.register("test_tool", lambda args: {"ok": True}, fastlane=False)
        fl = FastLaneDispatcher(registry=reg)
        result = fl.dispatch("test_tool", {})
        assert result["isError"] is True

    def test_dispatch_blocked_by_autonomy_policy(self) -> None:
        reg = McpRegistry()
        reg.register("git_push", lambda args: {"ok": True}, fastlane=True)
        fl = FastLaneDispatcher(registry=reg)
        from orchestra.engine.autonomy_policy import AutonomyPolicy
        with patch.object(AutonomyPolicy, "from_config") as mock_policy:
            mock_policy.return_value.check_before_tool_call.return_value = (
                False,
                "privileged_tool: 'git_push' matched privileged action 'git push'",
            )
            result = fl.dispatch("git_push", {"repo": "origin"})
            assert result["isError"] is True
            assert "blocked by autonomy policy" in result["content"][0]["text"]
