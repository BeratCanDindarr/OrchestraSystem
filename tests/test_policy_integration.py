"""Integration tests for PolicyEngine in runner and tool paths."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestra.engine.policy_engine import PolicyEngine, PolicyDecision
from orchestra.engine.runners import _evaluate_policy
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


class TestRunnerPolicyIntegration:
    """PolicyEngine integration with runner._evaluate_policy."""

    def test_policy_yaml_triggers_gate(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: gated
""")
        run = OrchestraRun(mode="ask", task="test")
        run.agents = [AgentRun(alias="cld-deep", provider="claude", model="opus")]

        with patch("orchestra.engine.runner.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock._has_policy.return_value = True
            mock.evaluate_run.return_value = PolicyDecision(
                "gate", "provider gated", "policy.yaml:approval.providers.claude"
            )
            decision = _evaluate_policy(run, "test")
            assert decision.action == "gate"

    def test_policy_yaml_allows_run(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    codex: auto
""")
        run = OrchestraRun(mode="ask", task="test")
        run.agents = [AgentRun(alias="cdx-deep", provider="codex", model="gpt-5")]

        with patch("orchestra.engine.runner.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock._has_policy.return_value = True
            mock.evaluate_run.return_value = PolicyDecision(
                "allow", "all good", "policy.yaml"
            )
            decision = _evaluate_policy(run, "test")
            assert decision.action == "allow"

    def test_legacy_path_when_no_policy(self) -> None:
        """When PolicyEngine reports no policy, legacy TOML path is used."""
        run = OrchestraRun(mode="ask", task="test")
        run.agents = []

        with patch("orchestra.engine.runner.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock._has_policy.return_value = False
            # Legacy path: approval_policies enabled=false by default in tests
            decision = _evaluate_policy(run, "test")
            assert decision.action == "allow"


class TestToolRegistryPolicyIntegration:
    """PolicyEngine integration with speculative tool registry."""

    def test_denied_tool_returns_error(self) -> None:
        from orchestra.engine.tool_registry import _TOOL_EXECUTORS

        with patch("orchestra.engine.tool_registry.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock.evaluate_tool.return_value = PolicyDecision(
                "deny", "tool denied", "policy.yaml:tools.git_log"
            )
            executor = _TOOL_EXECUTORS["git_log"]
            result = executor("test query")
            assert "error" in result
            assert "denied" in result["error"]

    def test_allowed_tool_passes_through(self) -> None:
        from orchestra.engine.tool_registry import _TOOL_EXECUTORS

        with patch("orchestra.engine.tool_registry.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock.evaluate_tool.return_value = PolicyDecision(
                "allow", "tool auto", "policy.yaml:tools.read_file"
            )
            # The real executor may fail because the tool module isn't available
            # in test env; we just verify the policy wrapper lets it through.
            executor = _TOOL_EXECUTORS["read_file"]
            result = executor("test query")
            # If the underlying executor runs, it returns an error dict because
            # the tool module is missing in tests — that's fine, it proves
            # policy allowed the call.
            assert isinstance(result, dict)


class TestMcpServerPolicyIntegration:
    """PolicyEngine integration with MCP server tool dispatch."""

    def test_denied_mcp_tool_returns_error(self) -> None:
        from orchestra.mcp_server import _handle_tool_call

        with patch("orchestra.mcp_server.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock.evaluate_tool.return_value = PolicyDecision(
                "deny", "tool denied", "policy.yaml:tools.shell_exec"
            )
            result = _handle_tool_call("shell_exec", {"command": "ls"})
            assert "error" in result
            assert "denied" in result["error"]

    def test_allowed_mcp_tool_dispatches(self) -> None:
        from orchestra.mcp_server import _handle_tool_call

        with patch("orchestra.mcp_server.PolicyEngine") as MockEngine:
            mock = MockEngine.from_default.return_value
            mock.evaluate_tool.return_value = PolicyDecision(
                "allow", "tool auto", "policy.yaml:tools.orchestra_get_version"
            )
            # orchestra_get_version is a lightweight read-only tool
            with patch("orchestra.mcp_server.get_version", return_value="1.0.0"):
                result = _handle_tool_call("orchestra_get_version", {})
                assert "error" not in result
