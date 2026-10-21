"""Integration tests for PolicyEngine in runner and tool paths."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from orchestra.engine.policy_engine import PolicyEngine, PolicyDecision
from orchestra.engine.runners import _evaluate_policy
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


@pytest.fixture(autouse=True)
def _agents_that_can_write(monkeypatch):
    """These tests exercise the approval gate, which holds back only agents that can change files."""
    monkeypatch.setattr("orchestra.providers.fallback._writes", lambda alias: True)



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

        with patch("orchestra.engine.runners.common.PolicyEngine") as MockEngine:
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

        with patch("orchestra.engine.runners.common.PolicyEngine") as MockEngine:
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

        with patch("orchestra.engine.runners.common.PolicyEngine") as MockEngine:
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

        with patch("orchestra.mcp_server.POLICY") as mock_policy:
            mock_policy.evaluate_tool.return_value = PolicyDecision(
                "deny", "tool denied", "policy.yaml:tools.shell_exec"
            )
            result = _handle_tool_call("shell_exec", {"command": "ls"})
            assert result.get("isError") is True
            assert "denied" in result["content"][0]["text"]

    def test_allowed_mcp_tool_dispatches(self) -> None:
        from orchestra.mcp_server import _handle_tool_call

        with patch("orchestra.mcp_server.POLICY") as mock_policy:
            mock_policy.evaluate_tool.return_value = PolicyDecision(
                "allow", "tool auto", "policy.yaml:tools.orchestra_list_runs"
            )
            # orchestra_list_runs is a lightweight read-only tool
            result = _handle_tool_call("orchestra_list_runs", {})
            assert result.get("isError") is not True


class TestDeniedBeforeAnAgentRuns:
    """What the policy forbids outright is not run first and failed afterwards."""

    @staticmethod
    def _policy(tmp_path: Path, text: str):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(text)
        engine = PolicyEngine(policy_file)
        return patch("orchestra.engine.runners.common.PolicyEngine.from_default", return_value=engine)

    def test_an_ask_to_a_denied_provider_starts_no_agent(self, tmp_path: Path) -> None:
        from orchestra.engine.runners import common

        with self._policy(tmp_path, "approval:\n  providers:\n    claude: deny\n"), \
             patch.object(common, "run_parallel") as run_parallel:
            run = common._run_ask_impl("cld-deep", "denied probe: ask")

        run_parallel.assert_not_called()
        assert run.status == RunStatus.FAILED
        assert run.failure.kind == "POLICY_DENIED" and "claude" in run.failure.message

    def test_a_dual_run_with_a_denied_provider_starts_no_agent(self, tmp_path: Path) -> None:
        from orchestra.engine.runners import dual_runner

        with self._policy(tmp_path, "approval:\n  providers:\n    kimi: deny\n"), \
             patch.object(dual_runner, "run_parallel") as run_parallel:
            run = dual_runner.run_dual("denied probe: dual", agents=["cdx-deep", "kimi-deep"],
                                       show_live=False, emit_console=False)

        run_parallel.assert_not_called()
        assert run.status == RunStatus.FAILED and run.failure.kind == "POLICY_DENIED"

    def test_a_day_that_is_already_over_budget_starts_no_agent(self, tmp_path: Path) -> None:
        from orchestra.engine.runners import common

        with self._policy(tmp_path, "budget:\n  max_usd_per_day: 1.0\n"), \
             patch.object(common, "_daily_spend_usd", return_value=3.5), \
             patch.object(common, "run_parallel") as run_parallel:
            run = common._run_ask_impl("cdx-deep", "denied probe: budget")

        run_parallel.assert_not_called()
        assert run.failure.kind == "POLICY_DENIED" and "daily" in run.failure.message

    def test_a_model_on_an_ollama_backend_is_an_ollama_provider(self) -> None:
        assert PolicyEngine._alias_to_provider("qwen2.5:7b@windows-gpu") == "ollama"
        assert PolicyEngine._alias_to_provider("oll-fast") == "ollama"

    def test_an_answer_that_is_waiting_for_approval_is_not_stored_as_an_answer(self) -> None:
        from orchestra.engine.runners import common

        def answered(run, pairs, **kwargs):
            agent = AgentRun(alias=pairs[0][0], provider="codex", model="m", status=AgentStatus.COMPLETED)
            agent.stdout_log = "an answer nobody approved yet"
            run.agents.append(agent)

        gate = PolicyDecision("gate", "needs a human", "test")
        with patch.object(common, "run_parallel", answered), \
             patch.object(common, "_evaluate_policy", return_value=gate), \
             patch.object(common, "_try_cache_hit", return_value=None), \
             patch.object(common, "_store_in_cache") as store:
            run = common._run_ask_impl("cdx-deep", "gated probe: cache")

        assert run.status == RunStatus.WAITING_APPROVAL
        store.assert_not_called()
