"""Tests for cost monitoring and reporting."""
import json
from unittest.mock import MagicMock, patch

import pytest

from orchestra.models import OrchestraRun, AgentRun, RunStatus, AgentStatus
from orchestra.monitoring import (
    AgentCost,
    CostReport,
    extract_cost_report,
    format_cost_report_for_broadcast,
    broadcast_cost_report,
)
from orchestra.monitoring.cost_monitor import _check_budget


@pytest.fixture
def sample_completed_run():
    """Create a sample completed run with cost data."""
    run = OrchestraRun(mode="dual", task="Test task")
    run.status = RunStatus.COMPLETED

    agent1 = AgentRun(alias="cdx-fast", provider="openai", model="gpt-4")
    agent1.status = AgentStatus.COMPLETED
    agent1.estimated_completion_tokens = 1250
    agent1.estimated_cost_usd = 0.025

    agent2 = AgentRun(alias="gmn-pro", provider="google", model="gemini")
    agent2.status = AgentStatus.COMPLETED
    agent2.estimated_completion_tokens = 2000
    agent2.estimated_cost_usd = 0.040

    run.agents = [agent1, agent2]
    run.total_cost_usd = 0.065

    return run


def test_extract_cost_report(sample_completed_run):
    """Test extracting cost report from run."""
    report = extract_cost_report(sample_completed_run)

    assert report.run_id == sample_completed_run.run_id
    assert report.total_cost_usd == 0.065
    assert report.num_agents == 2
    assert report.mode == "dual"
    assert len(report.agent_costs) == 2

    # Check first agent
    cost1 = report.agent_costs[0]
    assert cost1.alias == "cdx-fast"
    assert cost1.model == "gpt-4"
    assert cost1.estimated_tokens == 1250
    assert cost1.estimated_cost_usd == 0.025
    assert cost1.status == "completed"


def test_extract_cost_report_empty_agents():
    """Test cost report with no agents."""
    run = OrchestraRun(mode="ask", task="Test")
    run.status = RunStatus.COMPLETED
    run.agents = []
    run.total_cost_usd = 0.0

    report = extract_cost_report(run)

    assert report.total_cost_usd == 0.0
    assert report.num_agents == 0
    assert len(report.agent_costs) == 0


def test_format_cost_report_for_broadcast(sample_completed_run):
    """Test formatting cost report for WebSocket broadcast."""
    report = extract_cost_report(sample_completed_run)
    payload = format_cost_report_for_broadcast(report)

    assert payload["type"] == "cost_report"
    assert payload["run_id"] == sample_completed_run.run_id
    assert payload["mode"] == "dual"
    assert payload["total_cost_usd"] == 0.065
    assert payload["num_agents"] == 2
    assert len(payload["agent_breakdown"]) == 2

    # Check agent breakdown structure
    agent1_breakdown = payload["agent_breakdown"][0]
    assert agent1_breakdown["alias"] == "cdx-fast"
    assert agent1_breakdown["model"] == "gpt-4"
    assert agent1_breakdown["tokens"] == 1250
    assert agent1_breakdown["cost_usd"] == 0.025
    assert agent1_breakdown["status"] == "completed"


def test_format_cost_report_precision():
    """Test that cost values are properly rounded to 6 decimals."""
    run = OrchestraRun(mode="ask", task="Test")
    run.status = RunStatus.COMPLETED
    run.total_cost_usd = 0.0123456789

    agent = AgentRun(alias="test", provider="test", model="test")
    agent.status = AgentStatus.COMPLETED
    agent.estimated_cost_usd = 0.0045123789
    agent.estimated_completion_tokens = 100

    run.agents = [agent]

    report = extract_cost_report(run)
    payload = format_cost_report_for_broadcast(report)

    # Check that costs are rounded to 6 decimals
    assert payload["total_cost_usd"] == round(0.0123456789, 6)
    assert payload["agent_breakdown"][0]["cost_usd"] == round(0.0045123789, 6)


def test_broadcast_cost_report(sample_completed_run):
    """Test broadcasting cost report calls broadcast function."""
    mock_broadcast = MagicMock()

    broadcast_cost_report(sample_completed_run, mock_broadcast)

    # Verify broadcast was called
    assert mock_broadcast.called
    call_payload = mock_broadcast.call_args[0][0]

    # Verify payload structure
    assert call_payload["type"] == "cost_report"
    assert call_payload["run_id"] == sample_completed_run.run_id
    assert call_payload["total_cost_usd"] == 0.065


def test_broadcast_cost_report_skips_zero_cost():
    """Test that zero-cost runs are not broadcast."""
    run = OrchestraRun(mode="ask", task="Test")
    run.status = RunStatus.COMPLETED
    run.agents = []
    run.total_cost_usd = 0.0

    mock_broadcast = MagicMock()

    broadcast_cost_report(run, mock_broadcast)

    # Verify broadcast was NOT called for zero cost
    assert not mock_broadcast.called


def test_broadcast_cost_report_skips_none_run():
    """Test that None run is safely skipped."""
    mock_broadcast = MagicMock()

    broadcast_cost_report(None, mock_broadcast)

    # Verify broadcast was NOT called
    assert not mock_broadcast.called


def test_cost_report_json_serializable(sample_completed_run):
    """Test that cost report can be JSON serialized."""
    report = extract_cost_report(sample_completed_run)
    payload = format_cost_report_for_broadcast(report)

    # Should not raise
    json_str = json.dumps(payload)
    assert isinstance(json_str, str)

    # Should be able to deserialize
    parsed = json.loads(json_str)
    assert parsed["type"] == "cost_report"
    assert parsed["total_cost_usd"] == 0.065


class TestCheckBudget:
    """Tests for real-time budget enforcement (_check_budget)."""

    def test_check_budget_unlimited(self):
        """Budget of 0 means unlimited — should always pass."""
        with patch("orchestra.monitoring.cost_monitor.config.execution_config", return_value={"max_budget_usd": 0}):
            run = OrchestraRun(mode="dual", task="test")
            agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
            agent.status = AgentStatus.COMPLETED
            agent.estimated_cost_usd = 10.0
            run.agents = [agent]

            ok, reason = _check_budget(run)
            assert ok is True
            assert reason == ""

    def test_check_budget_within_limit(self):
        """Budget above total cost should pass."""
        with patch("orchestra.monitoring.cost_monitor.config.execution_config", return_value={"max_budget_usd": 5.0}):
            run = OrchestraRun(mode="dual", task="test")
            agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
            agent.status = AgentStatus.COMPLETED
            agent.estimated_cost_usd = 3.0
            run.agents = [agent]

            ok, reason = _check_budget(run)
            assert ok is True
            assert reason == ""

    def test_check_budget_exceeded(self):
        """Budget below total cost should fail."""
        with patch("orchestra.monitoring.cost_monitor.config.execution_config", return_value={"max_budget_usd": 2.0}):
            run = OrchestraRun(mode="dual", task="test")
            agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
            agent.status = AgentStatus.COMPLETED
            agent.estimated_cost_usd = 3.0
            run.agents = [agent]

            ok, reason = _check_budget(run)
            assert ok is False
            assert "Budget exceeded" in reason
            assert "$3.0000" in reason
            assert "$2.0000" in reason

    def test_check_budget_only_counts_completed(self):
        """Only COMPLETED agents should count toward budget."""
        with patch("orchestra.monitoring.cost_monitor.config.execution_config", return_value={"max_budget_usd": 2.0}):
            run = OrchestraRun(mode="dual", task="test")

            agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
            agent1.status = AgentStatus.COMPLETED
            agent1.estimated_cost_usd = 1.5

            agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro")
            agent2.status = AgentStatus.FAILED
            agent2.estimated_cost_usd = 1.0

            run.agents = [agent1, agent2]

            ok, reason = _check_budget(run)
            assert ok is True
            assert reason == ""

    def test_check_budget_multiple_agents_exceeded(self):
        """Cumulative cost across multiple completed agents can exceed budget."""
        with patch("orchestra.monitoring.cost_monitor.config.execution_config", return_value={"max_budget_usd": 2.0}):
            run = OrchestraRun(mode="dual", task="test")

            agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
            agent1.status = AgentStatus.COMPLETED
            agent1.estimated_cost_usd = 1.5

            agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro")
            agent2.status = AgentStatus.COMPLETED
            agent2.estimated_cost_usd = 1.0

            run.agents = [agent1, agent2]

            ok, reason = _check_budget(run)
            assert ok is False
            assert "$2.5000" in reason


class TestBudgetSetting:
    """One setting decides the per-run budget: the documented `[limits] max_budget_usd`."""

    @staticmethod
    def _run(cost: float) -> OrchestraRun:
        run = OrchestraRun(mode="ask", task="test")
        agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
        agent.status = AgentStatus.COMPLETED
        agent.estimated_cost_usd = cost
        run.agents = [agent]
        return run

    def test_the_documented_limit_is_enforced(self, monkeypatch):
        from orchestra import config
        monkeypatch.setitem(config._cfg, "limits", {"max_budget_usd": 0.01})
        monkeypatch.setitem(config._cfg, "execution", {})

        ok, reason = _check_budget(self._run(50.0))

        assert ok is False
        assert "$0.0100" in reason

    def test_the_legacy_execution_key_still_works(self, monkeypatch):
        from orchestra import config
        monkeypatch.setitem(config._cfg, "limits", {})
        monkeypatch.setitem(config._cfg, "execution", {"max_budget_usd": 2.0})

        assert config.max_budget_usd() == 2.0
        assert _check_budget(self._run(3.0))[0] is False

    def test_limits_wins_over_the_legacy_key(self, monkeypatch):
        from orchestra import config
        monkeypatch.setitem(config._cfg, "limits", {"max_budget_usd": 1.0})
        monkeypatch.setitem(config._cfg, "execution", {"max_budget_usd": 9.0})

        assert config.max_budget_usd() == 1.0

    def test_no_setting_means_no_limit(self, monkeypatch):
        from orchestra import config
        monkeypatch.setitem(config._cfg, "limits", {})
        monkeypatch.setitem(config._cfg, "execution", {})

        assert config.max_budget_usd() == 0.0
        assert _check_budget(self._run(50.0)) == (True, "")
