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


class TestCosting:
    """What a call cost: the provider's own figure, else its tokens at the model's prices."""

    @staticmethod
    def _priced(monkeypatch, pricing: dict) -> None:
        from orchestra import config
        monkeypatch.setitem(config._cfg, "pricing", pricing)

    def test_prompt_tokens_are_paid_for_too(self, monkeypatch):
        from orchestra.engine import costing

        self._priced(monkeypatch, {"x/model": 0.01})
        usage = costing.usage_of({"prompt_tokens": 1000, "completion_tokens": 500}, prompt="p", output="o")

        assert costing.cost_of("x/model", usage) == pytest.approx(0.015)
        assert usage.source == "reported"

    def test_a_price_table_separates_input_output_and_cached_input(self, monkeypatch):
        from orchestra.engine import costing

        self._priced(monkeypatch, {"x/model": {"input": 0.010, "output": 0.050, "cached": 0.001}})
        usage = costing.usage_of({"prompt_tokens": 1000, "completion_tokens": 1000, "cached_tokens": 400},
                                 prompt="p", output="o")

        # 600 fresh input + 400 cached input + 1000 output
        assert costing.cost_of("x/model", usage) == pytest.approx(0.006 + 0.0004 + 0.050)

    def test_what_the_provider_says_it_cost_is_what_it_cost(self, monkeypatch):
        from orchestra.engine import costing

        self._priced(monkeypatch, {"claude/opus": 99.0})
        usage = costing.usage_of({"completion_tokens": 10, "cost_usd": 0.132}, prompt="p", output="o")

        assert costing.cost_of("claude/opus", usage) == pytest.approx(0.132)

    def test_a_local_model_costs_nothing_unless_it_is_priced(self, monkeypatch):
        from orchestra.engine import costing

        self._priced(monkeypatch, {"ollama/glm-5.3:cloud": 0.003})
        usage = costing.usage_of({"prompt_tokens": 2000, "completion_tokens": 1000}, prompt="p", output="o")

        assert costing.cost_of("ollama/qwen2.5:7b", usage) == 0.0
        assert costing.cost_of("ollama/glm-5.3:cloud", usage) == pytest.approx(0.009)
        assert costing.cost_of("somebody/unknown-model", usage) > 0.0       # not local: the default price

    def test_without_reported_usage_both_sides_are_estimated_and_said_to_be(self):
        from orchestra.engine import costing

        usage = costing.usage_of(None, prompt="word " * 400, output="word " * 100)

        assert usage.source == "estimated"
        assert usage.prompt_tokens > usage.completion_tokens > 0

    def test_an_agent_is_charged_for_a_call_that_failed(self, monkeypatch):
        """A call that timed out after the model worked for minutes was free on paper."""
        import threading
        from unittest.mock import MagicMock

        from orchestra.engine.parallel import _run_one_agent

        self._priced(monkeypatch, {"p1/m1": 0.01})
        provider = MagicMock()
        provider.model_label.return_value = "p1/m1"
        provider.name = "p1"
        provider.last_usage.return_value = {"prompt_tokens": 3000, "completion_tokens": 1000}
        run = OrchestraRun(run_id="run-cost", mode="ask", task="t")

        with patch("orchestra.engine.parallel.resolve", return_value=(provider, "m1")), \
             patch("orchestra.engine.parallel.run_with_retry", return_value=("[ERROR] Timeout after 180s", 124)), \
             patch("orchestra.engine.artifacts.append_event"), \
             patch("orchestra.engine.artifacts.write_agent_log"), \
             patch("orchestra.engine.artifacts.write_normalized"), \
             patch("orchestra.engine.artifacts.write_manifest"):
            _run_one_agent(run, "cdx-fast", "prompt", threading.Lock(), MagicMock(), threading.Event())

        agent = run.agents[0]
        assert agent.status == AgentStatus.FAILED
        assert agent.estimated_cost_usd == pytest.approx(0.04)
        assert (agent.prompt_tokens, agent.estimated_completion_tokens, agent.cost_source) == (3000, 1000, "reported")
