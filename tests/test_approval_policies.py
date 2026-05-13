"""Tests for approval policy evaluation."""
import pytest
from unittest.mock import Mock, patch

from orchestra.engine.runner import (
    _should_require_approval,
    _estimate_run_cost,
    _estimate_avg_confidence,
    _alias_to_model_label,
)
from orchestra.models import OrchestraRun, AgentRun, AgentStatus
from orchestra import config


class TestApprovalPolicies:
    """Test approval policy evaluation logic."""

    def test_approval_disabled_returns_false(self):
        """When approval policies are disabled, should return False."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {"enabled": False}

            run = OrchestraRun(mode="critical", task="Test task")
            result = _should_require_approval(run, "Test task")

            assert result is False

    def test_cost_threshold_triggers_approval(self):
        """Approval triggers when estimated cost exceeds threshold."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": True,
                "cost_threshold_usd": 0.5,
                "require_approval_on_low_confidence": False,
                "require_approval_on_complex_task": False,
                "approval_required_for_modes": [],
            }
            with patch("orchestra.engine.runner._estimate_run_cost") as mock_cost:
                mock_cost.return_value = 1.5  # Exceeds threshold

                run = OrchestraRun(mode="critical", task="Test task")
                result = _should_require_approval(run, "Test task")

                assert result is True

    def test_cost_below_threshold_no_approval(self):
        """No approval when cost is below threshold."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": True,
                "cost_threshold_usd": 2.0,
                "require_approval_on_low_confidence": False,
                "require_approval_on_complex_task": False,
                "approval_required_for_modes": [],
            }
            with patch("orchestra.engine.runner._estimate_run_cost") as mock_cost:
                mock_cost.return_value = 0.5  # Below threshold

                run = OrchestraRun(mode="critical", task="Test task")
                result = _should_require_approval(run, "Test task")

                assert result is False

    def test_low_confidence_triggers_approval(self):
        """Approval triggers when confidence is below threshold."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": False,
                "require_approval_on_low_confidence": True,
                "confidence_threshold": 0.8,
                "require_approval_on_complex_task": False,
                "approval_required_for_modes": [],
            }
            with patch("orchestra.engine.runner._estimate_avg_confidence") as mock_conf:
                mock_conf.return_value = 0.6  # Below threshold

                run = OrchestraRun(mode="critical", task="Test task")
                result = _should_require_approval(run, "Test task")

                assert result is True

    def test_high_confidence_no_approval(self):
        """No approval when confidence is above threshold."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": False,
                "require_approval_on_low_confidence": True,
                "confidence_threshold": 0.7,
                "require_approval_on_complex_task": False,
                "approval_required_for_modes": [],
            }
            with patch("orchestra.engine.runner._estimate_avg_confidence") as mock_conf:
                mock_conf.return_value = 0.85  # Above threshold

                run = OrchestraRun(mode="critical", task="Test task")
                result = _should_require_approval(run, "Test task")

                assert result is False

    def test_complexity_keywords_trigger_approval(self):
        """Approval triggers when task contains complexity keywords."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": False,
                "require_approval_on_low_confidence": False,
                "require_approval_on_complex_task": True,
                "complexity_keywords": ["refactor", "migration", "architecture"],
                "approval_required_for_modes": [],
            }

            run = OrchestraRun(mode="critical", task="Refactor the database schema")
            result = _should_require_approval(run, "Refactor the database schema")

            assert result is True

    def test_complexity_keywords_case_insensitive(self):
        """Complexity keyword matching is case-insensitive."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": False,
                "require_approval_on_low_confidence": False,
                "require_approval_on_complex_task": True,
                "complexity_keywords": ["refactor", "migration"],
                "approval_required_for_modes": [],
            }

            run = OrchestraRun(mode="critical", task="REFACTOR the system")
            result = _should_require_approval(run, "REFACTOR the system")

            assert result is True

    def test_mode_based_approval_trigger(self):
        """Approval triggers for specific modes listed in policy."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": False,
                "require_approval_on_low_confidence": False,
                "require_approval_on_complex_task": False,
                "complexity_keywords": [],
                "approval_required_for_modes": ["critical", "planned"],
            }

            run = OrchestraRun(mode="critical", task="Test task")
            result = _should_require_approval(run, "Test task")

            assert result is True

    def test_mode_not_in_approval_required_modes(self):
        """No approval when mode is not in approval_required_for_modes."""
        with patch("orchestra.config.approval_policies_config") as mock_cfg:
            mock_cfg.return_value = {
                "enabled": True,
                "require_approval_on_high_cost": False,
                "require_approval_on_low_confidence": False,
                "require_approval_on_complex_task": False,
                "complexity_keywords": [],
                "approval_required_for_modes": ["planned"],
            }

            run = OrchestraRun(mode="critical", task="Test task")
            result = _should_require_approval(run, "Test task")

            assert result is False

    def test_estimate_run_cost_single_agent(self):
        """Cost estimation for single agent."""
        run = OrchestraRun(mode="ask", task="Test")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        agent.estimated_completion_tokens = 1000  # 1000 tokens
        run.agents = [agent]

        # cdx-fast = gpt-5.4/low = 0.002 per 1k tokens
        # Cost = (1000 / 1000) * 0.002 = $0.002
        cost = _estimate_run_cost(run)
        assert cost == pytest.approx(0.002, abs=0.0001)

    def test_estimate_run_cost_multiple_agents(self):
        """Cost estimation for multiple agents."""
        run = OrchestraRun(mode="dual", task="Test")

        agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED)
        agent1.estimated_completion_tokens = 1000

        agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED)
        agent2.estimated_completion_tokens = 500

        run.agents = [agent1, agent2]

        # cdx-deep = gpt-5.4/xhigh = 0.012 per 1k tokens → (1000 / 1000) * 0.012 = 0.012
        # gmn-pro = gemini/pro = 0.006 per 1k tokens → (500 / 1000) * 0.006 = 0.003
        # Total = 0.015
        cost = _estimate_run_cost(run)
        assert cost == pytest.approx(0.015, abs=0.0001)

    def test_estimate_run_cost_zero_tokens(self):
        """Cost is zero when agents have no tokens."""
        run = OrchestraRun(mode="ask", task="Test")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        agent.estimated_completion_tokens = 0
        run.agents = [agent]

        cost = _estimate_run_cost(run)
        assert cost == 0.0

    def test_estimate_avg_confidence_single_agent(self):
        """Average confidence for single agent."""
        run = OrchestraRun(mode="ask", task="Test")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        agent.confidence = 0.9
        run.agents = [agent]

        avg = _estimate_avg_confidence(run)
        assert avg == pytest.approx(0.9)

    def test_estimate_avg_confidence_multiple_agents(self):
        """Average confidence for multiple agents."""
        run = OrchestraRun(mode="dual", task="Test")

        agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED)
        agent1.confidence = 0.8

        agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED)
        agent2.confidence = 0.6

        run.agents = [agent1, agent2]

        avg = _estimate_avg_confidence(run)
        assert avg == pytest.approx(0.7)  # (0.8 + 0.6) / 2

    def test_estimate_avg_confidence_no_agents(self):
        """Default confidence is 1.0 when no agents."""
        run = OrchestraRun(mode="ask", task="Test")
        run.agents = []

        avg = _estimate_avg_confidence(run)
        assert avg == 1.0

    def test_estimate_avg_confidence_missing_confidence(self):
        """Missing confidence attribute is skipped."""
        run = OrchestraRun(mode="dual", task="Test")

        agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED)
        agent1.confidence = 0.8

        agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED)
        # agent2.confidence not set (will use default of 0.5)

        run.agents = [agent1, agent2]

        avg = _estimate_avg_confidence(run)
        assert avg == pytest.approx(0.65)  # (0.8 + 0.5) / 2, default confidence is 0.5

    def test_alias_to_model_label_known_aliases(self):
        """Model label lookup for known aliases."""
        assert _alias_to_model_label("cdx-fast") == "gpt-5.4/low"
        assert _alias_to_model_label("cdx-deep") == "gpt-5.4/xhigh"
        assert _alias_to_model_label("gmn-fast") == "gemini/flash"
        assert _alias_to_model_label("gmn-pro") == "gemini/pro"
        assert _alias_to_model_label("cld-fast") == "claude/sonnet"
        assert _alias_to_model_label("cld-deep") == "claude/opus"

    def test_alias_to_model_label_unknown_alias(self):
        """Unknown alias defaults to gpt-5.4/medium."""
        assert _alias_to_model_label("unknown-alias") == "gpt-5.4/medium"
