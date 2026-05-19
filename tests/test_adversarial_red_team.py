"""Tests for red-team agent injection."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from orchestra.engine.adversarial import inject_red_team
from orchestra.models import AgentRun, AgentStatus, OrchestraRun


class TestInjectRedTeam:
    """Red-team agent injection into a run."""

    def test_disabled_when_adversarial_off(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=False):
            inject_red_team(run, "test")
            assert len(run.agents) == 0

    def test_injects_red_team_agent(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        mock_agent = AgentRun(alias="cld-fast", provider="claude", model="sonnet")
        mock_agent.status = AgentStatus.COMPLETED
        mock_agent.stdout_log = "I disagree because X"
        mock_rt_run = MagicMock()
        mock_rt_run.agents = [mock_agent]

        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_red_team_alias", return_value="cld-fast"):
                with patch("orchestra.engine.runner.run_ask", return_value=mock_rt_run):
                    with patch("orchestra.engine.artifacts.append_event") as mock_append:
                        inject_red_team(run, "test")

        assert len(run.agents) == 1
        assert run.agents[0].alias == "red-team"
        assert run.agents[0].stdout_log == "I disagree because X"
        mock_append.assert_called_once()
        assert mock_append.call_args[0][1]["event"] == "red_team_injected"

    def test_skips_when_no_output(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        mock_rt_run = MagicMock()
        mock_rt_run.agents = []

        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.runner.run_ask", return_value=mock_rt_run):
                inject_red_team(run, "test")

        assert len(run.agents) == 0
