"""Tests for counter-example search on unanimous consensus."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from orchestra.engine.adversarial import search_counter_example
from orchestra.models import AgentRun, AgentStatus, OrchestraRun


class TestSearchCounterExample:
    """Counter-example search and consensus downgrade."""

    def test_disabled_when_adversarial_off(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        run.summary = "Consensus answer"
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=False):
            search_counter_example(run, "test")
            assert "Consensus challenged" not in (run.summary or "")

    def test_skips_when_already_challenged(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        run.summary = "Consensus answer"
        run.consensus_challenged = True
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            search_counter_example(run, "test")
            assert "Consensus challenged" not in run.summary

    def test_skips_when_no_summary(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        run.summary = None
        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            search_counter_example(run, "test")
            assert run.summary is None

    def test_downgrades_summary_on_counter_output(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        run.summary = "All agents agree that X is true."
        mock_agent = AgentRun(alias="gmn-pro", provider="gemini", model="pro")
        mock_agent.status = AgentStatus.COMPLETED
        mock_agent.stdout_log = "Edge case: when Y happens, X fails."
        mock_ce_run = MagicMock()
        mock_ce_run.agents = [mock_agent]

        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.adversarial.config.adversarial_counter_example_alias", return_value="gmn-pro"):
                with patch("orchestra.engine.runner.run_ask", return_value=mock_ce_run):
                    with patch("orchestra.engine.artifacts.append_event") as mock_append:
                        search_counter_example(run, "test")

        assert run.consensus_challenged is True
        assert "Consensus challenged" in run.summary
        assert "Edge case" in run.summary
        mock_append.assert_called_once()
        assert mock_append.call_args[0][1]["event"] == "consensus_challenged"

    def test_no_downgrade_when_counter_empty(self) -> None:
        run = OrchestraRun(mode="dual", task="test")
        run.summary = "All agents agree."
        mock_ce_run = MagicMock()
        mock_ce_run.agents = []

        with patch("orchestra.engine.adversarial.config.adversarial_enabled", return_value=True):
            with patch("orchestra.engine.runner.run_ask", return_value=mock_ce_run):
                search_counter_example(run, "test")

        assert run.summary == "All agents agree."
        assert not getattr(run, "consensus_challenged", False)
