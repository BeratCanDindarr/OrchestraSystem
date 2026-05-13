"""Tests for CriticAgent — Dissent Re-run (_critic_rerun_if_needed)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.state import ApprovalState


def _make_run(
    summary: str = "",
    avg_confidence: float = 0.5,
    run_id: str = "test-run-001",
) -> OrchestraRun:
    run = OrchestraRun(mode="critical", task="Fix the payment bug.", run_id=run_id)
    run.summary = summary
    run.avg_confidence = avg_confidence
    run.status = RunStatus.COMPLETED
    return run


SUMMARY_WITH_DISSENT = (
    "## Answer\nUse method A.\n\n"
    "## Key Signals\nBoth agents agree.\n\n"
    "## Dissent\nAgent B suggests method B is safer for edge cases.\n"
)

SUMMARY_NO_DISSENT = (
    "## Answer\nUse method A.\n\n"
    "## Key Signals\nBoth agents agree.\n\n"
    "## Dissent\n"
)


# ---------------------------------------------------------------------------
# Guard: no summary → no critic
# ---------------------------------------------------------------------------

def test_no_summary_skips_critic():
    from orchestra.engine.runner import _critic_rerun_if_needed

    run = _make_run(summary="", avg_confidence=0.4)
    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.runner.run_ask") as mock_ask:
            _critic_rerun_if_needed(run, "Fix the bug")
            mock_ask.assert_not_called()


# ---------------------------------------------------------------------------
# Guard: already triggered → no second critic
# ---------------------------------------------------------------------------

def test_already_triggered_skips_critic():
    from orchestra.engine.runner import _critic_rerun_if_needed

    run = _make_run(summary=SUMMARY_WITH_DISSENT, avg_confidence=0.4)
    events = [{"event": "critic_triggered", "alias": "cld-fast"}]
    with patch("orchestra.engine.artifacts.read_events", return_value=events):
        with patch("orchestra.engine.runner.run_ask") as mock_ask:
            _critic_rerun_if_needed(run, "Fix the bug")
            mock_ask.assert_not_called()


# ---------------------------------------------------------------------------
# Guard: confidence >= threshold → no critic
# ---------------------------------------------------------------------------

def test_high_confidence_skips_critic():
    from orchestra.engine.runner import _critic_rerun_if_needed

    run = _make_run(summary=SUMMARY_WITH_DISSENT, avg_confidence=0.9)
    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.runner.run_ask") as mock_ask:
            _critic_rerun_if_needed(run, "Fix the bug")
            mock_ask.assert_not_called()


# ---------------------------------------------------------------------------
# Guard: no dissent section → no critic
# ---------------------------------------------------------------------------

def test_no_dissent_in_summary_skips_critic():
    from orchestra.engine.runner import _critic_rerun_if_needed

    run = _make_run(summary=SUMMARY_NO_DISSENT, avg_confidence=0.4)
    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.runner.run_ask") as mock_ask:
            _critic_rerun_if_needed(run, "Fix the bug")
            mock_ask.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path: dissent + low confidence → critic fires, summary updated
# ---------------------------------------------------------------------------

def test_critic_fires_and_updates_summary():
    from orchestra.engine.runner import _critic_rerun_if_needed

    run = _make_run(summary=SUMMARY_WITH_DISSENT, avg_confidence=0.4)
    critic_agent = AgentRun(alias="cld-fast", provider="claude", model="claude-haiku-4-5")
    critic_agent.stdout_log = "## Answer\nUse method B for safety.\n"
    critic_agent.status = AgentStatus.COMPLETED
    mock_critic_run = MagicMock()
    mock_critic_run.agents = [critic_agent]

    events_emitted = []

    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.runner.append_event", side_effect=lambda rid, e: events_emitted.append(e)):
            with patch("orchestra.engine.runner.run_ask", return_value=mock_critic_run):
                _critic_rerun_if_needed(run, "Fix the bug")

    assert run.summary == "## Answer\nUse method B for safety.\n"
    event_names = [e["event"] for e in events_emitted]
    assert "critic_triggered" in event_names
    assert "critic_completed" in event_names


# ---------------------------------------------------------------------------
# Critic run fails (no output) → summary unchanged
# ---------------------------------------------------------------------------

def test_critic_empty_output_leaves_summary_unchanged():
    from orchestra.engine.runner import _critic_rerun_if_needed

    original_summary = SUMMARY_WITH_DISSENT
    run = _make_run(summary=original_summary, avg_confidence=0.4)
    critic_agent = AgentRun(alias="cld-fast", provider="claude", model="claude-haiku-4-5")
    critic_agent.stdout_log = ""
    critic_agent.status = AgentStatus.FAILED
    mock_critic_run = MagicMock()
    mock_critic_run.agents = [critic_agent]

    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.artifacts.append_event"):
            with patch("orchestra.engine.runner.run_ask", return_value=mock_critic_run):
                _critic_rerun_if_needed(run, "Fix the bug")

    assert run.summary == original_summary


# ---------------------------------------------------------------------------
# Critic run has no agents → summary unchanged
# ---------------------------------------------------------------------------

def test_critic_no_agents_leaves_summary_unchanged():
    from orchestra.engine.runner import _critic_rerun_if_needed

    original_summary = SUMMARY_WITH_DISSENT
    run = _make_run(summary=original_summary, avg_confidence=0.4)
    mock_critic_run = MagicMock()
    mock_critic_run.agents = []

    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.artifacts.append_event"):
            with patch("orchestra.engine.runner.run_ask", return_value=mock_critic_run):
                _critic_rerun_if_needed(run, "Fix the bug")

    assert run.summary == original_summary


# ---------------------------------------------------------------------------
# Configurable threshold via review_config
# ---------------------------------------------------------------------------

def test_custom_threshold_respected():
    from orchestra.engine.runner import _critic_rerun_if_needed

    # confidence=0.70, threshold=0.80 → should fire
    run = _make_run(summary=SUMMARY_WITH_DISSENT, avg_confidence=0.70)

    mock_review_cfg = {"critic_confidence_threshold": "0.80", "critic_alias": "cld-fast"}
    critic_agent = AgentRun(alias="cld-fast", provider="claude", model="claude-haiku-4-5")
    critic_agent.stdout_log = "Revised answer."
    mock_critic_run = MagicMock()
    mock_critic_run.agents = [critic_agent]

    with patch("orchestra.engine.artifacts.read_events", return_value=[]):
        with patch("orchestra.engine.artifacts.append_event"):
            with patch("orchestra.config.review_config", return_value=mock_review_cfg):
                with patch("orchestra.engine.runner.run_ask", return_value=mock_critic_run):
                    _critic_rerun_if_needed(run, "Fix the bug")

    assert run.summary == "Revised answer."
