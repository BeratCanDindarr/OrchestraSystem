"""Tests for semantic memory indexing during task finalization."""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

from orchestra.models import OrchestraRun, AgentRun, RunStatus, AgentStatus
from orchestra.engine.runners import _finalize


def mock_embed(text):
    """Create a mock embedding vector from text."""
    # Use hash-based pseudo-random vector for deterministic testing
    h = hash(text) % (2**32)
    np.random.seed(h)
    return np.random.randn(384).tolist()  # Standard embedding size


@pytest.fixture
def temp_memory_db(monkeypatch):
    """Create temporary memory database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "memory.db"
        monkeypatch.setenv("ORCHESTRA_MEMORY_DB", str(db_path))

        # Also patch .orchestra directory
        orchestra_dir = Path(tmpdir) / ".orchestra"
        orchestra_dir.mkdir(exist_ok=True)

        yield str(db_path)


@pytest.fixture
def sample_completed_run():
    """Create a sample completed run."""
    run = OrchestraRun(mode="ask", task="What is machine learning?")
    run.status = RunStatus.RUNNING

    agent1 = AgentRun(alias="cdx-fast", provider="openai", model="gpt-4")
    agent1.status = AgentStatus.COMPLETED
    agent1.stdout_log = "Agent output 1"
    agent1.confidence = 0.92

    agent2 = AgentRun(alias="gmn-pro", provider="google", model="gemini")
    agent2.status = AgentStatus.COMPLETED
    agent2.stdout_log = "Agent output 2"
    agent2.confidence = 0.87

    run.agents = [agent1, agent2]
    run.latest_review_status = "passed"

    return run


@patch('orchestra.engine.runners.common.artifacts')
@patch('orchestra.engine.runners.common.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_sampling_gates_judge(mock_mem, mock_ev, mock_art, sample_completed_run, monkeypatch):
    """sample_rate=0.0 skips the LLM judge entirely (production trace sampling)."""
    from orchestra import config
    monkeypatch.setattr(config, "eval_harness_enabled", lambda: True)
    monkeypatch.setattr(config, "eval_sample_rate", lambda: 0.0)
    sample_completed_run.summary = "an answer"
    with patch("orchestra.engine.eval_harness.run_judge") as judge:
        _finalize(sample_completed_run)
    judge.assert_not_called()


@patch('orchestra.engine.runners.common.artifacts')
@patch('orchestra.engine.runners.common.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_logs_scores_and_calibration(mock_mem, mock_ev, mock_art, sample_completed_run, monkeypatch):
    """When sampled, judge scores + calibration are persisted via log_score."""
    from orchestra import config
    from orchestra.engine.eval_harness import EvalResult
    monkeypatch.setattr(config, "eval_harness_enabled", lambda: True)
    monkeypatch.setattr(config, "eval_sample_rate", lambda: 1.0)
    monkeypatch.setattr(config, "eval_calibration_enabled", lambda: True)
    sample_completed_run.summary = "an answer"
    fake = EvalResult(
        task="t", run_id="r", task_hash="h", created_at="", mode="ask", output="an answer",
        answer_quality=0.8, factual_grounding=0.7, source_attribution=1.0,
        conciseness=0.9, composite_score=0.85, explanation="ok",
    )
    with patch("orchestra.engine.eval_harness.run_judge", return_value=fake), \
         patch("orchestra.engine.calibration.assess_calibration", return_value={"calibration_score": 0.6}), \
         patch("orchestra.storage.eval_tracker.EvalTracker.log_score") as log_score:
        _finalize(sample_completed_run)
    log_score.assert_called_once()
    scores = log_score.call_args[0][2]
    assert scores["composite_score"] == 0.85
    assert scores["calibration"] == 0.6
