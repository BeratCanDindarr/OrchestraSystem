"""Tests for semantic memory indexing during task finalization."""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

import pytest

from orchestra.models import OrchestraRun, AgentRun, RunStatus, AgentStatus
from orchestra.engine.runner import _finalize


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


@patch('orchestra.engine.runner.artifacts')
@patch('orchestra.engine.runner.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_indexes_completed_task(mock_get_memory, mock_append_event, mock_artifacts, sample_completed_run):
    """Test that completed tasks are indexed to semantic memory on completion."""
    # Mock get_memory at the source where _finalize imports it
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Run _finalize
    _finalize(sample_completed_run)

    # Verify memory.add() was called with correct arguments
    assert mock_memory.add.called
    call_args = mock_memory.add.call_args

    # Check positional argument (task)
    assert call_args[0][0] == sample_completed_run.task

    # Check keyword arguments
    kwargs = call_args[1]
    assert "metadata" in kwargs
    assert kwargs["metadata"]["run_id"] == sample_completed_run.run_id
    assert kwargs["metadata"]["mode"] == "ask"
    assert len(kwargs["metadata"]["agents"]) == 2


@patch('orchestra.engine.runner.artifacts')
@patch('orchestra.engine.runner.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_skips_failed_tasks(mock_get_memory, mock_append_event, mock_artifacts):
    """Test that failed tasks are not indexed to memory."""
    run = OrchestraRun(mode="ask", task="Test task")

    # Create agents with one failing
    agent1 = AgentRun(alias="cdx-fast", provider="openai", model="gpt-4")
    agent1.status = AgentStatus.COMPLETED

    agent2 = AgentRun(alias="gmn-pro", provider="google", model="gemini")
    agent2.status = AgentStatus.FAILED  # This will make run fail

    run.agents = [agent1, agent2]
    run.latest_review_status = "passed"

    # Mock get_memory to verify add() is NOT called for failed runs
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    # Run _finalize
    _finalize(run)

    # Check that run status is FAILED
    assert run.status == RunStatus.FAILED

    # Verify memory.add() was NOT called (failed tasks aren't indexed)
    assert not mock_memory.add.called


@patch('orchestra.engine.runner.artifacts')
@patch('orchestra.engine.runner.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_handles_indexing_errors_gracefully(mock_get_memory, mock_append_event, mock_artifacts, sample_completed_run):
    """Test that indexing errors don't prevent finalization."""
    # Mock get_memory to raise an exception
    mock_memory = MagicMock()
    mock_memory.add.side_effect = Exception("Indexing failed")
    mock_get_memory.return_value = mock_memory

    # Should not raise despite indexing error
    _finalize(sample_completed_run)

    # Should still have finalized despite the error
    assert sample_completed_run.status == RunStatus.COMPLETED


@patch('orchestra.engine.runner.artifacts')
@patch('orchestra.engine.runner.append_event')
@patch('orchestra.storage.memory.get_memory')
def test_finalize_creates_correct_metadata(mock_get_memory, mock_append_event, mock_artifacts, sample_completed_run):
    """Test that metadata is correctly created during indexing."""
    # Mock get_memory to verify metadata is correct
    mock_memory = MagicMock()
    mock_get_memory.return_value = mock_memory

    _finalize(sample_completed_run)

    # Verify memory.add() was called
    assert mock_memory.add.called
    call_args = mock_memory.add.call_args
    kwargs = call_args[1]
    meta = kwargs["metadata"]

    # Check all metadata fields are present and correct
    assert meta["run_id"] == sample_completed_run.run_id
    assert meta["mode"] == "ask"
    assert meta["agent_count"] == 2
    assert set(meta["agents"]) == {"cdx-fast", "gmn-pro"}

    # Check summary was created
    assert "summary" in kwargs
    assert "ASK:" in kwargs["summary"]
