"""Tests for speculative plan execution in storage/speculation.py."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestra.models import RunStatus
from orchestra.storage.speculation import create_speculative_plan, execute_speculative_plan


@pytest.fixture
def db_connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE speculative_plans (
            plan_id TEXT PRIMARY KEY,
            repo_root TEXT,
            base_ref TEXT,
            task TEXT,
            status TEXT,
            manifest_path TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    yield conn
    conn.close()


def _make_mock_run(confidence: float, stdout_log: str, status: str = "completed"):
    """Build a mock OrchestraRun for patching run_ask."""
    run = MagicMock()
    run.status = RunStatus(status)
    agent = MagicMock()
    agent.stdout_log = stdout_log
    agent.confidence = confidence
    run.agents = [agent]
    return run


def test_execute_speculative_plan_selects_winner_by_confidence(db_connection, tmp_path):
    """Highest confidence hypothesis wins."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    orchestra_root = tmp_path / "orchestra"

    plan = create_speculative_plan(
        db_connection,
        orchestra_root,
        repo_root=str(repo_root),
        task="Implement feature X",
        base_ref="main",
        hypothesis_names=["Approach A", "Approach B"],
    )
    plan_id = plan["plan_id"]

    # Create worktree directories so prepare_speculative_worktrees is skipped
    for hypothesis in plan["hypotheses"]:
        Path(hypothesis["worktree_path"]).mkdir(parents=True, exist_ok=True)

    mock_runs = [
        _make_mock_run(0.7, "output a"),
        _make_mock_run(0.9, "output b"),
    ]

    with patch("orchestra.engine.runner.run_ask", side_effect=mock_runs), patch(
        "os.chdir"
    ):
        result = execute_speculative_plan(
            db_connection, orchestra_root, plan_id, mode="ask"
        )

    assert result["status"] == "completed"
    hypotheses = result["hypotheses"]
    assert hypotheses[0]["status"] == "completed"
    assert hypotheses[0]["confidence"] == 0.7
    assert hypotheses[1]["status"] == "completed"
    assert hypotheses[1]["confidence"] == 0.9

    eval_ = result["evaluation"]
    assert eval_["selection"] == "winner_selected"
    assert eval_["selected_branch"] == hypotheses[1]["branch"]
    assert eval_["compile_check"] == "passed"
    assert eval_["tests"] == "passed"
    assert "Winner: Approach B" in eval_["notes"]


def test_execute_speculative_plan_tiebreaker_shortest_log(db_connection, tmp_path):
    """When confidence is tied, shortest stdout_log wins."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    orchestra_root = tmp_path / "orchestra"

    plan = create_speculative_plan(
        db_connection,
        orchestra_root,
        repo_root=str(repo_root),
        task="Implement feature X",
        base_ref="main",
        hypothesis_names=["Approach A", "Approach B"],
    )
    plan_id = plan["plan_id"]

    for hypothesis in plan["hypotheses"]:
        Path(hypothesis["worktree_path"]).mkdir(parents=True, exist_ok=True)

    mock_runs = [
        _make_mock_run(0.85, "this is a much longer output log with many characters"),
        _make_mock_run(0.85, "short"),
    ]

    with patch("orchestra.engine.runner.run_ask", side_effect=mock_runs), patch(
        "os.chdir"
    ):
        result = execute_speculative_plan(
            db_connection, orchestra_root, plan_id, mode="ask"
        )

    eval_ = result["evaluation"]
    assert eval_["selection"] == "winner_selected"
    # Approach B has shorter stdout_log, so it wins on tie
    assert "approach-b" in eval_["selected_branch"]


def test_execute_speculative_plan_all_failed(db_connection, tmp_path):
    """If all agents fail, selection is failed and no winner."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    orchestra_root = tmp_path / "orchestra"

    plan = create_speculative_plan(
        db_connection,
        orchestra_root,
        repo_root=str(repo_root),
        task="Implement feature X",
        base_ref="main",
        hypothesis_names=["Approach A"],
    )
    plan_id = plan["plan_id"]

    for hypothesis in plan["hypotheses"]:
        Path(hypothesis["worktree_path"]).mkdir(parents=True, exist_ok=True)

    mock_runs = [
        _make_mock_run(0.0, "", status="failed"),
    ]

    with patch("orchestra.engine.runner.run_ask", side_effect=mock_runs), patch(
        "os.chdir"
    ):
        result = execute_speculative_plan(
            db_connection, orchestra_root, plan_id, mode="ask"
        )

    assert result["status"] == "completed"
    assert result["hypotheses"][0]["status"] == "failed"
    assert result["evaluation"]["selection"] == "failed"
    assert result["evaluation"]["selected_branch"] == ""
    assert result["evaluation"]["compile_check"] == "failed"


def test_execute_speculative_plan_invalid_mode(db_connection, tmp_path):
    """Invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid mode"):
        execute_speculative_plan(
            db_connection, tmp_path / "orchestra", "nonexistent", mode="invalid"
        )
