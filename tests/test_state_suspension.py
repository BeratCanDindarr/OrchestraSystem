"""Tests for HITL state suspension and resumption."""
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from orchestra.engine.state_suspension import (
    suspend_run,
    resume_run,
    delete_paused_run,
    cleanup_expired_paused_runs,
    release_resume_claim,
)
from orchestra.models import OrchestraRun, AgentRun, RunStatus, AgentStatus
from orchestra.state import ApprovalState


@pytest.fixture
def db_connection():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Initialize paused_runs table
    conn.execute("""
        CREATE TABLE paused_runs (
            run_id TEXT PRIMARY KEY,
            checkpoint_data TEXT NOT NULL,
            paused_at TEXT NOT NULL,
            paused_by TEXT DEFAULT 'system',
            expires_at TEXT NOT NULL,
            resume_attempt_count INTEGER DEFAULT 0,
            last_resume_error TEXT,
            status TEXT DEFAULT 'pending',
            resume_owner TEXT,
            resume_lease_expires_at TEXT
        )
    """)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def sample_run():
    """Create a sample OrchestraRun for testing."""
    run = OrchestraRun(mode="critical", task="Test task")
    run.status = RunStatus.WAITING_APPROVAL
    run.approval_state = ApprovalState.PENDING

    agent1 = AgentRun(alias="cdx-fast", provider="openai", model="gpt-4")
    agent1.status = AgentStatus.COMPLETED
    agent1.stdout_log = "Agent output 1"
    agent1.confidence = 0.92
    agent1.validation_status = "passed"

    agent2 = AgentRun(alias="gmn-pro", provider="google", model="gemini")
    agent2.status = AgentStatus.COMPLETED
    agent2.stdout_log = "Agent output 2"
    agent2.confidence = 0.87
    agent2.validation_status = "passed"

    run.agents = [agent1, agent2]
    run.reviews = [
        {
            "stage": "round1",
            "status": "passed",
            "winner": "tie",
            "reason": "both_strong",
        }
    ]
    run.summary = "Synthesis summary"

    return run


def test_suspend_run(db_connection, sample_run):
    """Test suspending a run to database."""
    suspend_run(db_connection, sample_run, paused_by="user_approval")

    row = db_connection.execute(
        "SELECT checkpoint_data, paused_by FROM paused_runs WHERE run_id = ?",
        (sample_run.run_id,),
    ).fetchone()

    assert row is not None
    assert row["paused_by"] == "user_approval"

    data = json.loads(row["checkpoint_data"])
    assert data["run_id"] == sample_run.run_id
    assert data["mode"] == "critical"
    assert data["task"] == "Test task"
    assert len(data["agents"]) == 2
    assert data["agents"][0]["alias"] == "cdx-fast"
    assert data["agents"][0]["confidence"] == 0.92
    assert len(data["reviews"]) == 1


def test_resume_run(db_connection, sample_run):
    """Test resuming a run from database."""
    suspend_run(db_connection, sample_run)

    restored_run = resume_run(db_connection, sample_run.run_id)

    assert restored_run is not None
    assert restored_run.run_id == sample_run.run_id
    assert restored_run.mode == "critical"
    assert restored_run.task == "Test task"
    assert len(restored_run.agents) == 2
    assert restored_run.agents[0].alias == "cdx-fast"
    assert restored_run.agents[0].stdout_log == "Agent output 1"
    assert restored_run.agents[0].confidence == 0.92
    assert len(restored_run.reviews) == 1


def test_resume_nonexistent_run(db_connection):
    """Test resuming a run that doesn't exist."""
    result = resume_run(db_connection, "nonexistent_run")
    assert result is None


def test_expired_checkpoint_cleanup(db_connection, sample_run):
    """Test that expired checkpoints are cleaned up on resume attempt."""
    suspend_run(db_connection, sample_run)

    # Manually set expires_at to past
    db_connection.execute(
        "UPDATE paused_runs SET expires_at = ? WHERE run_id = ?",
        (
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            sample_run.run_id,
        ),
    )
    db_connection.commit()

    # Try to resume — should return None and delete the row
    result = resume_run(db_connection, sample_run.run_id)
    assert result is None

    row = db_connection.execute(
        "SELECT * FROM paused_runs WHERE run_id = ?", (sample_run.run_id,)
    ).fetchone()
    assert row is None


def test_delete_paused_run(db_connection, sample_run):
    """Test explicitly deleting a paused run."""
    suspend_run(db_connection, sample_run)

    row_before = db_connection.execute(
        "SELECT * FROM paused_runs WHERE run_id = ?", (sample_run.run_id,)
    ).fetchone()
    assert row_before is not None

    delete_paused_run(db_connection, sample_run.run_id)

    row_after = db_connection.execute(
        "SELECT * FROM paused_runs WHERE run_id = ?", (sample_run.run_id,)
    ).fetchone()
    assert row_after is None


def test_cleanup_expired_paused_runs(db_connection):
    """Test bulk cleanup of expired paused runs."""
    # Create 3 paused runs
    run1 = OrchestraRun(mode="ask", task="Task 1")
    run2 = OrchestraRun(mode="critical", task="Task 2")
    run3 = OrchestraRun(mode="critical", task="Task 3")

    suspend_run(db_connection, run1)
    suspend_run(db_connection, run2)
    suspend_run(db_connection, run3)

    # Expire run1 and run2
    past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    db_connection.execute(
        "UPDATE paused_runs SET expires_at = ? WHERE run_id IN (?, ?)",
        (past_time, run1.run_id, run2.run_id),
    )
    db_connection.commit()

    count = cleanup_expired_paused_runs(db_connection)
    assert count == 2

    # Check that run3 still exists, run1 and run2 deleted
    assert (
        db_connection.execute(
            "SELECT COUNT(*) FROM paused_runs WHERE run_id = ?", (run3.run_id,)
        ).fetchone()[0]
        == 1
    )
    assert (
        db_connection.execute(
            "SELECT COUNT(*) FROM paused_runs WHERE run_id IN (?, ?)",
            (run1.run_id, run2.run_id),
        ).fetchone()[0]
        == 0
    )


def test_suspension_preserves_all_state(db_connection, sample_run):
    """Test that suspension captures complete run state."""
    sample_run.summary = "Test synthesis"
    sample_run.latest_review_stage = "verification"
    sample_run.latest_review_status = "passed"
    sample_run.latest_review_winner = "cdx-fast"

    suspend_run(db_connection, sample_run)
    restored = resume_run(db_connection, sample_run.run_id)

    assert restored.summary == "Test synthesis"
    assert restored.latest_review_stage == "verification"
    assert restored.latest_review_status == "passed"
    assert restored.latest_review_winner == "cdx-fast"


def test_concurrent_resume_only_one_succeeds(tmp_path, sample_run):
    """Test that concurrent resume attempts only allow one to succeed.

    This test verifies the CAS (compare-and-swap) mechanism prevents
    duplicate execution of the same paused run.
    """
    import tempfile

    # Use a file-based SQLite DB to test concurrent connections (in-memory doesn't work)
    db_file = str(tmp_path / "test_concurrent.db")

    # Setup: create DB, table, and suspend the run
    conn_setup = sqlite3.connect(db_file)
    conn_setup.execute("PRAGMA journal_mode=WAL")
    conn_setup.execute("""
        CREATE TABLE paused_runs (
            run_id TEXT PRIMARY KEY,
            checkpoint_data TEXT NOT NULL,
            paused_at TEXT NOT NULL,
            paused_by TEXT DEFAULT 'system',
            expires_at TEXT NOT NULL,
            resume_attempt_count INTEGER DEFAULT 0,
            last_resume_error TEXT,
            status TEXT DEFAULT 'pending',
            resume_owner TEXT,
            resume_lease_expires_at TEXT
        )
    """)
    conn_setup.commit()

    suspend_run(conn_setup, sample_run)
    conn_setup.close()

    # Two threads attempt to resume the same run concurrently
    results = []

    def attempt_resume():
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA journal_mode=WAL")
        result = resume_run(conn, sample_run.run_id)
        results.append(result)
        conn.close()

    t1 = threading.Thread(target=attempt_resume)
    t2 = threading.Thread(target=attempt_resume)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one should succeed (return OrchestraRun), one should fail (return None)
    non_none_results = [r for r in results if r is not None]
    assert len(non_none_results) == 1, f"Expected exactly 1 successful resume, got {len(non_none_results)}"
    assert non_none_results[0].run_id == sample_run.run_id
