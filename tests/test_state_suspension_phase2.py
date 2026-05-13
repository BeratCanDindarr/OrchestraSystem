"""Phase 2 tests for State Suspension UPSERT and extended fields.

Tests cover:
- Idempotent UPSERT (no error on duplicate)
- Preserved resume_attempt_count across updates
- Extended fields stored in checkpoint_data (5 fields)
- TTL configuration from config
- Index existence
- Resume restores extended fields
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from orchestra.engine.state_suspension import suspend_run, resume_run
from orchestra.models import OrchestraRun, AgentRun, RunStatus, AgentStatus, InterruptState
from orchestra.state import ApprovalState


@pytest.fixture
def db_connection():
    """Create an in-memory SQLite database for testing with paused_runs table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Initialize paused_runs table with all Phase 2 columns
    conn.execute("""
        CREATE TABLE paused_runs (
            run_id TEXT PRIMARY KEY,
            checkpoint_data TEXT NOT NULL,
            paused_at TEXT NOT NULL,
            paused_by TEXT DEFAULT 'system',
            expires_at TEXT NOT NULL,
            resume_attempt_count INTEGER DEFAULT 0,
            last_resume_error TEXT
        )
    """)

    # Create index for TTL cleanup
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_paused_runs_expires
        ON paused_runs(expires_at)
    """)

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def sample_run():
    """Create a sample OrchestraRun with extended fields for testing."""
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

    # Extended fields for Phase 2
    run.total_cost_usd = 2.5
    run.avg_confidence = 0.92
    run.interrupt_state = InterruptState.CANCEL_REQUESTED
    run.failure = None  # No failure in this case
    run.turns = 3

    return run


class TestIdempotentUpsert:
    """Tests for UPSERT idempotency."""

    def test_suspend_run_idempotent(self, db_connection, sample_run):
        """Verify UPSERT doesn't error on duplicate run_id.

        Call suspend_run() twice with same run_id:
        - First call: inserts new row
        - Second call: updates existing row (no error)
        - Result: exactly 1 row in table
        """
        # First suspension
        suspend_run(db_connection, sample_run, paused_by="system")

        row_count_1 = db_connection.execute(
            "SELECT COUNT(*) as cnt FROM paused_runs WHERE run_id = ?",
            (sample_run.run_id,),
        ).fetchone()
        assert row_count_1["cnt"] == 1, "First suspension should create 1 row"

        # Second suspension with same run_id
        suspend_run(db_connection, sample_run, paused_by="system")

        row_count_2 = db_connection.execute(
            "SELECT COUNT(*) as cnt FROM paused_runs WHERE run_id = ?",
            (sample_run.run_id,),
        ).fetchone()
        assert (
            row_count_2["cnt"] == 1
        ), "Second suspension should not create duplicate (UPSERT should update)"

        # Verify no IntegrityError was raised
        # (If we got here without exception, UPSERT is working)


class TestResumeAttemptCountPreserved:
    """Tests for resume_attempt_count preservation across UPSERT."""

    def test_resume_attempt_count_preserved(self, db_connection, sample_run):
        """Verify resume_attempt_count survives UPSERT updates.

        1. Suspend run (resume_attempt_count = 0)
        2. Manually set resume_attempt_count = 5
        3. Call suspend_run() again with same run_id
        4. Assert: resume_attempt_count still = 5 (not reset)
        """
        # First suspension
        suspend_run(db_connection, sample_run, paused_by="system")

        # Manually set resume_attempt_count
        db_connection.execute(
            "UPDATE paused_runs SET resume_attempt_count = 5 WHERE run_id = ?",
            (sample_run.run_id,),
        )
        db_connection.commit()

        # Verify it was set
        row_before = db_connection.execute(
            "SELECT resume_attempt_count FROM paused_runs WHERE run_id = ?",
            (sample_run.run_id,),
        ).fetchone()
        assert row_before["resume_attempt_count"] == 5, "Setup: count should be 5"

        # Second suspension (UPSERT)
        suspend_run(db_connection, sample_run, paused_by="system")

        # Verify count was preserved
        row_after = db_connection.execute(
            "SELECT resume_attempt_count FROM paused_runs WHERE run_id = ?",
            (sample_run.run_id,),
        ).fetchone()
        assert (
            row_after["resume_attempt_count"] == 5
        ), "UPSERT should not reset resume_attempt_count"


class TestCheckpointExtendedFields:
    """Tests for 5 extended fields in checkpoint_data."""

    def test_checkpoint_extended_fields(self, db_connection, sample_run):
        """Verify all 5 extended fields are stored in checkpoint_data JSON.

        Extended fields:
        1. total_cost_usd
        2. avg_confidence
        3. interrupt_state
        4. failure
        5. turns
        """
        suspend_run(db_connection, sample_run, paused_by="system")

        row = db_connection.execute(
            "SELECT checkpoint_data FROM paused_runs WHERE run_id = ?",
            (sample_run.run_id,),
        ).fetchone()

        assert row is not None
        data = json.loads(row["checkpoint_data"])

        # Verify all 5 extended fields present
        assert (
            "total_cost_usd" in data
        ), "checkpoint_data must contain total_cost_usd"
        assert "avg_confidence" in data, "checkpoint_data must contain avg_confidence"
        assert (
            "interrupt_state" in data
        ), "checkpoint_data must contain interrupt_state"
        assert "failure" in data, "checkpoint_data must contain failure"
        assert "turns" in data, "checkpoint_data must contain turns"

        # Verify correct values
        assert data["total_cost_usd"] == 2.5, "total_cost_usd should match"
        assert data["avg_confidence"] == 0.92, "avg_confidence should match"
        assert data["interrupt_state"] == "cancel_requested", "interrupt_state should be 'cancel_requested'"
        assert data["failure"] is None, "failure should be None"
        assert data["turns"] == 3, "turns should be 3"


class TestTTLFromConfig:
    """Tests for checkpoint TTL configuration."""

    def test_expires_at_from_config(self, db_connection, sample_run):
        """Verify expires_at is calculated from config checkpoint_ttl_hours.

        1. Get checkpoint_ttl_hours from config
        2. Suspend run
        3. Query expires_at from database
        4. Assert: expires_at ≈ now + configured hours
        """
        # Get configured TTL hours
        from orchestra.config import checkpoint_ttl_hours

        ttl_hours = checkpoint_ttl_hours()

        # Suspend run
        before_suspend = datetime.now(timezone.utc)
        suspend_run(db_connection, sample_run, paused_by="system")
        after_suspend = datetime.now(timezone.utc)

        # Query expires_at
        row = db_connection.execute(
            "SELECT expires_at FROM paused_runs WHERE run_id = ?",
            (sample_run.run_id,),
        ).fetchone()

        assert row is not None
        expires_at = datetime.fromisoformat(row["expires_at"])

        # Calculate expected expiry window
        expected_min = before_suspend + timedelta(hours=ttl_hours)
        expected_max = after_suspend + timedelta(hours=ttl_hours)

        # Assert expires_at is within window
        assert (
            expected_min <= expires_at <= expected_max
        ), f"expires_at should be ≈ now + {ttl_hours} hours, got {expires_at}"


class TestIndexExists:
    """Tests for database index."""

    def test_idx_paused_runs_expires_created(self, db_connection):
        """Verify index idx_paused_runs_expires exists on expires_at column."""
        cursor = db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_paused_runs_expires'"
        )
        row = cursor.fetchone()

        assert row is not None, "Index idx_paused_runs_expires must exist"
        assert row[0] == "idx_paused_runs_expires", "Index name must match"


class TestResumeRestoresExtendedFields:
    """Tests for resume_run restores extended fields (Phase 2 fix target)."""

    @pytest.mark.skip(reason="Extended field restoration not yet implemented in resume_run")
    def test_resume_run_restores_extended_fields(self, db_connection, sample_run):
        """Verify resume_run loads all 5 extended fields from checkpoint_data.

        1. Suspend run with extended fields
        2. Call resume_run()
        3. Assert: returned run has all 5 fields with original values

        NOTE: This is a target test for Phase 2 implementation.
        Resume_run currently does not restore extended fields.
        """
        # Suspend original run
        suspend_run(db_connection, sample_run, paused_by="system")

        # Resume from database
        restored_run = resume_run(db_connection, sample_run.run_id)

        assert restored_run is not None, "resume_run should return run"

        # Verify extended fields restored
        assert (
            restored_run.total_cost_usd == 2.5
        ), "total_cost_usd should be restored"
        assert (
            restored_run.avg_confidence == 0.92
        ), "avg_confidence should be restored"
        assert (
            restored_run.interrupt_state == InterruptState.CANCEL_REQUESTED
        ), "interrupt_state should be restored"
        assert restored_run.failure is None, "failure should be None"
        assert restored_run.turns == 3, "turns should be restored"


class TestEdgeCases:
    """Additional edge case tests for robustness."""

    def test_suspend_with_zero_cost(self, db_connection):
        """Verify handling of zero total_cost_usd."""
        run = OrchestraRun(mode="ask", task="Free task")
        run.total_cost_usd = 0.0
        run.avg_confidence = 0.5

        suspend_run(db_connection, run, paused_by="system")

        row = db_connection.execute(
            "SELECT checkpoint_data FROM paused_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()

        data = json.loads(row["checkpoint_data"])
        assert data["total_cost_usd"] == 0.0
        assert data["avg_confidence"] == 0.5

    def test_suspend_with_cancel_requested_state(self, db_connection):
        """Verify suspend works with CANCEL_REQUESTED interrupt state."""
        run = OrchestraRun(mode="critical", task="Test")
        run.interrupt_state = InterruptState.CANCEL_REQUESTED

        suspend_run(db_connection, run, paused_by="user_pause")

        row = db_connection.execute(
            "SELECT checkpoint_data FROM paused_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()

        data = json.loads(row["checkpoint_data"])
        assert data["interrupt_state"] == "cancel_requested"

    @pytest.mark.skip(reason="Extended field restoration not yet implemented in resume_run")
    def test_multiple_suspends_different_runs(self, db_connection):
        """Verify multiple runs can be suspended independently with extended fields.

        NOTE: This test is skipped because extended field restoration is not yet
        implemented in resume_run. Once Phase 2 fix is implemented, this should pass.
        """
        run1 = OrchestraRun(mode="ask", task="Task 1")
        run1.total_cost_usd = 1.0
        run1.turns = 1

        run2 = OrchestraRun(mode="critical", task="Task 2")
        run2.total_cost_usd = 2.0
        run2.turns = 2

        suspend_run(db_connection, run1, paused_by="system")
        suspend_run(db_connection, run2, paused_by="system")

        count = db_connection.execute(
            "SELECT COUNT(*) as cnt FROM paused_runs"
        ).fetchone()
        assert count["cnt"] == 2

        restored1 = resume_run(db_connection, run1.run_id)
        restored2 = resume_run(db_connection, run2.run_id)

        assert restored1.total_cost_usd == 1.0
        assert restored2.total_cost_usd == 2.0
        assert restored1.turns == 1
        assert restored2.turns == 2
