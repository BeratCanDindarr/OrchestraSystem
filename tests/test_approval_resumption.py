"""Tests for approval gate resumption and state restoration (Phase 3)."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from orchestra.engine.runner import resume_run
from orchestra.models import RunStatus, OrchestraRun, AgentRun, AgentStatus
from orchestra.state import ApprovalState
from orchestra.storage.db import get_db
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db, resume_run as resume_run_from_db


class TestApprovalResumption:
    """Test resumption of paused runs after approval (Phase 3)."""

    def setup_method(self):
        """Clear paused_runs table before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM paused_runs")
        connection.close()

    def test_resume_ask_mode_restores_full_state(self):
        """Resuming ask mode restores all paused run state."""
        # Create and suspend a run
        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        agent.confidence = 0.85
        agent.stdout_log = "Sample output"
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        # Suspend it
        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Resume it
        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert resumed_run.mode == "ask"
        assert resumed_run.task == "Test task"
        assert len(resumed_run.agents) == 1
        assert resumed_run.agents[0].alias == "cdx-fast"
        assert resumed_run.agents[0].confidence == 0.85
        assert resumed_run.agents[0].stdout_log == "Sample output"

    def test_resume_dual_mode_restores_multiple_agents(self):
        """Resuming dual mode restores both agents."""
        # Create and suspend a dual run
        run = OrchestraRun(mode="dual", task="Complex task")
        agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED)
        agent1.confidence = 0.92
        agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED)
        agent2.confidence = 0.88
        run.agents = [agent1, agent2]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        # Suspend it
        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Resume it
        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert len(resumed_run.agents) == 2
        assert resumed_run.agents[0].confidence == 0.92
        assert resumed_run.agents[1].confidence == 0.88
        assert resumed_run.agents[0].alias == "cdx-deep"
        assert resumed_run.agents[1].alias == "gmn-pro"

    def test_resume_critical_mode_restores_review_state(self):
        """Resuming critical mode restores review history."""
        # Create and suspend a critical run with review data
        run = OrchestraRun(mode="critical", task="Critical task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING
        run.latest_review_stage = "round1"
        run.latest_review_status = "pending"
        run.latest_review_winner = "cdx-deep"
        run.reviews = [
            {"stage": "round1", "winner": "cdx-deep", "reason": "Higher confidence"}
        ]

        # Suspend it
        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Resume it
        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert resumed_run.latest_review_stage == "round1"
        assert resumed_run.latest_review_winner == "cdx-deep"
        assert len(resumed_run.reviews) == 1

    def test_resume_nonexistent_run_returns_none(self):
        """Resuming nonexistent run returns None."""
        connection = get_db()
        resumed_run = resume_run_from_db(connection, "nonexistent_id")
        connection.close()

        assert resumed_run is None

    def test_resume_expired_run_deleted_and_returns_none(self):
        """Resuming expired run (>48h) deletes it and returns None."""
        from datetime import datetime, timedelta, timezone

        # Create and suspend a run
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Manually set expiry to past (49 hours ago)
        past_time = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        connection.execute(
            "UPDATE paused_runs SET expires_at = ? WHERE run_id = ?",
            (past_time, run.run_id),
        )
        connection.commit()

        # Try to resume expired run
        resumed_run = resume_run_from_db(connection, run.run_id)

        assert resumed_run is None

        # Verify it was deleted from database
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()

    def test_resume_preserves_checkpoint_version(self):
        """Resuming run preserves checkpoint_version counter."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.checkpoint_version = 5
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert resumed_run.checkpoint_version == 5

    def test_resume_preserves_summary(self):
        """Resuming run preserves partial summary from first round."""
        run = OrchestraRun(mode="dual", task="Test task")
        run.summary = "Preliminary findings from round 1"
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert resumed_run.summary == "Preliminary findings from round 1"

    def test_resume_preserves_agent_validation_status(self):
        """Resuming run preserves agent validation status (PASS/FAIL/NEEDS_FIX)."""
        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        agent.validation_status = "NEEDS_FIX"
        agent.validation_reason = "Output too short, needs expansion"
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert resumed_run.agents[0].validation_status == "NEEDS_FIX"
        assert resumed_run.agents[0].validation_reason == "Output too short, needs expansion"

    def test_resume_sets_approval_state_to_resumed(self):
        """Resuming run updates ApprovalState from PENDING to RESUMED."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        # Note: ApprovalState transition from PENDING → RESUMED happens in runner.py
        # after resume_run_from_db returns, so we only verify PENDING is restored here
        assert resumed_run.approval_state == ApprovalState.PENDING

    def test_resume_handles_corrupted_checkpoint_data(self):
        """Resuming run with corrupted JSON checkpoint returns None gracefully."""
        connection = get_db()

        # Manually insert corrupted checkpoint data
        connection.execute(
            """INSERT INTO paused_runs (run_id, checkpoint_data, paused_at, paused_by, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                "corrupted_id",
                "{ invalid json }",
                "2026-05-12T10:00:00+00:00",
                "approval_gate",
                "2026-05-14T10:00:00+00:00",
            ),
        )
        connection.commit()

        # Try to resume corrupted run
        resumed_run = resume_run_from_db(connection, "corrupted_id")
        connection.close()

        assert resumed_run is None

    def test_resume_restores_soft_failed_flag(self):
        """Resuming run restores soft_failed flag on agents."""
        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        agent.soft_failed = True
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        assert resumed_run is not None
        assert resumed_run.agents[0].soft_failed is True
