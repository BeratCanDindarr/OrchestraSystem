"""Tests for Phase 4: Approval state machine edge cases and bug fixes."""
import pytest
from unittest.mock import patch

from orchestra.engine.runner import resume_run
from orchestra.models import RunStatus, OrchestraRun, AgentRun, AgentStatus
from orchestra.state import ApprovalState
from orchestra.storage.db import get_db
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db, resume_run as resume_run_from_db


class TestApprovalPhase4EdgeCases:
    """Test edge cases and bug fixes in approval state machine."""

    def setup_method(self):
        """Clear paused_runs table before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM paused_runs")
        connection.close()

    # Bug 1 Tests: Enum type restoration
    def test_status_restored_as_enum(self):
        """Resuming run restores status as RunStatus enum, not string."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        # Must be RunStatus enum, not string
        assert isinstance(resumed_run.status, RunStatus)
        assert resumed_run.status == RunStatus.WAITING_APPROVAL

    def test_approval_state_restored_as_enum(self):
        """Resuming run restores approval_state as ApprovalState enum, not string."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        # Must be ApprovalState enum, not string
        assert isinstance(resumed_run.approval_state, ApprovalState)
        assert resumed_run.approval_state == ApprovalState.PENDING

    # Edge case guard tests
    @patch("orchestra.engine.runner._finalize")
    @patch("orchestra.engine.runner.artifacts")
    def test_double_resume_raises_value_error(self, mock_artifacts, mock_finalize):
        """Resuming same run_id twice raises ValueError on second attempt."""
        # Mock artifacts to return None (no fallback manifest)
        mock_artifacts.load_manifest.return_value = None
        mock_artifacts.latest_checkpoint.return_value = None

        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        # First resume — should succeed
        first_resumed = resume_run(run.run_id, approval_decision="approve")
        assert first_resumed is not None
        assert first_resumed.approval_state == ApprovalState.APPROVED

        # Second resume — should raise ValueError (paused_run entry deleted)
        with pytest.raises(ValueError, match="Run not found|Run manifest missing"):
            resume_run(run.run_id, approval_decision="approve")

    @patch("orchestra.engine.runner._finalize")
    def test_resume_completed_run_raises_value_error(self, mock_finalize):
        """Resuming a run that is already COMPLETED raises ValueError."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Manually update paused run's status to COMPLETED (simulating corrupt state)
        connection.execute(
            "UPDATE paused_runs SET checkpoint_data = json_replace(checkpoint_data, '$.status', 'completed') WHERE run_id = ?",
            (run.run_id,)
        )
        connection.commit()
        connection.close()

        # Attempt to resume COMPLETED run — should raise ValueError
        with pytest.raises(ValueError, match="Cannot resume run.*status=completed"):
            resume_run(run.run_id, approval_decision="approve")

    def test_resume_invalid_run_id_raises_value_error(self):
        """Resuming nonexistent run raises ValueError."""
        with pytest.raises(ValueError, match="Run not found"):
            resume_run("nonexistent_run_id", approval_decision="approve")

    def test_checkpoint_version_preserved_not_incremented(self):
        """Checkpoint version is preserved after resume, not incremented."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.checkpoint_version = 5
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        resumed_run = resume_run_from_db(connection, run.run_id)
        connection.close()

        # Version should be preserved, not incremented
        assert resumed_run.checkpoint_version == 5
