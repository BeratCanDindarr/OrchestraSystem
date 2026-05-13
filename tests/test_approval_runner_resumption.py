"""Tests for resume_run() decision logic in runner.py (Phase 3)."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from orchestra.engine.runner import resume_run
from orchestra.models import RunStatus, OrchestraRun, AgentRun, AgentStatus
from orchestra.state import ApprovalState
from orchestra.storage.db import get_db
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db


class TestApprovalRunnerResumption:
    """Test resume_run() approval decision logic (Phase 3)."""

    def setup_method(self):
        """Clear paused_runs table before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM paused_runs")
        connection.close()

    @patch("orchestra.engine.runner._finalize")
    def test_resume_ask_mode_approved_finalizes(self, mock_finalize):
        """Resuming ask mode with approval=approve calls finalize."""
        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        # Resume with approval
        resumed_run = resume_run(run.run_id, approval_decision="approve")

        assert resumed_run.approval_state == ApprovalState.APPROVED
        assert resumed_run.status == RunStatus.RUNNING
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runner._finalize")
    def test_resume_ask_mode_rejected_terminates(self, mock_finalize):
        """Resuming ask mode with approval=reject fails run."""
        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        # Resume with rejection
        resumed_run = resume_run(run.run_id, approval_decision="reject")

        assert resumed_run.approval_state == ApprovalState.REJECTED
        assert resumed_run.status == RunStatus.FAILED
        mock_finalize.assert_not_called()

    @patch("orchestra.engine.runner._finalize")
    @patch("orchestra.engine.runner._synthesize_if_possible")
    def test_resume_dual_mode_approved_synthesizes(self, mock_synthesize, mock_finalize):
        """Resuming dual mode with approval=approve synthesizes then finalizes."""
        run = OrchestraRun(mode="dual", task="Test task")
        agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED)
        agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED)
        run.agents = [agent1, agent2]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        # Resume with approval
        resumed_run = resume_run(run.run_id, approval_decision="approve")

        assert resumed_run.approval_state == ApprovalState.APPROVED
        assert resumed_run.status == RunStatus.RUNNING
        mock_synthesize.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runner._finalize")
    @patch("orchestra.engine.runner._synthesize_if_possible")
    def test_resume_critical_mode_approved_synthesizes(self, mock_synthesize, mock_finalize):
        """Resuming critical mode with approval=approve synthesizes then finalizes."""
        run = OrchestraRun(mode="critical", task="Test task")
        agent1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED)
        agent2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED)
        run.agents = [agent1, agent2]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        # Resume with approval
        resumed_run = resume_run(run.run_id, approval_decision="approve")

        assert resumed_run.approval_state == ApprovalState.APPROVED
        assert resumed_run.status == RunStatus.RUNNING
        mock_synthesize.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runner.append_event")
    def test_resume_appends_approval_resumed_event(self, mock_append_event):
        """Resuming with approval appends approval_resumed event."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        resume_run(run.run_id, approval_decision="approve")

        # Verify approval_resumed event was appended
        approval_events = [call for call in mock_append_event.call_args_list if "approval_resumed" in str(call)]
        assert len(approval_events) > 0

    @patch("orchestra.engine.runner.append_event")
    def test_resume_appends_approval_rejected_event(self, mock_append_event):
        """Resuming with rejection appends approval_rejected event."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        resume_run(run.run_id, approval_decision="reject")

        # Verify approval_rejected event was appended
        approval_events = [call for call in mock_append_event.call_args_list if "approval_rejected" in str(call)]
        assert len(approval_events) > 0

    @patch("orchestra.engine.runner.artifacts")
    def test_resume_writes_approval_approved_checkpoint(self, mock_artifacts):
        """Resuming with approval writes approval_approved checkpoint."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        resume_run(run.run_id, approval_decision="approve")

        # Verify checkpoint was written
        checkpoint_calls = [call for call in mock_artifacts.write_checkpoint.call_args_list if "approval_approved" in str(call)]
        assert len(checkpoint_calls) > 0

    @patch("orchestra.engine.runner.artifacts")
    def test_resume_writes_approval_rejected_checkpoint(self, mock_artifacts):
        """Resuming with rejection writes approval_rejected checkpoint."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        resume_run(run.run_id, approval_decision="reject")

        # Verify checkpoint was written
        checkpoint_calls = [call for call in mock_artifacts.write_checkpoint.call_args_list if "approval_rejected" in str(call)]
        assert len(checkpoint_calls) > 0

    @patch("orchestra.engine.runner._finalize")
    def test_resume_defaults_to_approve(self, mock_finalize):
        """resume_run() defaults to approval_decision='approve'."""
        run = OrchestraRun(mode="ask", task="Test task")
        agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low", status=AgentStatus.COMPLETED)
        run.agents = [agent]
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")
        connection.close()

        # Resume without explicit approval_decision
        resumed_run = resume_run(run.run_id)

        assert resumed_run.approval_state == ApprovalState.APPROVED
        mock_finalize.assert_called_once()

    def test_resume_cleans_up_paused_run_on_approve(self):
        """Resuming with approval deletes paused_runs entry."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Verify paused run exists
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is not None
        connection.close()

        # Resume with approval
        resume_run(run.run_id, approval_decision="approve")

        # Verify paused run was deleted
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()

    def test_resume_cleans_up_paused_run_on_reject(self):
        """Resuming with rejection deletes paused_runs entry."""
        run = OrchestraRun(mode="ask", task="Test task")
        run.status = RunStatus.WAITING_APPROVAL
        run.approval_state = ApprovalState.PENDING

        connection = get_db()
        suspend_run_to_db(connection, run, paused_by="approval_gate")

        # Verify paused run exists
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is not None
        connection.close()

        # Resume with rejection
        resume_run(run.run_id, approval_decision="reject")

        # Verify paused run was deleted
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()
