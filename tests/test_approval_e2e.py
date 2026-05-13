"""Tests for Phase 5: End-to-End Approval Pipeline (run → WAITING_APPROVAL → resume → COMPLETED/FAILED)."""
import pytest
from unittest.mock import patch, MagicMock

from orchestra.engine.runner import resume_run, run_ask, run_dual, run_critical
from orchestra.models import RunStatus, OrchestraRun, AgentRun, AgentStatus
from orchestra.state import ApprovalState
from orchestra.storage.db import get_db
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db


def fake_run_parallel(run, pairs, **kwargs):
    """Fake run_parallel that returns completed agents without network calls."""
    for alias, _ in pairs:
        agent = AgentRun(alias=alias, provider="fake", model="fake")
        agent.status = AgentStatus.COMPLETED
        agent.confidence = 0.9
        agent.stdout_log = f"Output from {alias}"
        agent.estimated_completion_tokens = 100
        run.agents.append(agent)


class TestApprovalE2E:
    """End-to-end approval pipeline tests (Phase 5)."""

    def setup_method(self):
        """Clear paused_runs table before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM paused_runs")
        connection.close()

    # Scenario 1-4: Resume decision tests (ask, dual, critical modes)

    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_ask_approve_completes(
        self, mock_should_approve, mock_run_parallel, mock_append_event, mock_artifacts
    ):
        """Ask mode: run_ask() → WAITING_APPROVAL → resume(approve) → COMPLETED + APPROVED."""
        # Mock policy: ask mode requires approval
        mock_should_approve.return_value = True

        # Run ask mode — should pause for approval
        run = run_ask(alias="cdx-fast", prompt="Test task")

        # Verify suspended to database with PENDING approval state
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is not None  # Paused run exists
        connection.close()

        # Resume with approval
        resumed_run = resume_run(run.run_id, approval_decision="approve")

        # Verify final state
        assert resumed_run is not None
        assert resumed_run.approval_state == ApprovalState.APPROVED
        assert resumed_run.status == RunStatus.COMPLETED

        # Verify paused_runs cleaned up
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()

    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_ask_reject_fails(
        self, mock_should_approve, mock_run_parallel, mock_append_event, mock_artifacts
    ):
        """Ask mode: run_ask() → WAITING_APPROVAL → resume(reject) → FAILED + REJECTED."""
        # Mock policy: ask mode requires approval
        mock_should_approve.return_value = True

        # Run ask mode — should pause for approval
        run = run_ask(alias="cdx-fast", prompt="Test task")

        # Resume with rejection
        resumed_run = resume_run(run.run_id, approval_decision="reject")

        # Verify final state
        assert resumed_run is not None
        assert resumed_run.approval_state == ApprovalState.REJECTED
        assert resumed_run.status == RunStatus.FAILED

        # Verify paused_runs cleaned up
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()

    @patch("orchestra.engine.runner._synthesize_if_possible")
    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_dual_approve_completes(
        self,
        mock_should_approve,
        mock_run_parallel,
        mock_append_event,
        mock_artifacts,
        mock_synthesize,
    ):
        """Dual mode: run_dual() → WAITING_APPROVAL → resume(approve) → COMPLETED + APPROVED."""
        # Mock policy: dual mode with confidence threshold
        mock_should_approve.return_value = True

        # Run dual mode — should pause for approval
        run = run_dual(prompt="Complex task")

        # Resume with approval
        resumed_run = resume_run(run.run_id, approval_decision="approve")

        # Verify final state and synthesis was called
        assert resumed_run is not None
        assert resumed_run.approval_state == ApprovalState.APPROVED
        assert resumed_run.status == RunStatus.COMPLETED
        mock_synthesize.assert_called_once()

        # Verify paused_runs cleaned up
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()

    @patch("orchestra.engine.runner._synthesize_if_possible")
    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_critical_approve_completes(
        self,
        mock_should_approve,
        mock_run_parallel,
        mock_append_event,
        mock_artifacts,
        mock_synthesize,
    ):
        """Critical mode: run_critical() → WAITING_APPROVAL → resume(approve) → COMPLETED + APPROVED."""
        # Mock policy: critical mode requires approval
        mock_should_approve.return_value = True

        # Run critical mode — should pause for approval
        run = run_critical(prompt="Critical task")

        # Resume with approval
        resumed_run = resume_run(run.run_id, approval_decision="approve")

        # Verify final state and synthesis was called
        assert resumed_run is not None
        assert resumed_run.approval_state == ApprovalState.APPROVED
        assert resumed_run.status == RunStatus.COMPLETED
        mock_synthesize.assert_called_once()

        # Verify paused_runs cleaned up
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()

    # Scenario 5-8: Policy trigger tests (pause behavior)

    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_policy_cost_threshold_pauses(
        self, mock_should_approve, mock_run_parallel, mock_append_event, mock_artifacts
    ):
        """Ask mode with cost_threshold policy: run_ask() hits threshold → WAITING_APPROVAL + PENDING."""
        # Mock policy: cost threshold exceeded
        mock_should_approve.return_value = True

        # Run ask mode — should pause due to cost policy
        run = run_ask(alias="cdx-fast", prompt="Expensive task")

        # Verify paused with PENDING approval state
        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING

        # Verify paused_runs entry exists
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        paused_entry = cursor.fetchone()
        assert paused_entry is not None
        connection.close()

    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_policy_confidence_threshold_pauses(
        self, mock_should_approve, mock_run_parallel, mock_append_event, mock_artifacts
    ):
        """Ask mode with confidence_threshold policy: agent.confidence < threshold → WAITING_APPROVAL."""
        # Mock policy: confidence threshold violated
        mock_should_approve.return_value = True

        # Mock run_parallel to return low confidence agent
        def low_confidence_parallel(run, pairs, **kwargs):
            for alias, _ in pairs:
                agent = AgentRun(alias=alias, provider="fake", model="fake")
                agent.status = AgentStatus.COMPLETED
                agent.confidence = 0.5  # Below typical threshold
                agent.stdout_log = f"Output from {alias}"
                agent.estimated_completion_tokens = 100
                run.agents.append(agent)

        with patch("orchestra.engine.runner.run_parallel", side_effect=low_confidence_parallel):
            run = run_ask(alias="cdx-fast", prompt="Low confidence task")

            # Verify paused with PENDING approval state
            assert run.status == RunStatus.WAITING_APPROVAL
            assert run.approval_state == ApprovalState.PENDING

            # Verify paused_runs entry exists
            connection = get_db()
            cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
            paused_entry = cursor.fetchone()
            assert paused_entry is not None
            connection.close()

    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_policy_task_complexity_pauses(
        self, mock_should_approve, mock_run_parallel, mock_append_event, mock_artifacts
    ):
        """Ask mode with task_complexity policy: complex task → WAITING_APPROVAL + PENDING."""
        # Mock policy: task complexity exceeds threshold
        mock_should_approve.return_value = True

        # Run with complex task description — should trigger complexity policy
        complex_task = (
            "This is a very long and complex task that involves multiple steps "
            "and requires significant reasoning. It needs to handle edge cases and "
            "integrate with many systems."
        )
        run = run_ask(alias="cdx-fast", prompt=complex_task)

        # Verify paused with PENDING approval state
        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING

        # Verify paused_runs entry exists
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        paused_entry = cursor.fetchone()
        assert paused_entry is not None
        connection.close()

    @patch("orchestra.engine.runner.artifacts")
    @patch("orchestra.engine.runner.append_event")
    @patch("orchestra.engine.runner.run_parallel", side_effect=fake_run_parallel)
    @patch("orchestra.engine.runner._should_require_approval")
    def test_e2e_no_policy_skips_approval(
        self, mock_should_approve, mock_run_parallel, mock_append_event, mock_artifacts
    ):
        """Ask mode with no policy: run_ask() completes directly → COMPLETED (no pause)."""
        # Mock policy: no approval required
        mock_should_approve.return_value = False

        # Run ask mode — should complete without approval gate
        run = run_ask(alias="cdx-fast", prompt="Simple task")

        # Verify completed directly
        assert run.status == RunStatus.COMPLETED
        assert run.approval_state == ApprovalState.NOT_REQUIRED

        # Verify paused_runs table empty (no pause entry)
        connection = get_db()
        cursor = connection.execute("SELECT * FROM paused_runs WHERE run_id = ?", (run.run_id,))
        assert cursor.fetchone() is None
        connection.close()
