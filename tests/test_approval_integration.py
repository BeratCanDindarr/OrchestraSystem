"""Integration tests for approval gates in ask/dual/critical modes."""
import pytest
from unittest.mock import Mock, patch

from orchestra.engine.runner import (
    run_ask,
    run_dual,
    run_critical,
)
from orchestra.models import RunStatus, ApprovalState
from orchestra.storage.db import get_db


class TestApprovalGateIntegration:
    """Integration tests for approval policies across all modes."""

    def setup_method(self):
        """Clear database before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM eval_runs")
            connection.execute("DELETE FROM eval_scenarios")
            connection.execute("DELETE FROM eval_results")
        connection.close()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner.suspend_run_to_db")
    def test_ask_mode_triggers_approval(self, mock_suspend, mock_policy, mock_run_parallel):
        """Ask mode pauses when approval policy triggers."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None

        run = run_ask(alias="cdx-fast", prompt="Test task")

        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING
        mock_policy.assert_called_once()
        mock_suspend.assert_called_once()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner._finalize")
    def test_ask_mode_no_approval_finalizes(self, mock_finalize, mock_policy, mock_run_parallel):
        """Ask mode finalizes when no approval needed."""
        mock_policy.return_value = False
        mock_run_parallel.return_value = None

        run = run_ask(alias="cdx-fast", prompt="Test task")

        mock_policy.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._record_round1_review")
    @patch("orchestra.engine.runner._run_verification_loop")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner.suspend_run_to_db")
    def test_dual_mode_triggers_approval(
        self, mock_suspend, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode pauses when approval policy triggers."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None

        run = run_dual(prompt="Test task")

        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING
        mock_policy.assert_called_once()
        mock_suspend.assert_called_once()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._record_round1_review")
    @patch("orchestra.engine.runner._run_verification_loop")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner._synthesize_if_possible")
    @patch("orchestra.engine.runner._finalize")
    def test_dual_mode_no_approval_finalizes(
        self, mock_finalize, mock_synthesize, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode synthesizes and finalizes when no approval needed."""
        mock_policy.return_value = False
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None
        mock_synthesize.return_value = None

        run = run_dual(prompt="Test task")

        mock_policy.assert_called_once()
        mock_synthesize.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._record_round1_review")
    @patch("orchestra.engine.runner._run_verification_loop")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner.suspend_run_to_db")
    def test_critical_mode_triggers_approval(
        self, mock_suspend, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Critical mode pauses when approval policy triggers."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None

        run = run_critical(prompt="Test task")

        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING
        mock_policy.assert_called_once()
        mock_suspend.assert_called_once()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._record_round1_review")
    @patch("orchestra.engine.runner._run_verification_loop")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner._synthesize_if_possible")
    @patch("orchestra.engine.runner._finalize")
    def test_critical_mode_no_approval_finalizes(
        self, mock_finalize, mock_synthesize, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Critical mode synthesizes and finalizes when no approval needed."""
        mock_policy.return_value = False
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None
        mock_synthesize.return_value = None

        run = run_critical(prompt="Test task")

        mock_policy.assert_called_once()
        mock_synthesize.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner.suspend_run_to_db")
    def test_ask_approval_persists_to_database(self, mock_suspend, mock_policy, mock_run_parallel):
        """Ask mode persists approval state to database for resumption."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None

        run = run_ask(alias="cdx-fast", prompt="Test task")

        # Verify suspend_run_to_db was called with correct parameters
        assert mock_suspend.call_count == 1
        call_args = mock_suspend.call_args
        suspended_run = call_args[0][1]
        assert suspended_run.approval_state == ApprovalState.PENDING
        assert suspended_run.mode == "ask"

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._record_round1_review")
    @patch("orchestra.engine.runner._run_verification_loop")
    @patch("orchestra.engine.runner._should_require_approval")
    @patch("orchestra.engine.runner.suspend_run_to_db")
    def test_dual_approval_persists_to_database(
        self, mock_suspend, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode persists approval state to database for resumption."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None

        run = run_dual(prompt="Test task")

        # Verify suspend_run_to_db was called with correct parameters
        assert mock_suspend.call_count == 1
        call_args = mock_suspend.call_args
        suspended_run = call_args[0][1]
        assert suspended_run.approval_state == ApprovalState.PENDING
        assert suspended_run.mode == "dual"

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._should_require_approval")
    def test_approval_gate_checkpoint_written(self, mock_policy, mock_run_parallel):
        """Approval gate checkpoint is written when approval required."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None

        with patch("orchestra.engine.runner.artifacts.write_checkpoint") as mock_checkpoint:
            run = run_ask(alias="cdx-fast", prompt="Test task")

            # Verify checkpoint was written for approval gate
            checkpoint_calls = [call for call in mock_checkpoint.call_args_list if "approval_gate" in str(call)]
            assert len(checkpoint_calls) > 0

    @patch("orchestra.engine.runner.run_parallel")
    @patch("orchestra.engine.runner._should_require_approval")
    def test_approval_event_appended(self, mock_policy, mock_run_parallel):
        """Approval event is appended to run timeline."""
        mock_policy.return_value = True
        mock_run_parallel.return_value = None

        with patch("orchestra.engine.runner.append_event") as mock_event:
            run = run_ask(alias="cdx-fast", prompt="Test task")

            # Verify approval_requested event was appended
            approval_events = [call for call in mock_event.call_args_list if "approval_requested" in str(call)]
            assert len(approval_events) > 0
