"""Integration tests for approval gates in ask/dual/critical modes."""
import pytest
from unittest.mock import Mock, patch

from orchestra.engine.runners import (
    run_ask,
    run_dual,
    run_critical,
)
from orchestra.models import RunStatus, ApprovalState, AgentRun, AgentStatus
from orchestra.engine.policy_engine import PolicyDecision
from orchestra.storage.db import get_db


class TestApprovalGateIntegration:
    """Integration tests for approval policies across all modes."""

    def setup_method(self):
        """Clear database and mock autonomy policy before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM eval_runs")
            connection.execute("DELETE FROM eval_scenarios")
            connection.execute("DELETE FROM eval_results")
        connection.close()
        self.autonomy_patch = patch("orchestra.config.autonomy_policy_config", return_value={"mode_gates": []})
        self.autonomy_patch.start()

    def teardown_method(self):
        self.autonomy_patch.stop()

    @patch("orchestra.engine.runners.common.run_parallel")
    @patch("orchestra.engine.runners.common._evaluate_policy")
    @patch("orchestra.engine.runners.common.suspend_run_to_db")
    def test_ask_mode_triggers_approval(self, mock_suspend, mock_policy, mock_run_parallel):
        """Ask mode pauses when approval policy triggers."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
        mock_run_parallel.return_value = None

        run = run_ask(alias="cdx-fast", prompt="Test task")

        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING
        mock_policy.assert_called_once()
        mock_suspend.assert_called_once()

    @patch("orchestra.engine.runners.common.run_parallel")
    @patch("orchestra.engine.runners.common._evaluate_policy")
    @patch("orchestra.engine.runners.common._finalize")
    def test_ask_mode_no_approval_finalizes(self, mock_finalize, mock_policy, mock_run_parallel):
        """Ask mode finalizes when no approval needed."""
        mock_policy.return_value = PolicyDecision("allow", "mock", "test")
        mock_run_parallel.return_value = None

        run = run_ask(alias="cdx-fast", prompt="Test task")

        mock_policy.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runners.common.run_parallel")
    @patch("orchestra.engine.runners.common._evaluate_policy")
    def test_ask_mode_copies_completed_output_into_summary(self, mock_policy, mock_run_parallel):
        """Ask mode exposes completed agent output via run.summary for downstream consumers."""
        mock_policy.return_value = PolicyDecision("allow", "mock", "test")

        def _populate_agent(run, pairs, **kwargs):
            agent = AgentRun(alias="cdx-fast", provider="codex", model="gpt-5.4/low")
            agent.status = AgentStatus.COMPLETED
            agent.stdout_log = "OK"
            run.agents.append(agent)

        mock_run_parallel.side_effect = _populate_agent

        run = run_ask(alias="cdx-fast", prompt="Summarise the approval integration check")

        assert run.summary == "OK"

    @patch("orchestra.engine.runners.dual_runner.run_parallel")
    @patch("orchestra.engine.runners.dual_runner._record_round1_review")
    @patch("orchestra.engine.runners.dual_runner._run_verification_loop")
    @patch("orchestra.engine.runners.dual_runner._evaluate_policy")
    @patch("orchestra.engine.runners.dual_runner.suspend_run_to_db")
    def test_dual_mode_triggers_approval(
        self, mock_suspend, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode pauses when approval policy triggers."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None

        run = run_dual(prompt="Test task")

        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING
        mock_policy.assert_called_once()
        mock_suspend.assert_called_once()

    @patch("orchestra.engine.runners.dual_runner.run_parallel")
    @patch("orchestra.engine.runners.dual_runner._record_round1_review")
    @patch("orchestra.engine.runners.dual_runner._run_verification_loop")
    @patch("orchestra.engine.runners.dual_runner._evaluate_policy")
    @patch("orchestra.engine.runners.dual_runner._synthesize_if_possible")
    @patch("orchestra.engine.runners.dual_runner._finalize")
    def test_dual_mode_no_approval_finalizes(
        self, mock_finalize, mock_synthesize, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode synthesizes and finalizes when no approval needed."""
        mock_policy.return_value = PolicyDecision("allow", "mock", "test")
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None
        mock_synthesize.return_value = None

        run = run_dual(prompt="Test task")

        mock_policy.assert_called_once()
        mock_synthesize.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runners.dual_runner.run_parallel")
    @patch("orchestra.engine.runners.dual_runner._record_round1_review")
    @patch("orchestra.engine.runners.dual_runner._run_verification_loop")
    @patch("orchestra.engine.runners.dual_runner._evaluate_policy")
    @patch("orchestra.engine.runners.dual_runner._synthesize_if_possible")
    def test_dual_mode_falls_back_to_completed_output_when_no_summary(
        self, mock_synthesize, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode exposes a usable summary even when synthesis is skipped or leaves none."""
        mock_policy.return_value = PolicyDecision("allow", "mock", "test")
        mock_review.return_value = None
        mock_verify.return_value = None
        mock_synthesize.return_value = None

        def _populate_agents(run, pairs, **kwargs):
            for alias, _ in pairs:
                agent = AgentRun(alias=alias, provider="fake", model="fake")
                agent.status = AgentStatus.COMPLETED
                agent.stdout_log = f"Output from {alias}"
                run.agents.append(agent)

        mock_run_parallel.side_effect = _populate_agents

        run = run_dual(prompt="Test task")

        assert "Output from cdx-deep" in run.summary
        assert "Provenance Graph" in run.summary

    @patch("orchestra.engine.runners.dual_runner.run_parallel")
    @patch("orchestra.engine.runners.dual_runner._record_round1_review")
    @patch("orchestra.engine.runners.dual_runner._run_verification_loop")
    @patch("orchestra.engine.runners.dual_runner._evaluate_policy")
    @patch("orchestra.engine.runners.dual_runner._synthesize_if_possible")
    def test_dual_mode_does_not_use_verifier_output_as_summary_fallback(
        self, mock_synthesize, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Verifier output must not replace the user-facing candidate answer."""
        mock_policy.return_value = PolicyDecision("allow", "mock", "test")
        mock_review.return_value = None
        mock_verify.return_value = None
        mock_synthesize.return_value = None

        def _populate_agents(run, pairs, **kwargs):
            first = AgentRun(alias="cdx-deep", provider="fake", model="fake")
            first.status = AgentStatus.COMPLETED
            first.stdout_log = "Candidate output"
            run.agents.append(first)

            verifier = AgentRun(alias="verifier", provider="fake", model="fake")
            verifier.status = AgentStatus.COMPLETED
            verifier.stdout_log = "Verifier output"
            run.agents.append(verifier)

        mock_run_parallel.side_effect = _populate_agents

        run = run_dual(prompt="Test task")

        assert "Candidate output" in run.summary
        assert "Provenance Graph" in run.summary

    @patch("orchestra.engine.runners.critical_runner.run_parallel")
    @patch("orchestra.engine.runners.critical_runner._record_round1_review")
    @patch("orchestra.engine.runners.critical_runner._run_verification_loop")
    @patch("orchestra.engine.runners.critical_runner._evaluate_policy")
    @patch("orchestra.engine.runners.critical_runner.suspend_run_to_db")
    def test_critical_mode_triggers_approval(
        self, mock_suspend, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Critical mode pauses when approval policy triggers."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None

        run = run_critical(prompt="Test task")

        assert run.status == RunStatus.WAITING_APPROVAL
        assert run.approval_state == ApprovalState.PENDING
        mock_policy.assert_called_once()
        mock_suspend.assert_called_once()

    @patch("orchestra.engine.runners.critical_runner.run_parallel")
    @patch("orchestra.engine.runners.critical_runner._record_round1_review")
    @patch("orchestra.engine.runners.critical_runner._run_verification_loop")
    @patch("orchestra.engine.runners.critical_runner._evaluate_policy")
    @patch("orchestra.engine.runners.critical_runner._synthesize_if_possible")
    @patch("orchestra.engine.runners.critical_runner._finalize")
    def test_critical_mode_no_approval_finalizes(
        self, mock_finalize, mock_synthesize, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Critical mode synthesizes and finalizes when no approval needed."""
        mock_policy.return_value = PolicyDecision("allow", "mock", "test")
        mock_run_parallel.return_value = None
        mock_review.return_value = None
        mock_verify.return_value = None
        mock_synthesize.return_value = None

        run = run_critical(prompt="Test task")

        mock_policy.assert_called_once()
        mock_synthesize.assert_called_once()
        mock_finalize.assert_called_once()

    @patch("orchestra.engine.runners.common.run_parallel")
    @patch("orchestra.engine.runners.common._evaluate_policy")
    @patch("orchestra.engine.runners.common.suspend_run_to_db")
    def test_ask_approval_persists_to_database(self, mock_suspend, mock_policy, mock_run_parallel):
        """Ask mode persists approval state to database for resumption."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
        mock_run_parallel.return_value = None

        run = run_ask(alias="cdx-fast", prompt="Test task")

        # Verify suspend_run_to_db was called with correct parameters
        assert mock_suspend.call_count == 1
        call_args = mock_suspend.call_args
        suspended_run = call_args[0][1]
        assert suspended_run.approval_state == ApprovalState.PENDING
        assert suspended_run.mode == "ask"

    @patch("orchestra.engine.runners.dual_runner.run_parallel")
    @patch("orchestra.engine.runners.dual_runner._record_round1_review")
    @patch("orchestra.engine.runners.dual_runner._run_verification_loop")
    @patch("orchestra.engine.runners.dual_runner._evaluate_policy")
    @patch("orchestra.engine.runners.dual_runner.suspend_run_to_db")
    def test_dual_approval_persists_to_database(
        self, mock_suspend, mock_policy, mock_verify, mock_review, mock_run_parallel
    ):
        """Dual mode persists approval state to database for resumption."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
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

    @patch("orchestra.engine.runners.common.run_parallel")
    @patch("orchestra.engine.runners.common._evaluate_policy")
    def test_approval_gate_checkpoint_written(self, mock_policy, mock_run_parallel):
        """Approval gate checkpoint is written when approval required."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
        mock_run_parallel.return_value = None

        with patch("orchestra.engine.runners.common.artifacts.write_checkpoint") as mock_checkpoint:
            run = run_ask(alias="cdx-fast", prompt="Test task")

            # Verify checkpoint was written for approval gate
            checkpoint_calls = [call for call in mock_checkpoint.call_args_list if "approval_gate" in str(call)]
            assert len(checkpoint_calls) > 0

    @patch("orchestra.engine.runners.common.run_parallel")
    @patch("orchestra.engine.runners.common._evaluate_policy")
    def test_approval_event_appended(self, mock_policy, mock_run_parallel):
        """Approval event is appended to run timeline."""
        mock_policy.return_value = PolicyDecision("gate", "mock", "test")
        mock_run_parallel.return_value = None

        with patch("orchestra.engine.runners.common.append_event") as mock_event:
            run = run_ask(alias="cdx-fast", prompt="Test task")

            # Verify approval_requested event was appended
            approval_events = [call for call in mock_event.call_args_list if "approval_requested" in str(call)]
            assert len(approval_events) > 0
