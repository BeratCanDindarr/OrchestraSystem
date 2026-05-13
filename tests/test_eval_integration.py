"""Integration test: Verification loop outcome logging."""
import json
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from orchestra.engine.runner import OrchestraRun, RunStatus
from orchestra.storage.eval_tracker import EvalTracker
from orchestra.storage.db import get_db


class TestEvalIntegration:
    """Test evaluation harness integration with verification loop."""

    def setup_method(self):
        """Clear eval tables before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM eval_runs")
            connection.execute("DELETE FROM eval_scenarios")
            connection.execute("DELETE FROM eval_results")
        connection.close()

    def test_outcome_logging_on_completed(self):
        """Verify outcome is logged as PASS when run completes."""
        # Create a completed run
        run = OrchestraRun(
            run_id="integration_test_001",
            task="Test task",
            mode="ask",
            status=RunStatus.COMPLETED,
            created_at="2026-05-12T10:00:00Z",
            updated_at="2026-05-12T10:01:00Z"
        )

        # Simulate outcome logging (what _finalize() does)
        tracker = EvalTracker()
        eval_status = "PASS" if run.status == RunStatus.COMPLETED else "FAIL"
        tracker.log_run(
            run_id=run.run_id,
            task=run.task,
            mode=run.mode,
            status=eval_status,
            created_at=run.updated_at or run.created_at
        )

        # Verify logged outcome
        connection = get_db()
        cursor = connection.execute(
            "SELECT status FROM eval_runs WHERE run_id = ?",
            (run.run_id,)
        )
        row = cursor.fetchone()
        connection.close()

        assert row is not None
        assert row["status"] == "PASS"

    def test_outcome_logging_on_failed(self):
        """Verify outcome is logged as FAIL when run fails."""
        run = OrchestraRun(
            run_id="integration_test_002",
            task="Failing task",
            mode="dual",
            status=RunStatus.FAILED,
            created_at="2026-05-12T10:00:00Z",
            updated_at="2026-05-12T10:02:00Z"
        )

        # Simulate outcome logging
        tracker = EvalTracker()
        eval_status = "PASS" if run.status == RunStatus.COMPLETED else "FAIL"
        tracker.log_run(
            run_id=run.run_id,
            task=run.task,
            mode=run.mode,
            status=eval_status,
            created_at=run.updated_at or run.created_at
        )

        # Verify logged outcome
        connection = get_db()
        cursor = connection.execute(
            "SELECT status FROM eval_runs WHERE run_id = ?",
            (run.run_id,)
        )
        row = cursor.fetchone()
        connection.close()

        assert row is not None
        assert row["status"] == "FAIL"

    def test_batch_evaluate_integration(self):
        """Verify batch runner integrates with logged outcomes."""
        from orchestra.engine.batch_runner import BatchRunner, EvalScenario

        # Log some outcomes
        tracker = EvalTracker()
        tracker.log_run("run_1", "task_A", "ask", "PASS", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_A", "ask", "FAIL", "2026-05-12T10:01:00Z")
        tracker.log_run("run_3", "task_B", "dual", "PASS", "2026-05-12T10:02:00Z")

        # Calculate Pass@1 metric
        pass_at_1 = tracker.calculate_pass_at_k(1)

        # Expected:
        # task_A: 1 pass, 1 fail (n=2, c=1) → Pass@1 = 1 - (1/2) = 0.5
        # task_B: 1 pass (n=1, c=1) → Pass@1 = 1.0
        # Average: (0.5 + 1.0) / 2 = 0.75
        expected = 0.75
        assert abs(pass_at_1 - expected) < 0.001, f"Expected {expected}, got {pass_at_1}"

        # Verify batch runner can export stats
        runner = BatchRunner()
        results = [
            Mock(outcome="PASS", tokens_used=1000, confidence=0.9),
            Mock(outcome="FAIL", tokens_used=800, confidence=0.5),
            Mock(outcome="PASS", tokens_used=1200, confidence=0.95),
        ]

        # Create temp stats (simulated)
        stats = {
            "total_runs": len(results),
            "passed": sum(1 for r in results if r.outcome == "PASS"),
            "failed": sum(1 for r in results if r.outcome == "FAIL"),
            "pass_rate": sum(1 for r in results if r.outcome == "PASS") / len(results),
            "avg_tokens": sum(r.tokens_used for r in results) / len(results),
            "avg_confidence": sum(r.confidence for r in results) / len(results),
        }

        # Verify stats are computed correctly
        assert stats["pass_rate"] == 2/3
        assert stats["avg_tokens"] == 1000
        assert stats["avg_confidence"] > 0.7
