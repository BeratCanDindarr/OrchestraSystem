"""Integration test: Verification loop outcome logging."""
import json
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from orchestra.engine.runners import OrchestraRun, RunStatus
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

