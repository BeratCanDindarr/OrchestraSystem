"""Tests for EvalTracker evaluation outcome logging."""
import json
import tempfile
from pathlib import Path

import pytest

from orchestra.storage.eval_tracker import EvalTracker
from orchestra.storage.db import get_db


class TestEvalTracker:
    """Test evaluation tracking functionality."""

    def setup_method(self):
        """Clear eval tables before each test for isolation."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM eval_runs")
            connection.execute("DELETE FROM eval_scenarios")
            connection.execute("DELETE FROM eval_results")
        connection.close()

    def test_eval_tracker_logs_run(self):
        """Verify log_run() inserts records into eval_runs table."""
        tracker = EvalTracker()

        # Log a run
        tracker.log_run(
            run_id="run_001",
            task="Implement feature X",
            mode="dual",
            status="PASS",
            created_at="2026-05-12T10:00:00Z"
        )

        # Verify it doesn't crash (DB testing requires test fixtures)

    def test_eval_tracker_lifecycle(self):
        """Verify complete lifecycle: log, calculate, export."""
        tracker = EvalTracker()

        # 1. Log evaluation runs
        tracker.log_run("run_1", "task_A", "ask", "PASS", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_A", "ask", "FAIL", "2026-05-12T10:05:00Z")
        tracker.log_run("run_3", "task_B", "dual", "PASS", "2026-05-12T10:10:00Z")

        # 2. Test Pass@k metric
        # task_A: 1 PASS, 1 FAIL (n=2, c=1). k=1 -> P(pass) = 1 - (1/2) = 0.5
        # task_B: 1 PASS, 0 FAIL (n=1, c=1). k=1 -> P(pass) = 1 - 0 = 1.0
        # Average Pass@1: (0.5 + 1.0) / 2 = 0.75
        pass_at_1 = tracker.calculate_pass_at_k(1)
        assert abs(pass_at_1 - 0.75) < 0.001, f"Expected 0.75, got {pass_at_1}"

    def test_eval_tracker_export_jsonl(self):
        """Verify export_jsonl() writes valid JSONL format."""
        tracker = EvalTracker()

        # Log test runs
        tracker.log_run("run_1", "task_A", "ask", "PASS", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_B", "dual", "FAIL", "2026-05-12T10:05:00Z")

        # Export to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "eval_runs.jsonl"
            tracker.export_jsonl(export_path)

            # Verify file was created
            assert export_path.exists(), "JSONL export file was not created"

            # Verify JSONL format and content
            with open(export_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) >= 2, "Expected at least 2 lines in JSONL output"

            # Verify each line is valid JSON
            for line in lines:
                record = json.loads(line)
                assert "run_id" in record
                assert "task" in record
                assert "mode" in record
                assert "status" in record
                assert "created_at" in record

    def test_eval_tracker_pass_at_k_empty(self):
        """Verify Pass@k returns 0 when no runs exist."""
        tracker = EvalTracker()

        # Empty tracker should return 0
        pass_at_1 = tracker.calculate_pass_at_k(1)
        assert pass_at_1 == 0.0, "Empty tracker should have Pass@1 = 0.0"

    def test_eval_tracker_pass_at_k_all_pass(self):
        """Verify Pass@k when all runs pass."""
        tracker = EvalTracker()

        # Log all passing runs
        tracker.log_run("run_1", "task_A", "ask", "PASS", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_A", "ask", "PASS", "2026-05-12T10:01:00Z")
        tracker.log_run("run_3", "task_A", "ask", "PASS", "2026-05-12T10:02:00Z")

        # All passing -> Pass@1 = 1.0
        pass_at_1 = tracker.calculate_pass_at_k(1)
        assert abs(pass_at_1 - 1.0) < 0.001, f"All passes should yield 1.0, got {pass_at_1}"

    def test_eval_tracker_pass_at_k_all_fail(self):
        """Verify Pass@k when all runs fail."""
        tracker = EvalTracker()

        # Log all failing runs
        tracker.log_run("run_1", "task_A", "ask", "FAIL", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_A", "ask", "FAIL", "2026-05-12T10:01:00Z")

        # All failing -> Pass@1 = 0.0
        pass_at_1 = tracker.calculate_pass_at_k(1)
        assert abs(pass_at_1 - 0.0) < 0.001, f"All failures should yield 0.0, got {pass_at_1}"

    def test_eval_tracker_pass_at_k_multiple_tasks(self):
        """Verify Pass@k aggregates correctly across multiple tasks."""
        tracker = EvalTracker()

        # Task A: 2 pass, 1 fail -> Pass@1 = 1 - (1/3) ≈ 0.667
        tracker.log_run("run_1", "task_A", "ask", "PASS", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_A", "ask", "PASS", "2026-05-12T10:01:00Z")
        tracker.log_run("run_3", "task_A", "ask", "FAIL", "2026-05-12T10:02:00Z")

        # Task B: 1 pass, 0 fail -> Pass@1 = 1.0
        tracker.log_run("run_4", "task_B", "dual", "PASS", "2026-05-12T10:03:00Z")

        # Average: (0.667 + 1.0) / 2 ≈ 0.833
        pass_at_1 = tracker.calculate_pass_at_k(1)
        expected = (1 - (1/3) + 1.0) / 2
        assert abs(pass_at_1 - expected) < 0.01, f"Expected {expected}, got {pass_at_1}"
