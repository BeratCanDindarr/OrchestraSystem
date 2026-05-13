"""End-to-end test: Full eval harness pipeline."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from orchestra.engine.runner import OrchestraRun, RunStatus
from orchestra.engine.batch_runner import BatchRunner, EvalScenario, EvalResult
from orchestra.storage.eval_tracker import EvalTracker
from orchestra.storage.db import get_db


class TestEvalE2E:
    """End-to-end evaluation harness pipeline test."""

    def setup_method(self):
        """Clear eval tables before each test."""
        connection = get_db()
        with connection:
            connection.execute("DELETE FROM eval_runs")
            connection.execute("DELETE FROM eval_scenarios")
            connection.execute("DELETE FROM eval_results")
        connection.close()

    def test_e2e_run_to_stats_export(self):
        """Verify complete pipeline: run → finalize → log → batch → stats."""
        # Step 1: Simulate run completion and outcome logging (what _finalize() does)
        runs = [
            OrchestraRun(
                run_id="e2e_001",
                task="Implement feature A",
                mode="ask",
                status=RunStatus.COMPLETED,
                created_at="2026-05-12T10:00:00Z",
                updated_at="2026-05-12T10:01:00Z"
            ),
            OrchestraRun(
                run_id="e2e_002",
                task="Implement feature B",
                mode="dual",
                status=RunStatus.COMPLETED,
                created_at="2026-05-12T10:02:00Z",
                updated_at="2026-05-12T10:03:00Z"
            ),
            OrchestraRun(
                run_id="e2e_003",
                task="Debug feature C",
                mode="ask",
                status=RunStatus.FAILED,
                created_at="2026-05-12T10:04:00Z",
                updated_at="2026-05-12T10:05:00Z"
            ),
        ]

        # Step 2: Log outcomes after finalization
        tracker = EvalTracker()
        for run in runs:
            eval_status = "PASS" if run.status == RunStatus.COMPLETED else "FAIL"
            tracker.log_run(
                run_id=run.run_id,
                task=run.task,
                mode=run.mode,
                status=eval_status,
                created_at=run.updated_at or run.created_at
            )

        # Step 3: Verify outcomes logged
        connection = get_db()
        cursor = connection.execute("SELECT COUNT(*) as cnt FROM eval_runs")
        count = cursor.fetchone()["cnt"]
        connection.close()

        assert count == 3, f"Expected 3 logged runs, got {count}"

        # Step 4: Calculate Pass@K metric
        pass_at_1 = tracker.calculate_pass_at_k(1)

        # Expected:
        # Each of 3 tasks: 1 run each (n=1)
        # task "Implement feature A": 1 PASS (c=1) → Pass@1 = 1.0
        # task "Implement feature B": 1 PASS (c=1) → Pass@1 = 1.0
        # task "Debug feature C": 1 FAIL (c=0) → Pass@1 = 0.0
        # Average: (1.0 + 1.0 + 0.0) / 3 = 0.667
        expected = 2.0 / 3.0
        assert abs(pass_at_1 - expected) < 0.01, f"Expected {expected}, got {pass_at_1}"

        # Step 5: Export JSONL for analysis
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "eval_runs.jsonl"
            tracker.export_jsonl(export_path)

            # Verify JSONL file
            assert export_path.exists()
            with open(export_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == 3
            records = [json.loads(line) for line in lines]
            outcomes = [r["status"] for r in records]
            assert outcomes.count("PASS") == 2
            assert outcomes.count("FAIL") == 1

        # Step 6: Simulate batch runner stats export
        runner = BatchRunner()
        results = [
            EvalResult("r1", "s1", "PASS", 1000, 0.95, "2026-05-12T10:00:00Z"),
            EvalResult("r2", "s2", "PASS", 1100, 0.92, "2026-05-12T10:01:00Z"),
            EvalResult("r3", "s3", "FAIL", 900, 0.60, "2026-05-12T10:02:00Z"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            runner.export_stats(results, stats_path)

            # Verify stats
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)

            assert stats["total_runs"] == 3
            assert stats["passed"] == 2
            assert stats["failed"] == 1
            assert abs(stats["pass_rate"] - 2/3) < 0.01
            assert stats["avg_tokens"] == 1000
            assert stats["avg_confidence"] > 0.8

    def test_e2e_multiple_tasks_metrics(self):
        """Verify Pass@K aggregation across multiple tasks."""
        tracker = EvalTracker()

        # Log mixed results across multiple tasks
        tracker.log_run("run_1", "task_1", "ask", "PASS", "2026-05-12T10:00:00Z")
        tracker.log_run("run_2", "task_1", "ask", "PASS", "2026-05-12T10:01:00Z")
        tracker.log_run("run_3", "task_1", "ask", "FAIL", "2026-05-12T10:02:00Z")

        tracker.log_run("run_4", "task_2", "dual", "PASS", "2026-05-12T10:03:00Z")
        tracker.log_run("run_5", "task_2", "dual", "FAIL", "2026-05-12T10:04:00Z")

        tracker.log_run("run_6", "task_3", "critical", "PASS", "2026-05-12T10:05:00Z")

        # Calculate Pass@1
        pass_at_1 = tracker.calculate_pass_at_k(1)

        # Expected:
        # task_1 (n=3, c=2): Pass@1 = 1 - C(1,1)/C(3,1) = 1 - 1/3 = 0.667
        # task_2 (n=2, c=1): Pass@1 = 1 - C(1,1)/C(2,1) = 1 - 1/2 = 0.5
        # task_3 (n=1, c=1): Pass@1 = 1.0
        # Average = (0.667 + 0.5 + 1.0) / 3 = 0.722
        expected = (2/3 + 0.5 + 1.0) / 3
        assert abs(pass_at_1 - expected) < 0.01
