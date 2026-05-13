"""Tests for BatchRunner parallel scenario evaluation."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from orchestra.engine.batch_runner import BatchRunner, EvalScenario, EvalResult
from orchestra.models import RunStatus


class TestBatchRunner:
    """Test batch evaluation runner functionality."""

    def test_batch_runner_initialization(self):
        """Verify BatchRunner initializes with correct parameters."""
        runner = BatchRunner(max_workers=8, timeout_seconds=60, max_retries=3)

        assert runner.max_workers == 8
        assert runner.timeout_seconds == 60
        assert runner.max_retries == 3
        assert runner.executor is not None

    @patch("orchestra.engine.batch_runner.run_ask")
    def test_run_scenario_ask_mode(self, mock_run_ask):
        """Verify run_scenario() executes ask mode correctly."""
        # Mock orchestra run result
        mock_run = Mock()
        mock_run.status = RunStatus.COMPLETED
        mock_run.tokens = 1500
        mock_run.avg_confidence = 0.95
        mock_run_ask.return_value = mock_run

        runner = BatchRunner()
        result = runner.run_scenario(
            run_id="test_001",
            task="Implement feature X",
            mode="ask",
            alias="cdx-fast"
        )

        assert result.outcome == "PASS"
        assert result.tokens_used == 1500
        assert result.confidence == 0.95
        assert result.scenario_id == "test_001_cdx-fast"
        mock_run_ask.assert_called_once_with(alias="cdx-fast", prompt="Implement feature X")

    @patch("orchestra.engine.batch_runner.run_dual")
    def test_run_scenario_dual_mode(self, mock_run_dual):
        """Verify run_scenario() executes dual mode correctly."""
        mock_run = Mock()
        mock_run.status = RunStatus.COMPLETED
        mock_run.tokens = 2000
        mock_run.avg_confidence = 0.85
        mock_run_dual.return_value = mock_run

        runner = BatchRunner()
        result = runner.run_scenario(
            run_id="test_002",
            task="Debug issue Y",
            mode="dual",
            alias="gmn-pro"
        )

        assert result.outcome == "PASS"
        assert result.tokens_used == 2000
        assert result.confidence == 0.85
        mock_run_dual.assert_called_once_with(prompt="Debug issue Y", agents=["gmn-pro"])

    @patch("orchestra.engine.batch_runner.run_ask")
    def test_run_scenario_failure_status(self, mock_run_ask):
        """Verify outcome is FAIL when Orchestra run doesn't complete."""
        mock_run = Mock()
        mock_run.status = RunStatus.FAILED
        mock_run.tokens = 500
        mock_run.avg_confidence = 0.0
        mock_run_ask.return_value = mock_run

        runner = BatchRunner()
        result = runner.run_scenario(
            run_id="test_003",
            task="Bad task",
            mode="ask",
            alias="test-alias"
        )

        assert result.outcome == "FAIL"

    @patch("orchestra.engine.batch_runner.run_ask")
    def test_run_scenario_exception_handling(self, mock_run_ask):
        """Verify exception handling returns FAIL outcome."""
        mock_run_ask.side_effect = RuntimeError("Connection failed")

        runner = BatchRunner()
        result = runner.run_scenario(
            run_id="test_004",
            task="Failing task",
            mode="ask",
            alias="bad-alias"
        )

        assert result.outcome == "FAIL"
        assert result.tokens_used == 0
        assert result.confidence == 0.0

    def test_export_stats(self):
        """Verify export_stats() generates valid JSON."""
        results = [
            EvalResult("r1", "s1", "PASS", 1000, 0.9, "2026-05-12T10:00:00Z"),
            EvalResult("r2", "s2", "PASS", 1200, 0.85, "2026-05-12T10:01:00Z"),
            EvalResult("r3", "s3", "FAIL", 800, 0.5, "2026-05-12T10:02:00Z"),
        ]

        runner = BatchRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stats.json"
            runner.export_stats(results, output_path)

            # Verify file was created
            assert output_path.exists()

            # Verify JSON structure
            with open(output_path, "r", encoding="utf-8") as f:
                stats = json.load(f)

            assert stats["total_runs"] == 3
            assert stats["passed"] == 2
            assert stats["failed"] == 1
            assert abs(stats["pass_rate"] - 2/3) < 0.01
            assert stats["avg_tokens"] > 0
            assert stats["avg_confidence"] > 0
