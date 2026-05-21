"""Unit tests for orchestra.engine.eval_harness."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from orchestra.engine.eval_harness import task_hash, run_judge

def test_task_hash():
    h1 = task_hash("task1")
    h2 = task_hash("task1")
    h3 = task_hash("task2")
    assert h1 == h2
    assert h1 != h3

def test_run_judge_mock():
    # Mocking the judge call in the correct namespace
    with patch("orchestra.engine.runners.ask_runner.run_ask") as mock_ask:
        mock_run = MagicMock()
        judge_data = {
            "answer_quality": {"score": 0.8, "reasoning": "Good"},
            "factual_grounding": {"score": 0.9, "reasoning": "Good"},
            "source_attribution": {"score": 0.8, "reasoning": "Good"},
            "conciseness": {"score": 1.0, "reasoning": "Good"},
            "composite_score": 0.8,
            "explanation": "Good"
        }
        import json
        mock_run.agents = [MagicMock(stdout_log=json.dumps(judge_data))]
        mock_run.total_cost_usd = 0.01
        mock_ask.return_value = mock_run
        
        res = run_judge("task", "output", "run-1", "h1", judge_alias="cdx-fast")
        # composite_score() is a method? Let's check EvalResult
        assert res.task == "task"
