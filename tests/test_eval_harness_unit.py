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


def test_is_judging_default_false():
    from orchestra.engine.eval_harness import is_judging
    assert is_judging() is False


def test_build_judge_prompt_requests_json_rubric():
    from orchestra.engine.eval_harness import build_judge_prompt
    p = build_judge_prompt("my task", "ask", "my answer")
    assert "answer_quality" in p and "composite_score" in p
    assert "my task" in p and "my answer" in p


def test_extract_json_tolerates_fences_and_prose():
    from orchestra.engine.eval_harness import _extract_json
    assert _extract_json('```json\n{"composite_score": 0.5}\n```')["composite_score"] == 0.5
    assert _extract_json('here it is: {"composite_score": 0.7} done')["composite_score"] == 0.7


def test_run_judge_sets_and_clears_reentrancy_guard():
    from orchestra.engine import eval_harness
    seen = {}

    def fake_ask(alias, prompt, **kwargs):
        seen["judging"] = eval_harness.is_judging()   # guard must be set during the nested run
        m = MagicMock()
        m.agents = [MagicMock(stdout_log='{"composite_score": 0.9, "explanation": "ok"}')]
        return m

    with patch("orchestra.engine.runners.ask_runner.run_ask", side_effect=fake_ask):
        res = eval_harness.run_judge("t", "o", "r1", "h1", mode="dual")
    assert seen["judging"] is True
    assert eval_harness.is_judging() is False          # cleared after the run
    assert res.mode == "dual" and res.composite_score == 0.9


def test_eval_harness_disabled_by_default():
    from orchestra import config
    assert config.eval_harness_enabled() is False
    assert config.eval_judge_alias() == "cld-fast"
