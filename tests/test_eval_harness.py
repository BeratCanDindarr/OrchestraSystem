"""Tests for eval harness judge execution."""
import json
from unittest.mock import patch, MagicMock

import pytest

from orchestra.engine.eval_harness import (
    run_judge,
    EvalResult,
    JudgeCriteria,
    parse_judge_response,
    build_judge_prompt,
    _extract_json,
)


def test_extract_json_plain():
    """Extract JSON when no markdown is present."""
    text = '{"answer_quality": {"score": 0.8}}'
    assert _extract_json(text) == '{"answer_quality": {"score": 0.8}}'


def test_extract_json_from_markdown_block():
    """Extract JSON from ```json markdown block."""
    text = 'Some text\n```json\n{"answer_quality": {"score": 0.8}}\n```\nMore text'
    assert _extract_json(text) == '{"answer_quality": {"score": 0.8}}'


def test_extract_json_from_generic_markdown_block():
    """Extract JSON from generic ``` markdown block."""
    text = 'Prefix\n```\n{"answer_quality": {"score": 0.8}}\n```\nSuffix'
    assert _extract_json(text) == '{"answer_quality": {"score": 0.8}}'


def test_extract_json_empty():
    """Handle empty input."""
    assert _extract_json("") == ""


def test_build_judge_prompt_basic():
    """Build judge prompt with required fields."""
    prompt = build_judge_prompt("What is 2+2?", "ask", "The answer is 4.")
    assert "What is 2+2?" in prompt
    assert "ask" in prompt
    assert "The answer is 4." in prompt


def test_build_judge_prompt_with_gold_standard():
    """Build judge prompt including gold standard section."""
    prompt = build_judge_prompt(
        "What is 2+2?", "ask", "The answer is 4.", gold_standard="Four."
    )
    assert "Gold Standard" in prompt
    assert "Four." in prompt


def test_build_judge_prompt_validation():
    """Validate empty inputs are rejected."""
    with pytest.raises(ValueError, match="task cannot be empty"):
        build_judge_prompt("", "ask", "output")
    with pytest.raises(ValueError, match="mode cannot be empty"):
        build_judge_prompt("task", "", "output")
    with pytest.raises(ValueError, match="output cannot be empty"):
        build_judge_prompt("task", "ask", "")


def test_parse_judge_response_valid():
    """Parse a well-formed JSON judge response."""
    response = json.dumps(
        {
            "answer_quality": {"score": 0.85, "reasoning": "Good"},
            "factual_grounding": {"score": 0.90, "reasoning": "Accurate"},
            "source_attribution": {"score": 0.75, "reasoning": "Some sources"},
            "conciseness": {"score": 0.80, "reasoning": "Concise"},
            "composite_score": 0.825,
        }
    )
    criteria = parse_judge_response(response)
    assert criteria.answer_quality == 0.85
    assert criteria.factual_grounding == 0.90
    assert criteria.source_attribution == 0.75
    assert criteria.conciseness == 0.80


def test_parse_judge_response_markdown_wrapped():
    """Parse JSON wrapped in markdown code blocks."""
    response = (
        "Here is the evaluation:\n"
        "```json\n"
        + json.dumps(
            {
                "answer_quality": {"score": 0.85, "reasoning": "Good"},
                "factual_grounding": {"score": 0.90, "reasoning": "Accurate"},
                "source_attribution": {"score": 0.75, "reasoning": "Some sources"},
                "conciseness": {"score": 0.80, "reasoning": "Concise"},
                "composite_score": 0.825,
            }
        )
        + "\n```"
    )
    criteria = parse_judge_response(response)
    assert criteria.answer_quality == 0.85


def test_parse_judge_response_empty():
    """Empty response raises ValueError."""
    with pytest.raises(ValueError, match="response_text cannot be empty"):
        parse_judge_response("")


def test_parse_judge_response_invalid_json():
    """Invalid JSON raises ValueError."""
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_judge_response("not json")


def test_parse_judge_response_missing_field():
    """Missing required field raises ValueError."""
    response = json.dumps({"answer_quality": {"score": 0.8, "reasoning": "OK"}})
    with pytest.raises(ValueError, match="Missing required field"):
        parse_judge_response(response)


def test_parse_judge_response_out_of_range():
    """Score out of valid range raises ValueError."""
    response = json.dumps(
        {
            "answer_quality": {"score": 1.5, "reasoning": "Too high"},
            "factual_grounding": {"score": 0.9, "reasoning": "OK"},
            "source_attribution": {"score": 0.9, "reasoning": "OK"},
            "conciseness": {"score": 0.9, "reasoning": "OK"},
            "composite_score": 1.05,
        }
    )
    with pytest.raises(ValueError, match="out of valid range"):
        parse_judge_response(response)


def test_eval_result_passed():
    """Test passed property on EvalResult."""
    criteria = JudgeCriteria(
        answer_quality=0.8,
        answer_quality_reasoning="OK",
        factual_grounding=0.8,
        factual_grounding_reasoning="OK",
        source_attribution=0.8,
        source_attribution_reasoning="OK",
        conciseness=0.8,
        conciseness_reasoning="OK",
    )
    result = EvalResult(
        task="task",
        run_id="r1",
        task_hash="hash1234",
        created_at="2026-01-01T00:00:00Z",
        mode="ask",
        output="output",
        judge_criteria=criteria,
        judge_cost_usd=0.01,
        judge_timestamp="2026-01-01T00:00:00Z",
        composite_score=criteria.composite_score(),
    )
    assert result.passed is True
    assert result.composite_score == 0.8


def test_eval_result_failed():
    """Test passed property when score is below threshold."""
    criteria = JudgeCriteria(
        answer_quality=0.5,
        answer_quality_reasoning="OK",
        factual_grounding=0.5,
        factual_grounding_reasoning="OK",
        source_attribution=0.5,
        source_attribution_reasoning="OK",
        conciseness=0.5,
        conciseness_reasoning="OK",
    )
    result = EvalResult(
        task="task",
        run_id="r1",
        task_hash="hash1234",
        created_at="2026-01-01T00:00:00Z",
        mode="ask",
        output="output",
        judge_criteria=criteria,
        judge_cost_usd=0.01,
        judge_timestamp="2026-01-01T00:00:00Z",
        composite_score=criteria.composite_score(),
    )
    assert result.passed is False
    assert result.composite_score == 0.5


def test_run_judge_success():
    """Mock run_ask and verify run_judge returns correct EvalResult."""
    judge_response = {
        "answer_quality": {"score": 0.85, "reasoning": "Good answer"},
        "factual_grounding": {"score": 0.90, "reasoning": "Well grounded"},
        "source_attribution": {"score": 0.75, "reasoning": "Some sources"},
        "conciseness": {"score": 0.80, "reasoning": "Fairly concise"},
        "composite_score": 0.825,
    }

    mock_agent = MagicMock()
    mock_agent.stdout_log = json.dumps(judge_response)
    mock_run = MagicMock()
    mock_run.agents = [mock_agent]
    mock_run.total_cost_usd = 0.0015

    with patch("orchestra.engine.runner.run_ask", return_value=mock_run):
        result = run_judge(
            task_text="What is 2+2?",
            agent_output="The answer is 4.",
            run_id="run_123",
            task_hash="abc12345",
            mode="ask",
        )

    assert isinstance(result, EvalResult)
    assert result.task == "What is 2+2?"
    assert result.run_id == "run_123"
    assert result.task_hash == "abc12345"
    assert result.mode == "ask"
    assert result.output == "The answer is 4."
    assert result.composite_score == 0.825
    assert result.judge_cost_usd == 0.0015
    assert result.passed is True
    assert result.judge_criteria.answer_quality == 0.85
    assert result.judge_criteria.factual_grounding == 0.90
    assert result.judge_criteria.source_attribution == 0.75
    assert result.judge_criteria.conciseness == 0.80


def test_run_judge_empty_agents():
    """Empty agent list should lead to parse error (empty response)."""
    mock_run = MagicMock()
    mock_run.agents = []
    mock_run.total_cost_usd = 0.0

    with patch("orchestra.engine.runner.run_ask", return_value=mock_run):
        with pytest.raises(ValueError, match="response_text cannot be empty"):
            run_judge(
                task_text="What is 2+2?",
                agent_output="The answer is 4.",
                run_id="run_123",
                task_hash="abc12345",
                mode="ask",
            )


def test_run_judge_with_gold_standard():
    """run_judge passes gold_standard through to prompt builder."""
    judge_response = {
        "answer_quality": {"score": 0.9, "reasoning": "Great"},
        "factual_grounding": {"score": 0.9, "reasoning": "Great"},
        "source_attribution": {"score": 0.9, "reasoning": "Great"},
        "conciseness": {"score": 0.9, "reasoning": "Great"},
        "composite_score": 0.9,
    }

    mock_agent = MagicMock()
    mock_agent.stdout_log = json.dumps(judge_response)
    mock_run = MagicMock()
    mock_run.agents = [mock_agent]
    mock_run.total_cost_usd = 0.001

    with patch("orchestra.engine.runner.run_ask", return_value=mock_run) as mock_ask:
        result = run_judge(
            task_text="What is 2+2?",
            agent_output="The answer is 4.",
            run_id="run_123",
            task_hash="abc12345",
            mode="ask",
            gold_standard="Four.",
        )

    assert result.passed is True
    # Verify prompt contains gold standard
    call_args = mock_ask.call_args
    prompt = call_args[0][1]  # second positional arg is the prompt
    assert "Gold Standard" in prompt
    assert "Four." in prompt
