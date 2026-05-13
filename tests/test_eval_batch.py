"""Tests for batch evaluation and statistics aggregation."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestra.engine.eval_harness import (
    EvalResult,
    JudgeCriteria,
    EvalBatchResult,
    batch_evaluate,
    compute_eval_stats,
    task_hash,
)


@pytest.fixture
def temp_evals_dir(monkeypatch):
    """Create temporary evals directory and patch Path.home()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evals_dir = Path(tmpdir) / ".orchestra" / "evals"
        evals_dir.mkdir(parents=True, exist_ok=True)

        # Patch Path.home() to return tmpdir
        monkeypatch.setattr(Path, "home", lambda: Path(tmpdir))
        yield evals_dir


def create_sample_eval(
    task: str,
    run_id: str,
    score: float,
    quality: float = None,
    grounding: float = None,
    attribution: float = None,
    conciseness: float = None,
) -> EvalResult:
    """Create a sample EvalResult for testing."""
    if quality is None:
        quality = score
    if grounding is None:
        grounding = score
    if attribution is None:
        attribution = score
    if conciseness is None:
        conciseness = score

    criteria = JudgeCriteria(
        answer_quality=quality,
        answer_quality_reasoning="Sample reasoning",
        factual_grounding=grounding,
        factual_grounding_reasoning="Sample reasoning",
        source_attribution=attribution,
        source_attribution_reasoning="Sample reasoning",
        conciseness=conciseness,
        conciseness_reasoning="Sample reasoning",
    )

    return EvalResult(
        task=task,
        run_id=run_id,
        task_hash=task_hash(task),
        created_at=datetime.now(timezone.utc).isoformat(),
        mode="ask",
        output="Sample output",
        judge_criteria=criteria,
        judge_cost_usd=0.01,
        judge_timestamp=datetime.now(timezone.utc).isoformat(),
        composite_score=score,
    )


def test_batch_evaluate_single_task(temp_evals_dir, monkeypatch):
    """Test batch_evaluate with a single task containing multiple evals."""
    # Create test data
    task = "What is 2+2?"
    t_hash = task_hash(task)

    evals = [
        create_sample_eval(task, f"run_{i}", 0.8 + (i * 0.02))
        for i in range(5)
    ]

    # Write evals to file
    eval_file = temp_evals_dir / f"{t_hash}.jsonl"
    with open(eval_file, "w") as f:
        for e in evals:
            f.write(json.dumps(e.to_dict()) + "\n")

    # Run batch_evaluate
    results = batch_evaluate([t_hash])

    assert len(results) == 1
    batch = results[0]
    assert batch.task_hash == t_hash
    assert batch.total_evals == 5
    assert batch.passed_evals == 5  # All have score >= 0.7
    assert batch.failed_evals == 0
    assert 0.8 <= batch.avg_composite_score <= 0.88
    assert batch.min_composite_score == 0.8
    assert batch.max_composite_score == 0.88


def test_batch_evaluate_multiple_tasks(temp_evals_dir, monkeypatch):
    """Test batch_evaluate with multiple tasks."""
    task1 = "Task 1"
    task2 = "Task 2"
    hash1 = task_hash(task1)
    hash2 = task_hash(task2)

    # Task 1: 3 evals
    evals1 = [
        create_sample_eval(task1, f"run1_{i}", 0.9) for i in range(3)
    ]

    # Task 2: 2 evals with lower scores
    evals2 = [
        create_sample_eval(task2, f"run2_{i}", 0.5) for i in range(2)
    ]

    # Write to files
    eval_file1 = temp_evals_dir / f"{hash1}.jsonl"
    with open(eval_file1, "w") as f:
        for e in evals1:
            f.write(json.dumps(e.to_dict()) + "\n")

    eval_file2 = temp_evals_dir / f"{hash2}.jsonl"
    with open(eval_file2, "w") as f:
        for e in evals2:
            f.write(json.dumps(e.to_dict()) + "\n")

    # Run batch_evaluate
    results = batch_evaluate([hash1, hash2])

    assert len(results) == 2

    # Task 1 results
    batch1 = next(r for r in results if r.task_hash == hash1)
    assert batch1.total_evals == 3
    assert batch1.passed_evals == 3
    assert batch1.avg_composite_score == 0.9

    # Task 2 results
    batch2 = next(r for r in results if r.task_hash == hash2)
    assert batch2.total_evals == 2
    assert batch2.passed_evals == 0  # All have score < 0.7
    assert batch2.failed_evals == 2
    assert batch2.avg_composite_score == 0.5


def test_batch_evaluate_pass_fail_threshold(temp_evals_dir, monkeypatch):
    """Test that pass/fail threshold of 0.7 is correctly applied."""
    task = "Threshold test"
    t_hash = task_hash(task)

    scores = [0.65, 0.69, 0.70, 0.75, 0.80]
    evals = [
        create_sample_eval(task, f"run_{i}", score)
        for i, score in enumerate(scores)
    ]

    eval_file = temp_evals_dir / f"{t_hash}.jsonl"
    with open(eval_file, "w") as f:
        for e in evals:
            f.write(json.dumps(e.to_dict()) + "\n")

    results = batch_evaluate([t_hash])
    batch = results[0]

    assert batch.passed_evals == 3  # 0.70, 0.75, 0.80
    assert batch.failed_evals == 2  # 0.65, 0.69


def test_batch_evaluate_empty_task(temp_evals_dir, monkeypatch):
    """Test batch_evaluate with nonexistent task hash."""
    results = batch_evaluate(["nonexistent"])
    assert results == []


def test_batch_evaluate_percentiles(temp_evals_dir, monkeypatch):
    """Test percentile calculation."""
    task = "Percentile test"
    t_hash = task_hash(task)

    # Create 10 evals with scores 0.1, 0.2, ..., 1.0
    evals = [
        create_sample_eval(task, f"run_{i}", (i + 1) / 10.0)
        for i in range(10)
    ]

    eval_file = temp_evals_dir / f"{t_hash}.jsonl"
    with open(eval_file, "w") as f:
        for e in evals:
            f.write(json.dumps(e.to_dict()) + "\n")

    results = batch_evaluate([t_hash])
    batch = results[0]

    # Percentiles should roughly be:
    # p25 ≈ 0.25, p50 ≈ 0.5, p75 ≈ 0.75
    assert 0.2 <= batch.percentiles["p25"] <= 0.4
    assert 0.4 <= batch.percentiles["p50"] <= 0.6
    assert 0.6 <= batch.percentiles["p75"] <= 0.85


def test_compute_eval_stats_single_task(temp_evals_dir, monkeypatch):
    """Test compute_eval_stats with a single task."""
    task = "Stats test"
    t_hash = task_hash(task)

    evals = [
        create_sample_eval(task, f"run_{i}", 0.85) for i in range(4)
    ]

    eval_file = temp_evals_dir / f"{t_hash}.jsonl"
    with open(eval_file, "w") as f:
        for e in evals:
            f.write(json.dumps(e.to_dict()) + "\n")

    stats = compute_eval_stats([t_hash])

    assert stats["total_tasks_evaluated"] == 1
    assert stats["total_evals"] == 4
    assert stats["overall_pass_rate"] == 1.0
    assert stats["overall_avg_score"] == 0.85
    assert stats["best_task"] == t_hash
    assert stats["worst_task"] == t_hash


def test_compute_eval_stats_multiple_tasks(temp_evals_dir, monkeypatch):
    """Test compute_eval_stats with multiple tasks."""
    task1 = "Task 1"
    task2 = "Task 2"
    hash1 = task_hash(task1)
    hash2 = task_hash(task2)

    # Task 1: high scores (0.9)
    evals1 = [create_sample_eval(task1, f"run1_{i}", 0.9) for i in range(3)]

    # Task 2: low scores (0.6)
    evals2 = [create_sample_eval(task2, f"run2_{i}", 0.6) for i in range(3)]

    eval_file1 = temp_evals_dir / f"{hash1}.jsonl"
    with open(eval_file1, "w") as f:
        for e in evals1:
            f.write(json.dumps(e.to_dict()) + "\n")

    eval_file2 = temp_evals_dir / f"{hash2}.jsonl"
    with open(eval_file2, "w") as f:
        for e in evals2:
            f.write(json.dumps(e.to_dict()) + "\n")

    stats = compute_eval_stats([hash1, hash2])

    assert stats["total_tasks_evaluated"] == 2
    assert stats["total_evals"] == 6
    assert stats["best_task"] == hash1
    assert stats["worst_task"] == hash2
    # Overall pass rate: 3 passed (hash1) + 0 passed (hash2) = 3/6 = 0.5
    assert stats["overall_pass_rate"] == 0.5


def test_compute_eval_stats_no_tasks(temp_evals_dir, monkeypatch):
    """Test compute_eval_stats with no tasks."""
    stats = compute_eval_stats([])

    # When there are no tasks, dict still contains these keys but with None/0 values
    assert stats["total_tasks_evaluated"] == 0
    assert stats["total_evals"] == 0
    assert stats["overall_pass_rate"] == 0.0
    assert stats["overall_avg_score"] == 0.0


def test_eval_result_dimension_scores(temp_evals_dir, monkeypatch):
    """Test that per-dimension scores are correctly aggregated."""
    task = "Dimension test"
    t_hash = task_hash(task)

    # Create evals with varying scores per dimension
    criteria1 = JudgeCriteria(
        answer_quality=0.9,
        answer_quality_reasoning="Good",
        factual_grounding=0.8,
        factual_grounding_reasoning="OK",
        source_attribution=0.7,
        source_attribution_reasoning="Fair",
        conciseness=0.6,
        conciseness_reasoning="Verbose",
    )

    criteria2 = JudgeCriteria(
        answer_quality=0.8,
        answer_quality_reasoning="Good",
        factual_grounding=0.9,
        factual_grounding_reasoning="Excellent",
        source_attribution=0.8,
        source_attribution_reasoning="Good",
        conciseness=0.7,
        conciseness_reasoning="Fair",
    )

    eval1 = EvalResult(
        task=task,
        run_id="run1",
        task_hash=t_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode="ask",
        output="Output 1",
        judge_criteria=criteria1,
        judge_cost_usd=0.01,
        judge_timestamp=datetime.now(timezone.utc).isoformat(),
        composite_score=criteria1.composite_score(),
    )

    eval2 = EvalResult(
        task=task,
        run_id="run2",
        task_hash=t_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode="ask",
        output="Output 2",
        judge_criteria=criteria2,
        judge_cost_usd=0.01,
        judge_timestamp=datetime.now(timezone.utc).isoformat(),
        composite_score=criteria2.composite_score(),
    )

    eval_file = temp_evals_dir / f"{t_hash}.jsonl"
    with open(eval_file, "w") as f:
        f.write(json.dumps(eval1.to_dict()) + "\n")
        f.write(json.dumps(eval2.to_dict()) + "\n")

    results = batch_evaluate([t_hash])
    batch = results[0]

    assert batch.avg_answer_quality == pytest.approx(0.85)  # (0.9 + 0.8) / 2
    assert batch.avg_factual_grounding == pytest.approx(0.85)  # (0.8 + 0.9) / 2
    assert batch.avg_source_attribution == pytest.approx(0.75)  # (0.7 + 0.8) / 2
    assert batch.avg_conciseness == pytest.approx(0.65)  # (0.6 + 0.7) / 2
