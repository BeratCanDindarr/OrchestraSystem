"""LLM-as-Judge evaluation harness for offline quality measurement.

Phase 1: Core evaluation infrastructure.
- Runs three modes in parallel (ask, dual, critical)
- Judges outputs on four dimensions via Claude
- Stores results in append-only JSONL format
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orchestra.models import OrchestraRun

logger = logging.getLogger(__name__)


@dataclass
class JudgeCriteria:
    """Single dimension score with reasoning."""

    answer_quality: float
    answer_quality_reasoning: str
    factual_grounding: float
    factual_grounding_reasoning: str
    source_attribution: float
    source_attribution_reasoning: str
    conciseness: float
    conciseness_reasoning: str

    def composite_score(self) -> float:
        """Return mean of four dimensions."""
        return (
            self.answer_quality
            + self.factual_grounding
            + self.source_attribution
            + self.conciseness
        ) / 4.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "answer_quality": {
                "score": self.answer_quality,
                "reasoning": self.answer_quality_reasoning,
            },
            "factual_grounding": {
                "score": self.factual_grounding,
                "reasoning": self.factual_grounding_reasoning,
            },
            "source_attribution": {
                "score": self.source_attribution,
                "reasoning": self.source_attribution_reasoning,
            },
            "conciseness": {
                "score": self.conciseness,
                "reasoning": self.conciseness_reasoning,
            },
            "composite_score": self.composite_score(),
        }


@dataclass
class EvalResult:
    """Complete evaluation run result for one task."""

    task: str
    run_id: str
    task_hash: str
    created_at: str
    mode: str
    output: str
    judge_criteria: JudgeCriteria
    judge_cost_usd: float
    judge_timestamp: str
    composite_score: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "task": self.task,
            "run_id": self.run_id,
            "task_hash": self.task_hash,
            "created_at": self.created_at,
            "mode": self.mode,
            "output": self.output,
            "judge_criteria": self.judge_criteria.to_dict(),
            "judge_cost_usd": self.judge_cost_usd,
            "judge_timestamp": self.judge_timestamp,
            "composite_score": self.composite_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalResult:
        """Deserialize from dictionary."""
        judge_data = data["judge_criteria"]
        judge_criteria = JudgeCriteria(
            answer_quality=judge_data["answer_quality"]["score"],
            answer_quality_reasoning=judge_data["answer_quality"]["reasoning"],
            factual_grounding=judge_data["factual_grounding"]["score"],
            factual_grounding_reasoning=judge_data["factual_grounding"]["reasoning"],
            source_attribution=judge_data["source_attribution"]["score"],
            source_attribution_reasoning=judge_data["source_attribution"]["reasoning"],
            conciseness=judge_data["conciseness"]["score"],
            conciseness_reasoning=judge_data["conciseness"]["reasoning"],
        )
        return cls(
            task=data["task"],
            run_id=data["run_id"],
            task_hash=data["task_hash"],
            created_at=data["created_at"],
            mode=data["mode"],
            output=data["output"],
            judge_criteria=judge_criteria,
            judge_cost_usd=data["judge_cost_usd"],
            judge_timestamp=data["judge_timestamp"],
            composite_score=data["composite_score"],
        )


def task_hash(task: str) -> str:
    """Compute SHA256 hash of task (first 8 chars) for grouping results."""
    if not task:
        raise ValueError("task cannot be empty")
    return hashlib.sha256(task.encode()).hexdigest()[:8]


def save_eval(result: EvalResult) -> None:
    """Append evaluation result to JSONL file (immutable, append-only).

    Args:
        result: EvalResult to save

    Raises:
        ValueError: If file append fails
    """
    evals_dir = Path.home() / ".orchestra" / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    eval_file = evals_dir / f"{result.task_hash}.jsonl"

    try:
        with open(eval_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        logger.debug(f"Saved eval to {eval_file}")
    except IOError as e:
        raise ValueError(f"Failed to append eval to {eval_file}: {e}") from e


def load_evals(task_hash_: str) -> list[EvalResult]:
    """Load all evaluations for a task from JSONL file.

    Args:
        task_hash_: Task hash (8-char SHA256 prefix)

    Returns:
        List of EvalResult objects, empty list if file doesn't exist

    Raises:
        ValueError: If JSONL parsing fails
    """
    if not task_hash_ or len(task_hash_) < 8:
        raise ValueError(f"Invalid task_hash: {task_hash_}")

    evals_dir = Path.home() / ".orchestra" / "evals"
    eval_file = evals_dir / f"{task_hash_}.jsonl"

    if not eval_file.exists():
        return []

    results = []
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(EvalResult.from_dict(data))
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} of {eval_file}: {e}"
                    ) from e
        logger.debug(f"Loaded {len(results)} evals from {eval_file}")
    except IOError as e:
        raise ValueError(f"Failed to read {eval_file}: {e}") from e

    return results


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of AI-generated answers.

## Evaluation Context
User Task: {task}

## Output to Evaluate
{output_section}

## Evaluation Dimensions

### 1. Answer Quality (0.0–1.0)
Does the answer directly address the user's request? Is it actionable and complete?
- 0.0–0.3: Irrelevant, off-topic, or fundamentally flawed
- 0.3–0.6: Partially addresses the question; missing key information
- 0.6–0.8: Addresses most of the question; minor gaps
- 0.8–1.0: Fully addresses the question; clear and actionable

### 2. Factual Grounding (0.0–1.0)
Are statements factually accurate? Are claims verifiable?
- 0.0–0.3: Contains significant factual errors
- 0.3–0.6: Mix of correct and questionable claims; unclear grounding
- 0.6–0.8: Mostly correct; minor errors or unverified claims
- 0.8–1.0: Factually accurate throughout; claims are grounded or appropriately hedged

### 3. Source Attribution (0.0–1.0)
If external sources or examples are referenced, are they cited clearly?
- 0.0–0.3: No attribution; claims presented as universal truth without grounding
- 0.3–0.6: Some attribution, but vague or incomplete
- 0.6–0.8: Clear attribution for key claims; minor gaps
- 0.8–1.0: Comprehensive, specific attribution; sources are verifiable

### 4. Conciseness (0.0–1.0)
Is the response appropriately concise? Does it avoid unnecessary verbosity?
- 0.0–0.3: Extremely verbose; padded with irrelevant information
- 0.3–0.6: Somewhat verbose; could be condensed without losing meaning
- 0.6–0.8: Mostly concise; minor redundancy
- 0.8–1.0: Concise and direct; no wasted words

## Your Task

Score the output on the four dimensions above. Provide:
1. A score for each dimension (0.0–1.0)
2. Brief reasoning (1–2 sentences per dimension)
3. An overall composite score (mean of the four dimensions)

Return your evaluation as JSON.

## Output Format

Return ONLY valid JSON matching this schema:
```json
{{
  "answer_quality": {{"score": 0.75, "reasoning": "..."}},
  "factual_grounding": {{"score": 0.82, "reasoning": "..."}},
  "source_attribution": {{"score": 0.65, "reasoning": "..."}},
  "conciseness": {{"score": 0.88, "reasoning": "..."}},
  "composite_score": 0.775
}}
```
"""


def build_judge_prompt(task: str, mode: str, output: str) -> str:
    """Build judge prompt with task, mode, and output to evaluate.

    Args:
        task: User task/prompt
        mode: Execution mode (ask, dual, critical)
        output: Agent output to evaluate

    Returns:
        Formatted judge prompt string

    Raises:
        ValueError: If any parameter is empty or invalid
    """
    if not task:
        raise ValueError("task cannot be empty")
    if not mode:
        raise ValueError("mode cannot be empty")
    if not output:
        raise ValueError("output cannot be empty")

    output_section = f"""Mode: {mode}
Output:
{output}"""

    prompt = JUDGE_PROMPT_TEMPLATE.format(task=task, output_section=output_section)
    return prompt


def parse_judge_response(response_text: str) -> JudgeCriteria:
    """Parse Claude's JSON response into JudgeCriteria.

    Args:
        response_text: JSON string from Claude judge

    Returns:
        JudgeCriteria object

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    if not response_text:
        raise ValueError("response_text cannot be empty")

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in judge response: {e}") from e

    # Validate required fields
    required_fields = [
        "answer_quality",
        "factual_grounding",
        "source_attribution",
        "conciseness",
        "composite_score",
    ]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field in judge response: {field}")

    # Extract scores and reasoning
    try:
        criteria = JudgeCriteria(
            answer_quality=float(data["answer_quality"]["score"]),
            answer_quality_reasoning=data["answer_quality"]["reasoning"],
            factual_grounding=float(data["factual_grounding"]["score"]),
            factual_grounding_reasoning=data["factual_grounding"]["reasoning"],
            source_attribution=float(data["source_attribution"]["score"]),
            source_attribution_reasoning=data["source_attribution"]["reasoning"],
            conciseness=float(data["conciseness"]["score"]),
            conciseness_reasoning=data["conciseness"]["reasoning"],
        )

        # Validate scores are in valid range
        for dimension, score in [
            ("answer_quality", criteria.answer_quality),
            ("factual_grounding", criteria.factual_grounding),
            ("source_attribution", criteria.source_attribution),
            ("conciseness", criteria.conciseness),
        ]:
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"{dimension} score {score} is out of valid range [0.0, 1.0]"
                )

        return criteria
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to parse judge response: {e}") from e


@dataclass
class EvalBatchResult:
    """Aggregated results from batch evaluation of multiple outputs."""

    task_hash: str
    task_sample: str
    total_evals: int
    passed_evals: int
    failed_evals: int
    avg_composite_score: float
    avg_answer_quality: float
    avg_factual_grounding: float
    avg_source_attribution: float
    avg_conciseness: float
    min_composite_score: float
    max_composite_score: float
    percentiles: dict

    def to_dict(self) -> dict:
        return {
            "task_hash": self.task_hash,
            "task_sample": self.task_sample,
            "total_evals": self.total_evals,
            "passed_evals": self.passed_evals,
            "failed_evals": self.failed_evals,
            "avg_composite_score": self.avg_composite_score,
            "avg_answer_quality": self.avg_answer_quality,
            "avg_factual_grounding": self.avg_factual_grounding,
            "avg_source_attribution": self.avg_source_attribution,
            "avg_conciseness": self.avg_conciseness,
            "min_composite_score": self.min_composite_score,
            "max_composite_score": self.max_composite_score,
            "percentiles": self.percentiles,
        }


def batch_evaluate(task_hashes: list[str]) -> list[EvalBatchResult]:
    """Evaluate multiple tasks and return aggregated statistics.

    Args:
        task_hashes: List of 8-char task hashes to evaluate

    Returns:
        List of EvalBatchResult with aggregated metrics per task
    """
    from statistics import mean, median, quantiles

    results = []

    for task_hash_ in task_hashes:
        try:
            evals = load_evals(task_hash_)
            if not evals:
                continue

            scores = [e.composite_score for e in evals]
            quality = [e.judge_criteria.answer_quality for e in evals]
            grounding = [e.judge_criteria.factual_grounding for e in evals]
            attribution = [e.judge_criteria.source_attribution for e in evals]
            conciseness = [e.judge_criteria.conciseness for e in evals]

            # Calculate percentiles
            n = len(scores)
            pct_data = quantiles(scores, n=4) if n >= 4 else []
            percentiles = {
                "p25": pct_data[0] if len(pct_data) > 0 else min(scores),
                "p50": pct_data[1] if len(pct_data) > 1 else median(scores),
                "p75": pct_data[2] if len(pct_data) > 2 else max(scores),
            }

            batch_result = EvalBatchResult(
                task_hash=task_hash_,
                task_sample=evals[0].task[:100] if evals else "",
                total_evals=len(evals),
                passed_evals=sum(1 for e in evals if e.composite_score >= 0.7),
                failed_evals=sum(1 for e in evals if e.composite_score < 0.7),
                avg_composite_score=mean(scores),
                avg_answer_quality=mean(quality),
                avg_factual_grounding=mean(grounding),
                avg_source_attribution=mean(attribution),
                avg_conciseness=mean(conciseness),
                min_composite_score=min(scores),
                max_composite_score=max(scores),
                percentiles=percentiles,
            )
            results.append(batch_result)
        except Exception as e:
            logger.warning(f"Failed to evaluate task {task_hash_}: {e}")
            continue

    return results


def compute_eval_stats(task_hashes: list[str]) -> dict:
    """Compute overall statistics across multiple evaluated tasks.

    Args:
        task_hashes: List of 8-char task hashes

    Returns:
        Dictionary with aggregate metrics and trends
    """
    from statistics import mean

    batch_results = batch_evaluate(task_hashes)

    if not batch_results:
        return {
            "total_tasks_evaluated": 0,
            "total_evals": 0,
            "overall_pass_rate": 0.0,
            "overall_avg_score": 0.0,
        }

    all_scores = []
    all_quality = []
    all_grounding = []
    all_attribution = []
    all_conciseness = []
    total_passed = 0
    total_failed = 0

    for batch in batch_results:
        all_scores.extend([batch.avg_composite_score])
        total_passed += batch.passed_evals
        total_failed += batch.failed_evals

    total_evals = total_passed + total_failed

    return {
        "total_tasks_evaluated": len(batch_results),
        "total_evals": total_evals,
        "overall_pass_rate": (
            total_passed / total_evals if total_evals > 0 else 0.0
        ),
        "overall_avg_score": mean(all_scores) if all_scores else 0.0,
        "best_task": max(batch_results, key=lambda x: x.avg_composite_score).task_hash
        if batch_results
        else None,
        "worst_task": min(batch_results, key=lambda x: x.avg_composite_score).task_hash
        if batch_results
        else None,
        "task_results": [b.to_dict() for b in batch_results],
    }
