from __future__ import annotations

import contextvars
import json
import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from orchestra import config
from orchestra.models import OrchestraRun, RunStatus

from orchestra.engine import structured  # noqa: E402

logger = logging.getLogger(__name__)

# Set while an eval judge run is executing. run_judge() starts its own run via run_ask,
# which finalizes and could otherwise trigger another judge -> unbounded recursion.
# Finalize checks is_judging() and skips auto-judging when a judge is already in flight.
_in_judge: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_eval_judge", default=False)


def is_judging() -> bool:
    """True while an eval judge run is executing (guards against recursive re-judging)."""
    return _in_judge.get()

def task_hash(task: str) -> str:
    """Return stable hash for a task string."""
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]

@dataclass
class EvalResult:
    task: str
    run_id: str
    task_hash: str
    created_at: str
    mode: str
    output: str
    answer_quality: float
    factual_grounding: float
    source_attribution: float
    conciseness: float
    composite_score: float
    explanation: str

def build_judge_prompt(task: str, mode: str, output: str, gold_standard: str = "") -> str:
    """Rubric prompt asking a judge model to score an answer and reply with strict JSON."""
    gold = f"\n\nReference answer (ground truth):\n{gold_standard}" if gold_standard else ""
    return (
        "You are a strict evaluation judge. Score the ANSWER to the TASK on four axes, "
        "each a float from 0.0 (worst) to 1.0 (best):\n"
        "- answer_quality: correctness and usefulness of the answer.\n"
        "- factual_grounding: claims are accurate and not fabricated.\n"
        "- source_attribution: sources/citations are given where they matter (1.0 if none needed).\n"
        "- conciseness: as short as the task allows without dropping needed content.\n"
        "Then give composite_score (0.0-1.0, your overall judgement) and a one-sentence explanation.\n"
        "Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:\n"
        '{"answer_quality": {"score": 0.0}, "factual_grounding": {"score": 0.0}, '
        '"source_attribution": {"score": 0.0}, "conciseness": {"score": 0.0}, '
        '"composite_score": 0.0, "explanation": ""}\n\n'
        f"TASK:\n{task}\n\nANSWER:\n{output}{gold}"
    )


def _extract_json(text: str) -> dict:
    """Parse a JSON object from judge output, tolerating markdown fences or surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    payload = text[start : end + 1] if (start != -1 and end > start) else text
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("judge output is not a JSON object")
    return obj


def run_judge(
    task_text: str,
    agent_output: str,
    run_id: str,
    t_hash: str,
    gold_standard: str = "",
    judge_alias: str = "cld-fast",
    mode: str = "ask",
) -> EvalResult:
    from orchestra.engine.runners.ask_runner import run_ask

    judge_prompt = build_judge_prompt(task_text, mode, agent_output, gold_standard)
    # Guard the nested run so its finalize does not recursively spawn another judge.
    token = _in_judge.set(True)
    try:
        # A step of the run it judges: its prompt as written, no stored answer, no gate of its
        # own, not indexed as a task, hidden from `ps`, cancelled with its parent.
        data, _ = structured.ask_for_object(
            run_ask, judge_alias, judge_prompt,
            check=lambda scores: None if isinstance(scores.get("composite_score"), (int, float))
            else "composite_score must be a number",
            emit_console=False, show_live=False, internal=True, parent_run_id=run_id,
        )
    finally:
        _in_judge.reset(token)
    if data is None:
        raise ValueError("Failed to parse judge response: no usable scores after two answers")
    return EvalResult(
        task=task_text,
        run_id=run_id,
        task_hash=t_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        mode=mode,
        output=agent_output,
        answer_quality=float(data.get("answer_quality", {}).get("score", 0.0)),
        factual_grounding=float(data.get("factual_grounding", {}).get("score", 0.0)),
        source_attribution=float(data.get("source_attribution", {}).get("score", 0.0)),
        conciseness=float(data.get("conciseness", {}).get("score", 0.0)),
        composite_score=float(data.get("composite_score", 0.0)),
        explanation=data.get("explanation", ""),
    )

def benchmark_context_strategies(task: str, alias: str | None = None):
    """Benchmark full vs surgical context."""
    from orchestra.engine.runners.ask_runner import run_ask

    alias = alias or config.default_light_alias()
    results = {}
    for strategy in ["full", "surgical"]:
        config._cfg.setdefault("orchestra", {})["context_strategy"] = strategy
        run = run_ask(alias, task, show_live=False)
        
        # Extract token info from the first agent run
        tokens = run.agents[0].estimated_completion_tokens if run.agents else 0
        results[strategy] = tokens
        print(f"Strategy: {strategy:10} | Tokens: {tokens}")
        
    delta = results["full"] - results["surgical"]
    if results["full"] > 0:
        percent = (delta / results["full"]) * 100
        print(f"Reduction: {delta} tokens ({percent:.1f}%)")
    return results

