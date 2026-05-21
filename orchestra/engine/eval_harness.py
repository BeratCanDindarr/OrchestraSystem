from __future__ import annotations

import json
import logging
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from orchestra import config
from orchestra.models import OrchestraRun, RunStatus

logger = logging.getLogger(__name__)

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
    # prompt construction logic here...
    return f"Judge this: {output}"

def run_judge(task_text: str, agent_output: str, run_id: str, t_hash: str, gold_standard: str = "", judge_alias: str = "cld-fast") -> EvalResult:
    from orchestra.engine.runners.ask_runner import run_ask

    judge_prompt = build_judge_prompt(task_text, "mode", agent_output, gold_standard)
    judge_run = run_ask(judge_alias, judge_prompt, emit_console=False, show_live=False)
    judge_output = judge_run.agents[0].stdout_log if judge_run.agents else ""
    
    try:
        data = json.loads(judge_output)
        return EvalResult(
            task=task_text,
            run_id=run_id,
            task_hash=t_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            mode="mode",
            output=agent_output,
            answer_quality=data.get("answer_quality", {}).get("score", 0.0),
            factual_grounding=data.get("factual_grounding", {}).get("score", 0.0),
            source_attribution=data.get("source_attribution", {}).get("score", 0.0),
            conciseness=data.get("conciseness", {}).get("score", 0.0),
            composite_score=data.get("composite_score", 0.0),
            explanation=data.get("explanation", "")
        )
    except Exception as e:
        raise ValueError(f"Failed to parse judge response: {e}")

def benchmark_context_strategies(task: str, alias: str = "cdx-fast"):
    """Benchmark full vs surgical context."""
    from orchestra.engine.runners.ask_runner import run_ask
    
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

