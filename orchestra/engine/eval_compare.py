"""Run a task in ask, dual and consensus mode, judge each answer, rank them (`orchestra eval`).

Moved from cli.py so the terminal only prints: the runs and the judge are the engine's work.
"""
from __future__ import annotations

import concurrent.futures
from contextvars import copy_context
from datetime import datetime, timezone
from typing import Callable, Optional

from orchestra import config
from orchestra.engine.eval_judge import EvalResult, build_judge_prompt, parse_judge_response, save_eval, task_hash

MODES = ("ask", "dual", "consensus")


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _run_mode(task: str, mode: str) -> dict:
    """Run one mode; the result holds its answer, or the error that stopped it."""
    from orchestra.engine.runners import run_ask, run_dual

    quiet = {"emit_console": False, "show_live": False}
    try:
        if mode == "ask":
            run = run_ask(config.default_light_alias(), task, **quiet)
        else:
            run = run_dual(task, stances=mode == "consensus", **quiet)
        output = run.summary or (run.agents[0].stdout_log if run.agents and run.agents[0].stdout_log else "")
        return {"mode": mode, "run_id": run.run_id, "status": run.status.value,
                "confidence": round(float(run.avg_confidence or 0), 3),
                "cost_usd": round(float(run.total_cost_usd or 0), 6), "output": output, "error": None}
    except Exception as e:
        return {"mode": mode, "run_id": "", "status": "failed", "confidence": 0.0, "cost_usd": 0.0,
                "output": "", "error": str(e)}


def _judge(task: str, task_id: str, res: dict, note: Callable[[str], None]) -> None:
    """Score one mode's answer in place; a judge that fails scores it 0 and says why."""
    from orchestra.engine.runners import run_ask

    if res["error"] or not res["output"]:
        res["judge"] = {"score": 0.0, "reasoning": "No output to judge"}
        res["cost_per_quality"] = 0.0
        return
    mode = res["mode"]
    try:
        # The judge is a step of the run it judges: unprefixed, uncached, not a run of its own.
        judge_run = run_ask(str(config.personas_config().get("reviewer") or "cld-fast"),
                            build_judge_prompt(task, mode, res["output"]),
                            emit_console=False, show_live=False, internal=True, parent_run_id=res["run_id"])
        judge_output = judge_run.agents[0].stdout_log if judge_run.agents else ""
        judge_cost = round(float(judge_run.total_cost_usd or 0), 6)
    except Exception as judge_err:
        note(f"[red]  {mode}: judge error: {judge_err}[/red]")
        res["judge"] = {"score": 0.0, "reasoning": f"Judge error: {judge_err}"}
        res["cost_per_quality"] = 0.0
        return
    try:
        criteria = parse_judge_response(judge_output)
    except Exception as parse_err:
        note(f"[red]  {mode}: judge parse error: {parse_err}[/red]")
        res["judge"] = {"score": 0.0, "reasoning": f"Parse error: {parse_err}"}
        res["cost_per_quality"] = 0.0
        return
    score = criteria.composite_score()
    res["judge"] = {"score": score, "quality": criteria.answer_quality, "grounding": criteria.factual_grounding,
                    "conciseness": criteria.conciseness, "reasoning": criteria.answer_quality_reasoning}
    res["judge_cost_usd"] = judge_cost
    res["cost_per_quality"] = round((res["cost_usd"] + judge_cost) / max(score, 0.1), 6)
    save_eval(EvalResult(task=task, run_id=res["run_id"], task_hash=task_id, created_at=_now_str(), mode=mode,
                         output=res["output"], judge_criteria=criteria, judge_cost_usd=judge_cost,
                         judge_timestamp=_now_str(), composite_score=score))
    note(f"[green]  {mode}: scored {score:.2f}[/green]")


def evaluate_modes(task: str, note: Optional[Callable[[str], None]] = None) -> dict:
    """Run the three modes in parallel, judge each answer, and rank them by score (best first).

    `note` receives progress lines (Rich markup) as the work goes on.
    """
    note = note or (lambda line: None)
    task_id = task_hash(task)
    note("[dim]Running ask / dual / consensus in parallel...[/dim]")
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODES)) as pool:
        futures = [pool.submit(copy_context().run, _run_mode, task, mode) for mode in MODES]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results[res["mode"]] = res
            note(f"[yellow]  {res['mode']}: {res['error']}[/yellow]" if res["error"]
                 else f"[green]  {res['mode']}: completed[/green]")
    note("\n[dim]Judging outputs...[/dim]")
    for res in results.values():
        _judge(task, task_id, res, note)
    ordered = sorted(results.values(), key=lambda r: -float((r.get("judge") or {}).get("score") or 0))
    return {"task": task, "task_hash": task_id, "results": ordered}
