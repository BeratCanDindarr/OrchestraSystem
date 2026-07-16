"""Hierarchical subagent spawning: a supervisor decomposes a task into subtasks, dispatches
each to a subagent, then synthesizes the results — recursively, up to a depth limit.

Depth-first control flow bounded by ``supervisor.max_depth`` (guards against infinite
spawning). run_ask is injected so the recursion is pure and testable without real models.
A subtask that does not decompose (0 or 1 parts) is run directly as a leaf.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from orchestra import config
from orchestra.engine.eval_harness import _extract_json
from orchestra.engine.reflection import _first_output
from orchestra.models import OrchestraRun

logger = logging.getLogger(__name__)


def build_decompose_prompt(task: str) -> str:
    """Ask a planner model to split a task into independent subtasks (strict JSON)."""
    return (
        "You are a planning supervisor. Split the TASK into a small number (2-5) of "
        "independent subtasks that can each be solved on their own, then combined.\n"
        "If the task is already atomic and cannot be usefully split, return an empty list.\n"
        "Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:\n"
        '{"subtasks": ["...", "..."]}\n\n'
        f"TASK:\n{task}"
    )


def build_synthesis_prompt(task: str, parts: list[str]) -> str:
    """Combine subagent answers into one final answer for the original task."""
    blocks = "\n\n".join(f"[Subresult {i + 1}]\n{p}" for i, p in enumerate(parts))
    return (
        "You are a synthesizer. Combine the subresults below into one coherent, correct "
        "answer to the ORIGINAL TASK. Resolve overlaps, drop redundancy, and keep only what "
        "the final answer needs.\n\n"
        f"ORIGINAL TASK:\n{task}\n\n{blocks}"
    )


def _decompose(task: str, planner_alias: str, run_ask_fn: Callable[..., OrchestraRun]) -> list[str]:
    """Return the planner's subtask list for a task; empty list if atomic/unparseable."""
    plan_run = run_ask_fn(planner_alias, build_decompose_prompt(task))
    output = _first_output(plan_run)
    if not output:
        return []
    try:
        data = _extract_json(output)
    except Exception:
        logger.warning("Supervisor: could not parse decomposition", exc_info=True)
        return []
    subtasks = data.get("subtasks", [])
    if not isinstance(subtasks, list):
        return []
    return [str(s).strip() for s in subtasks if str(s).strip()]


def _supervise(
    task: str,
    depth: int,
    max_depth: int,
    *,
    run_ask_fn: Callable[..., OrchestraRun],
    worker_alias: str,
    planner_alias: str,
    synthesizer_alias: str,
) -> OrchestraRun:
    """Recursively decompose/dispatch/synthesize a task, bounded by max_depth."""
    # Leaf: at the depth limit, run the task directly with a worker.
    if depth >= max_depth:
        return run_ask_fn(worker_alias, task)

    subtasks = _decompose(task, planner_alias, run_ask_fn)
    # Atomic (0 or 1 parts): run directly rather than spawn a pointless single child.
    if len(subtasks) <= 1:
        return run_ask_fn(worker_alias, task)

    logger.info("Supervisor depth %d: split into %d subtasks", depth, len(subtasks))
    parts: list[str] = []
    for subtask in subtasks:
        sub_run = _supervise(
            subtask, depth + 1, max_depth,
            run_ask_fn=run_ask_fn, worker_alias=worker_alias,
            planner_alias=planner_alias, synthesizer_alias=synthesizer_alias,
        )
        parts.append(_first_output(sub_run) or "")

    return run_ask_fn(synthesizer_alias, build_synthesis_prompt(task, parts))


def run_supervised(
    task: str,
    *,
    max_depth: Optional[int] = None,
    worker_alias: Optional[str] = None,
    planner_alias: Optional[str] = None,
    synthesizer_alias: Optional[str] = None,
    run_ask_fn: Optional[Callable[..., OrchestraRun]] = None,
) -> OrchestraRun:
    """Public entry: supervise ``task`` hierarchically and return the final OrchestraRun.

    Defaults are read from ``[supervisor]`` config. run_ask is resolved lazily to avoid a
    circular import; inject ``run_ask_fn`` in tests to run without real models.
    """
    if run_ask_fn is None:
        from orchestra.engine.runners.ask_runner import run_ask
        run_ask_fn = run_ask

    return _supervise(
        task,
        depth=0,
        max_depth=max_depth if max_depth is not None else config.supervisor_max_depth(),
        run_ask_fn=run_ask_fn,
        worker_alias=worker_alias or config.supervisor_worker_alias(),
        planner_alias=planner_alias or config.supervisor_planner_alias(),
        synthesizer_alias=synthesizer_alias or config.supervisor_synthesizer_alias(),
    )
