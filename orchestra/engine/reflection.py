"""Self-reflection loop: critique an agent's output and retry while a critic flags issues.

After a run produces output, a critic model checks it for concrete errors; if the critic says
the answer needs revision, the original alias retries with the critique fed back as guidance.
The loop is bounded by ``reflection.max_rounds`` and the run budget, and is config-gated
(``reflection.enabled``, default off). run_ask is injected so the loop is pure and testable.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any, Callable, Optional

from orchestra import config
from orchestra.models import AgentStatus, OrchestraRun

logger = logging.getLogger(__name__)

# Set while a reflection loop is running so nested run_ask calls (critique + retry) do not
# recursively spawn their own reflection -> unbounded recursion.
_in_reflection: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_reflection", default=False)


def is_reflecting() -> bool:
    """True while a reflection loop is running (guards against recursive reflection)."""
    return _in_reflection.get()


def _first_output(run: OrchestraRun) -> Optional[str]:
    """Return the first completed agent's non-empty output text, or None."""
    for agent in run.agents:
        if agent.status == AgentStatus.COMPLETED and agent.stdout_log.strip():
            return agent.stdout_log
    return None


def build_critique_prompt(task: str, output: str) -> str:
    """Ask a critic model to judge an answer and reply with a strict JSON verdict."""
    return (
        "You are a strict reviewer. Decide whether the ANSWER to the TASK needs revision.\n"
        "Look for: factual errors, missing requirements, faulty logic, unsupported claims, "
        "and unsafe or incorrect code. Be specific and fair; do not invent faults.\n"
        "Respond with ONLY a JSON object of this exact shape, no prose, no markdown fences:\n"
        '{"needs_revision": false, "issues": ["..."], "guidance": "one sentence on how to fix"}\n'
        "Set needs_revision to true ONLY when there is a concrete, fixable problem.\n\n"
        f"TASK:\n{task}\n\nANSWER:\n{output}"
    )


def build_retry_prompt(task: str, issues: list[str], guidance: str) -> str:
    """Feed the critic's findings back to the original alias for a corrected answer."""
    issue_lines = "\n".join(f"- {i}" for i in issues) or "- (see guidance)"
    return (
        f"{task}\n\n"
        "Your previous answer had the following problems flagged by a reviewer:\n"
        f"{issue_lines}\n"
        f"Guidance: {guidance}\n\n"
        "Produce a corrected answer that resolves these problems. Do not repeat the mistakes."
    )


def _parse_verdict(text: str) -> Optional[dict]:
    """Parse the critic's JSON verdict, tolerating fences/prose. None if unparseable."""
    try:
        from orchestra.engine.eval_harness import _extract_json
        return _extract_json(text)
    except Exception:
        logger.warning("Reflection: could not parse critic verdict", exc_info=True)
        return None


def _record_lesson(task: str, issues: list[str], guidance: str, memory: Any) -> None:
    """Write a lesson to episodic memory so RAG can surface it for future similar tasks.

    Embeds on the TASK (content, no summary) so retrieval matches by task similarity; the
    lesson itself is stored in metadata under ``reflection_lesson``. No-op when disabled.
    """
    if not config.reflection_record_lessons():
        return
    lesson = f"avoid: {'; '.join(i for i in issues if i)}. {guidance}".strip()
    try:
        if memory is None:
            from orchestra.storage.memory import get_memory
            memory = get_memory()
        memory.add(
            task,
            metadata={"reflection_lesson": lesson},
            source="reflection_lesson",
            owner="orchestra",
            sensitivity_label="normal",
            memory_type="episodic",
        )
    except Exception:
        logger.warning("Reflection: failed to record lesson", exc_info=True)


def _within_budget(runs: list[OrchestraRun]) -> bool:
    """True if the cumulative cost of all reflection runs is within the configured budget."""
    max_budget = float(config.execution_config().get("max_budget_usd", 0))
    if max_budget == 0:
        return True
    total = sum(
        agent.estimated_cost_usd
        for run in runs
        for agent in run.agents
        if agent.status == AgentStatus.COMPLETED
    )
    return total <= max_budget


def reflect(
    alias: str,
    task: str,
    base_run: OrchestraRun,
    *,
    run_ask_fn: Callable[..., OrchestraRun],
    critique_alias: Optional[str] = None,
    max_rounds: Optional[int] = None,
    memory: Any = None,
) -> OrchestraRun:
    """Critique base_run's output and retry while the critic flags issues.

    Returns the latest revised run, or ``base_run`` if nothing needed fixing. Bounded by
    ``max_rounds`` and the run budget; guards against recursive reflection.
    """
    if is_reflecting() or _first_output(base_run) is None:
        return base_run

    rounds = max_rounds if max_rounds is not None else config.reflection_max_rounds()
    critic = critique_alias or config.reflection_critique_alias()

    token = _in_reflection.set(True)
    try:
        current = base_run
        history = [base_run]
        for round_i in range(max(0, rounds)):
            output = _first_output(current)
            if output is None:
                break
            critique_run = run_ask_fn(critic, build_critique_prompt(task, output))
            history.append(critique_run)
            verdict = _parse_verdict(_first_output(critique_run) or "")
            if not verdict or not verdict.get("needs_revision"):
                break  # accepted
            issues = [str(i) for i in verdict.get("issues", [])]
            guidance = str(verdict.get("guidance", ""))
            _record_lesson(task, issues, guidance, memory)
            revised = run_ask_fn(alias, build_retry_prompt(task, issues, guidance))
            if _first_output(revised) is None:
                break  # retry produced nothing usable; keep the last good run
            history.append(revised)
            current = revised
            logger.info("Reflection round %d: revised output for %s", round_i + 1, alias)
            if not _within_budget(history):
                logger.info("Reflection: budget reached, stopping after round %d", round_i + 1)
                break
        return current
    finally:
        _in_reflection.reset(token)
