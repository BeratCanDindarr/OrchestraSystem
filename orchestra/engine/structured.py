"""Answers that must be a JSON object: find it, check it, and ask once more when it is not usable.

A judge, a critic or a planner is asked for "ONLY a JSON object". Models still answer in
prose now and then, or with an object of another shape. Taking that for "no verdict" silently
drops the step; asking once more, with the reason, usually gets the object.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

from orchestra.models import AgentStatus

MAX_QUOTED_ANSWER_CHARS = 1500

Check = Callable[[dict], Optional[str]]


def extract_object(text: str) -> dict:
    """The JSON object in a model's answer, tolerating code fences and prose around it.

    Raises ValueError when there is none.
    """
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the answer contains no JSON object")
    candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as error:
        try:
            import json_repair
        except ImportError:
            raise ValueError(f"the JSON in the answer is not valid: {error}") from error
        data = json_repair.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    return data


def first_output(run: Any) -> str:
    """The text of the run's first completed agent; failing that, of its first agent; or ''."""
    agents = getattr(run, "agents", None) or []
    for agent in agents:
        if agent.status == AgentStatus.COMPLETED and (agent.stdout_log or "").strip():
            return str(agent.stdout_log)
    first = agents[0].stdout_log if agents else ""
    return first if isinstance(first, str) else ""


def _problem(text: str, check: Optional[Check]) -> tuple[Optional[dict], Optional[str]]:
    try:
        data = extract_object(text)
    except ValueError as error:
        return None, str(error)
    problem = check(data) if check else None
    return (None, problem) if problem else (data, None)


def correction_prompt(prompt: str, answer: str, problem: str) -> str:
    """The original request, followed by what was wrong with the answer to it."""
    return (
        f"{prompt}\n\n"
        f"[Your previous answer could not be used: {problem}. "
        "Answer again with ONLY the JSON object described above: no prose, no code fences.]\n\n"
        f"Previous answer:\n{answer[:MAX_QUOTED_ANSWER_CHARS]}"
    )


def ask_for_object(
    ask: Callable[..., Any],
    alias: str,
    prompt: str,
    *,
    check: Optional[Check] = None,
    **options,
) -> tuple[Optional[dict], Any]:
    """Ask for a JSON object; when the answer is not one (or fails *check*), ask once more.

    *ask* is called as ``ask(alias, prompt, **options)`` and returns a run. *check* returns
    what is wrong with a parsed object, or None when it is fine. Returns the object (None
    when both answers were unusable) and the run that produced the last answer.
    """
    run = ask(alias, prompt, **options)
    answer = first_output(run)
    data, problem = _problem(answer, check)
    if data is not None or not answer:
        return data, run                # usable, or the agent did not answer at all
    retry = ask(alias, correction_prompt(prompt, answer, problem or "it was not readable"), **options)
    data, _ = _problem(first_output(retry), check)
    return data, retry
