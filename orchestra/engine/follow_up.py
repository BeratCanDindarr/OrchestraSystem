"""Follow-up runs: a run may continue an earlier one and is then given that discussion.

There is no separate conversation object. A run records the run it follows
(``follows_run_id``); the discussion is the chain of those runs, read back from their
manifests. Any finished run can therefore be continued, and a run that follows nothing
is exactly what it always was.
"""
from __future__ import annotations

from typing import Optional

from orchestra import config
from orchestra.engine import artifacts

_STEP_ALIASES = {"verifier"}


def max_turns() -> int:
    """How many earlier turns a follow-up is shown ([follow_up] max_turns)."""
    return int(config.get("follow_up", "max_turns", 6))


def answer_chars() -> int:
    """How much of each earlier answer is shown ([follow_up] answer_chars)."""
    return int(config.get("follow_up", "answer_chars", 4000))


def history(run_id: Optional[str]) -> list[dict]:
    """The turns leading up to and including ``run_id``, oldest first, newest ``max_turns``."""
    turns: list[dict] = []
    seen: set[str] = set()
    limit = max_turns()
    while run_id and run_id not in seen and len(turns) < limit:
        seen.add(run_id)
        manifest = artifacts.load_manifest(run_id)
        if not manifest:
            break
        agents = [a.get("alias", "") for a in manifest.get("agents") or [] if a.get("alias") not in _STEP_ALIASES]
        turns.append({
            "run_id": run_id,
            "task": str(manifest.get("task") or ""),
            "answer": str(manifest.get("summary") or ""),
            "agents": agents,
        })
        run_id = manifest.get("follows_run_id")
    return list(reversed(turns))


def with_history(follows_run_id: Optional[str], prompt: str) -> str:
    """``prompt`` preceded by the discussion it continues; unchanged when it continues nothing."""
    turns = history(follows_run_id)
    if not turns:
        return prompt
    limit = answer_chars()
    lines = ["[EARLIER IN THIS DISCUSSION: oldest turn first. The new message below continues it.]"]
    for number, turn in enumerate(turns, start=1):
        answer = turn["answer"].strip()
        if len(answer) > limit:
            answer = answer[:limit] + "\n[... the rest of this answer is left out]"
        who = ", ".join(turn["agents"]) or "unknown"
        lines.append(f"--- Turn {number} ---\nAsked: {turn['task'].strip()}\nAnswered ({who}): {answer or '(no answer was recorded)'}")
    lines.append("[NEW MESSAGE]")
    return "\n\n".join(lines) + "\n" + prompt


def previous_alias(follows_run_id: Optional[str]) -> Optional[str]:
    """The agent that answered the run being followed, when one agent did."""
    manifest = artifacts.load_manifest(follows_run_id) if follows_run_id else None
    agents = [a.get("alias") for a in (manifest or {}).get("agents") or [] if a.get("alias") not in _STEP_ALIASES]
    return agents[0] if len(agents) == 1 else None
