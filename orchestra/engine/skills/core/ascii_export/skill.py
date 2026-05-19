"""Core skill: render consensus graph as Unicode ASCII art."""
from __future__ import annotations

from orchestra.engine.consensus import format_consensus_ascii
from orchestra.engine.consensus_graph import ConsensusGraph
from orchestra.engine.skills.base import SkillContext, SkillResult


def can_execute(context: SkillContext) -> bool:
    return True


def execute(context: SkillContext) -> SkillResult:
    from orchestra.engine.consensus import build_consensus_graph
    from orchestra.storage.db import get_db

    run_id = context.args.get("run_id", "")
    if not run_id:
        return SkillResult(success=False, output="run_id is required")

    # Load run manifest from DB
    conn = get_db()
    row = conn.execute(
        "SELECT run_id, mode, status, task FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if not row:
        return SkillResult(success=False, output=f"Run '{run_id}' not found")

    agent_rows = conn.execute(
        "SELECT alias, provider, model, status, confidence, soft_failed FROM agents WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    from orchestra.models import OrchestraRun, AgentRun, AgentStatus, RunStatus

    run = OrchestraRun(
        run_id=row["run_id"],
        mode=row["mode"],
        task=row["task"],
        status=RunStatus(row["status"]),
    )
    for ar in agent_rows:
        run.agents.append(
            AgentRun(
                alias=ar["alias"],
                provider=ar["provider"],
                model=ar["model"],
                status=AgentStatus(ar["status"]),
                confidence=ar["confidence"],
                soft_failed=bool(ar["soft_failed"]),
            )
        )

    graph = build_consensus_graph(run)
    text = format_consensus_ascii(graph)
    return SkillResult(success=True, output=text)
