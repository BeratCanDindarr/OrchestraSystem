"""Core skill: render consensus graph as SVG/PNG via Graphviz."""
from __future__ import annotations

import shutil

from orchestra.engine.consensus import format_consensus_dot
from orchestra.engine.skills.base import SkillContext, SkillResult


def can_execute(context: SkillContext) -> bool:
    return shutil.which("dot") is not None


def execute(context: SkillContext) -> SkillResult:
    import subprocess
    from pathlib import Path

    from orchestra.engine.consensus import build_consensus_graph
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
    from orchestra.storage.db import get_db

    run_id = context.args.get("run_id", "")
    fmt = context.args.get("format", "svg")
    out_path = context.args.get("out", f"{run_id}_consensus.{fmt}")
    if not run_id:
        return SkillResult(success=False, output="run_id is required")

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
    dot = format_consensus_dot(graph)

    try:
        result = subprocess.run(
            ["dot", f"-T{fmt}", "-o", out_path],
            input=dot,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return SkillResult(success=False, output=result.stderr or "graphviz failed")
        return SkillResult(success=True, output=f"Saved to {Path(out_path).resolve()}")
    except FileNotFoundError:
        return SkillResult(success=False, output="graphviz 'dot' not found in PATH")
