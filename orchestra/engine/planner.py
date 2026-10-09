"""Graph-Based Planning: transform tasks into a Directed Acyclic Graph."""
from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional
from orchestra import config
from orchestra.engine.parallel import run_parallel
from orchestra.engine.graph import OrchestraGraph, build_graph_from_plan
from orchestra.models import OrchestraRun, AgentStatus

_GRAPH_PLANNER_PROMPT = """\
Analyze this complex task and design an execution GRAPH.
You must return ONLY a JSON object with 'nodes' and 'edges'.

NODES: {{"name": "unique_id", "action": "detailed instruction", "agent": "alias"}}
EDGES: {{"source": "id", "target": "id", "condition": "on_success | on_failure | default"}}

TASK:
{task}

Example Format:
{{
  "nodes": [
    {{"name": "research", "action": "Analyze code", "agent": "gmn-pro"}},
    {{"name": "implement", "action": "Apply fix", "agent": "cdx-deep"}}
  ],
  "edges": [
    {{"source": "research", "target": "implement", "condition": "on_success"}}
  ]
}}
"""

def create_graph_plan(run: OrchestraRun) -> Optional[OrchestraGraph]:
    """Produce a graph-based plan with the configured planner model."""
    prompt = _GRAPH_PLANNER_PROMPT.format(task=run.task)
    results = run_parallel(run, [(config.supervisor_planner_alias(), prompt)], show_live=False, install_signal_handlers=False)

    if not results or results[0].status != AgentStatus.COMPLETED:
        return None

    raw = results[0].stdout_log

    # 1. Try markdown code block extraction first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        # 2. Fall back to greedy regex
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(
                f"No JSON object found in planner response. Raw response: {raw[:500]}..."
            )
        candidate = match.group(0)

    # 3. Validate / repair JSON
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        try:
            import json_repair
        except ImportError:
            raise ValueError(
                f"Invalid JSON in planner response and json_repair is not installed. "
                f"Error: {exc}. Extracted JSON: {candidate[:500]}..."
            ) from exc
        repaired = json_repair.repair_json(candidate)
        if not repaired:
            raise ValueError(
                f"json_repair could not fix planner response JSON. "
                f"Error: {exc}. Extracted JSON: {candidate[:500]}..."
            ) from exc
        candidate = repaired
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc2:
            raise ValueError(
                f"Repaired JSON is still invalid. "
                f"Error: {exc2}. Repaired JSON: {candidate[:500]}..."
            ) from exc2

    return build_graph_from_plan(run.run_id, candidate)
