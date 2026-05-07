"""Graph-Based Planning: transform tasks into a Directed Acyclic Graph."""
from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional
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
    """Produce a graph-based plan using gmn-pro."""
    prompt = _GRAPH_PLANNER_PROMPT.format(task=run.task)
    results = run_parallel(run, [("gmn-pro", prompt)], show_live=False, install_signal_handlers=False)
    
    if not results or results[0].status != AgentStatus.COMPLETED:
        return None
        
    try:
        raw = results[0].stdout_log
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match: return None
        return build_graph_from_plan(run.run_id, match.group(0))
    except Exception:
        return None
