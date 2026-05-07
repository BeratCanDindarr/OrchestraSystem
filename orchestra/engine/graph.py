"""Graph Engine: Directed Acyclic Graph (DAG) state machine for agents."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

@dataclass
class Node:
    name: str
    action: str  # task description
    agent: str   # alias to run
    status: str = "pending" # pending, running, completed, failed
    result: Optional[str] = None

@dataclass
class Edge:
    source: str
    target: str
    condition: Optional[str] = None # e.g. "if_failed", "if_success", "needs_more_info"

class OrchestraGraph:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.current_node_name: Optional[str] = None

    def add_node(self, name: str, action: str, agent: str):
        self.nodes[name] = Node(name=name, action=action, agent=agent)

    def add_edge(self, source: str, target: str, condition: str = "default"):
        self.edges.append(Edge(source=source, target=target, condition=condition))

    def get_next_nodes(self, current_name: str, last_result_status: str) -> List[str]:
        """Determine which node(s) to run next based on the result of the current node."""
        next_nodes = []
        for edge in self.edges:
            if edge.source == current_name:
                # Basic condition logic
                if edge.condition == "default":
                    next_nodes.append(edge.target)
                elif edge.condition == "on_success" and last_result_status == "completed":
                    next_nodes.append(edge.target)
                elif edge.condition == "on_failure" and last_result_status == "failed":
                    next_nodes.append(edge.target)
        return next_nodes

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "nodes": {n: vars(v) for n, v in self.nodes.items()},
            "edges": [vars(e) for e in self.edges]
        }

def build_graph_from_plan(run_id: str, plan_json: str) -> Optional[OrchestraGraph]:
    """Helper to convert LLM output into an executable Graph."""
    try:
        data = json.loads(plan_json)
        graph = OrchestraGraph(run_id)
        for node in data.get("nodes", []):
            graph.add_node(node["name"], node["action"], node["agent"])
        for edge in data.get("edges", []):
            graph.add_edge(edge["source"], edge["target"], edge.get("condition", "default"))
        return graph
    except Exception:
        return None
