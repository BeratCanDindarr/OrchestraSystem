"""Graph Engine: Directed Acyclic Graph (DAG) state machine for agents.

Supports:
  - Static edges defined at plan time (default, on_success, on_failure)
  - Dynamic conditional edges: add_conditional_node() inserts a new node+edge
    mid-run based on the output of a completed node
  - max_steps hard cap: prevents unbounded growth from dynamic insertion
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

_DEFAULT_MAX_STEPS = 20  # hard cap on total nodes executed per run


@dataclass
class Node:
    name: str
    action: str   # task description
    agent: str    # alias to run
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    dynamic: bool = False  # True if inserted at runtime


@dataclass
class Edge:
    source: str
    target: str
    condition: Optional[str] = None  # default | on_success | on_failure


class OrchestraGraph:
    def __init__(self, run_id: str, max_steps: int = _DEFAULT_MAX_STEPS):
        self.run_id = run_id
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.current_node_name: Optional[str] = None
        self.max_steps: int = max_steps
        self._steps_executed: int = 0

    # ── Static graph construction ──────────────────────────────────────────

    def add_node(self, name: str, action: str, agent: str) -> None:
        self.nodes[name] = Node(name=name, action=action, agent=agent)

    def add_edge(self, source: str, target: str, condition: str = "default") -> None:
        self.edges.append(Edge(source=source, target=target, condition=condition))

    # ── Dynamic node insertion (conditional edges) ─────────────────────────

    def add_conditional_node(
        self,
        *,
        after: str,
        name: str,
        action: str,
        agent: str,
        condition: str = "on_success",
    ) -> bool:
        """Insert a new node dynamically and connect it from an existing node.

        Returns False (no-op) if:
          - max_steps would be exceeded
          - a node with this name already exists
          - the source node (after) does not exist
        """
        if self._steps_executed >= self.max_steps:
            return False
        if name in self.nodes:
            return False
        if after not in self.nodes:
            return False
        self.nodes[name] = Node(name=name, action=action, agent=agent, dynamic=True)
        self.edges.append(Edge(source=after, target=name, condition=condition))
        return True

    # ── Step counter ───────────────────────────────────────────────────────

    def record_step(self) -> bool:
        """Increment step counter. Returns False if max_steps cap is reached."""
        if self._steps_executed >= self.max_steps:
            return False
        self._steps_executed += 1
        return True

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self._steps_executed)

    # ── Traversal ──────────────────────────────────────────────────────────

    def get_next_nodes(self, current_name: str, last_result_status: str) -> List[str]:
        """Determine which node(s) to run next based on the result of the current node."""
        next_nodes = []
        for edge in self.edges:
            if edge.source == current_name:
                if edge.condition == "default":
                    next_nodes.append(edge.target)
                elif edge.condition == "on_success" and last_result_status == "completed":
                    next_nodes.append(edge.target)
                elif edge.condition == "on_failure" and last_result_status == "failed":
                    next_nodes.append(edge.target)
        return next_nodes

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "max_steps": self.max_steps,
            "steps_executed": self._steps_executed,
            "nodes": {n: vars(v) for n, v in self.nodes.items()},
            "edges": [vars(e) for e in self.edges],
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
