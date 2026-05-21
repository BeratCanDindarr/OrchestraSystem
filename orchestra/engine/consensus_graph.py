from __future__ import annotations
"""Consensus graph data model for multi-agent runs.

Nodes = agents (colored by model family).
Edges = pairwise agreement / disagreement / partial alignment.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Literal

EdgeKind = Literal["aligned", "dissent", "partial"]
ConsensusLevel = Literal["unanimous", "majority", "split"]


@dataclass
class ConsensusNode:
    """An agent in the consensus graph."""

    alias: str
    model_family: str = ""
    output_digest: str = ""
    confidence: float = 0.0
    status: str = ""
    is_red_team: bool = False
    is_synthesizer: bool = False


@dataclass
class ConsensusEdge:
    """Pairwise relationship between two agents."""

    source: str
    target: str
    kind: EdgeKind
    similarity_score: float = 0.0


@dataclass
class ConsensusGraph:
    """Full consensus DAG for a run."""

    run_id: str = ""
    nodes: dict[str, ConsensusNode] = field(default_factory=dict)
    edges: list[ConsensusEdge] = field(default_factory=list)
    consensus_level: ConsensusLevel = "split"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "consensus_level": self.consensus_level,
            "nodes": [
                {
                    "alias": n.alias,
                    "model_family": n.model_family,
                    "output_digest": n.output_digest,
                    "confidence": n.confidence,
                    "status": n.status,
                    "is_red_team": n.is_red_team,
                    "is_synthesizer": n.is_synthesizer,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "kind": e.kind,
                    "similarity_score": e.similarity_score,
                }
                for e in self.edges
            ],
        }
