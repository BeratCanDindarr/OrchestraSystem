"""Consensus graph builder for multi-agent runs.

Leverages Orchestra's own features:
- Semantic index (Ollama embeddings) for agent-output similarity
- Context compressor for summarising long outputs before comparison
- Adversarial flags (consensus_challenged) to mark red-team nodes
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING

from orchestra.engine.consensus_graph import (
    ConsensusEdge,
    ConsensusGraph,
    ConsensusNode,
    ConsensusLevel,
    EdgeKind,
)
from orchestra.engine.policy_engine import PolicyEngine

if TYPE_CHECKING:
    from orchestra.models import OrchestraRun, AgentRun

logger = logging.getLogger(__name__)

# Similarity thresholds
_ALIGNED = 0.85
_PARTIAL_LOW = 0.60


def _alias_to_family(alias: str) -> str:
    mapping = {
        "cdx": "codex",
        "gmn": "gemini",
        "cld": "claude",
        "oll": "ollama",
        "ds": "deepseek",
        "kimi": "kimi",
    }
    prefix = alias.split("-")[0]
    return mapping.get(prefix, prefix)


def _extract_output(agent: "AgentRun") -> str:
    """Return a representative snippet of the agent's output."""
    text = (agent.stdout_log or "").strip()
    # Cap to ~2k chars to keep embedding fast
    if len(text) > 2000:
        text = text[:2000]
    return text


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(inter) / len(union)


def _embed(text: str) -> list[float] | None:
    """Try Ollama embedding; return None on failure."""
    try:
        from orchestra.providers.ollama import OllamaProvider

        provider = OllamaProvider()
        vec = provider.embed(text)
        if vec:
            return vec
    except Exception as exc:
        logger.debug("Ollama embed failed: %s", exc)
    return None


def _compute_similarity(text_a: str, text_b: str) -> float:
    """Semantic similarity with Jaccard fallback."""
    vec_a = _embed(text_a)
    vec_b = _embed(text_b)
    if vec_a is not None and vec_b is not None:
        return _cosine_similarity(vec_a, vec_b)
    return _jaccard_similarity(text_a, text_b)


def _kind_from_score(score: float) -> EdgeKind:
    if score >= _ALIGNED:
        return "aligned"
    if score >= _PARTIAL_LOW:
        return "partial"
    return "dissent"


def _consensus_level(edges: list[ConsensusEdge], node_count: int) -> ConsensusLevel:
    if node_count <= 1:
        return "unanimous"
    aligned_pairs = sum(1 for e in edges if e.kind == "aligned")
    total_pairs = len(edges)
    if total_pairs == 0:
        return "split"
    ratio = aligned_pairs / total_pairs
    if ratio >= 0.99:
        return "unanimous"
    if ratio >= 0.5:
        return "majority"
    return "split"


def build_consensus_graph(run: "OrchestraRun") -> ConsensusGraph:
    """Build a consensus graph from a completed run.

    Uses Orchestra's semantic embedding when available, otherwise falls back
    to Jaccard token overlap.
    """
    graph = ConsensusGraph(run_id=run.run_id)
    agents = [a for a in run.agents if a.status.value == "completed"]

    # Build nodes
    for agent in agents:
        digest = hashlib.sha256((agent.stdout_log or "").encode()).hexdigest()[:16]
        graph.nodes[agent.alias] = ConsensusNode(
            alias=agent.alias,
            model_family=_alias_to_family(agent.alias),
            output_digest=digest,
            confidence=agent.confidence,
            status=agent.status.value,
            is_red_team=run.consensus_challenged and agent.alias == run.agents[-1].alias if run.agents else False,
            is_synthesizer=agent.alias.startswith("syn"),
        )

    # Build edges (pairwise, undirected stored as two directed or just one)
    texts = {a.alias: _extract_output(a) for a in agents}
    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            score = _compute_similarity(texts[a.alias], texts[b.alias])
            kind = _kind_from_score(score)
            graph.edges.append(
                ConsensusEdge(source=a.alias, target=b.alias, kind=kind, similarity_score=round(score, 4))
            )

    graph.consensus_level = _consensus_level(graph.edges, len(agents))
    return graph


def format_consensus_ascii(graph: ConsensusGraph) -> str:
    """Render the consensus graph as Unicode box art."""
    if not graph.nodes:
        return "(empty consensus graph)\n"

    lines: list[str] = []
    lines.append(f"Consensus Graph  —  {graph.run_id}  —  level: {graph.consensus_level}")
    lines.append("")

    # Node list
    for node in graph.nodes.values():
        rt = " [RED-TEAM]" if node.is_red_team else ""
        synth = " [SYNTH]" if node.is_synthesizer else ""
        lines.append(
            f"  ┌─ {node.alias} ({node.model_family}){rt}{synth}\n"
            f"  │   conf={node.confidence:.2f}  digest={node.output_digest}"
        )

    lines.append("")
    if not graph.edges:
        lines.append("  (no edges)")
        return "\n".join(lines)

    # Edge list
    for edge in graph.edges:
        symbol = "✓" if edge.kind == "aligned" else "~" if edge.kind == "partial" else "✗"
        lines.append(
            f"  {edge.source} {symbol}── {edge.target}   "
            f"[{edge.kind}]  score={edge.similarity_score:.3f}"
        )

    return "\n".join(lines) + "\n"


def format_consensus_dot(graph: ConsensusGraph) -> str:
    """Render the consensus graph as Graphviz DOT."""
    family_colors = {
        "codex": "#4285F4",
        "gemini": "#EA4335",
        "claude": "#D7AEFB",
        "ollama": "#34A853",
        "deepseek": "#1A73E8",
        "kimi": "#F9AB00",
    }
    lines = ["digraph consensus {", "  rankdir=TB;", "  node [shape=box, style=filled, fontname=Helvetica];"]

    for node in graph.nodes.values():
        color = family_colors.get(node.model_family, "#cccccc")
        label = f"{node.alias}\\n{node.model_family}"
        if node.is_red_team:
            color = "#ff4444"
            label += "\\n[RED-TEAM]"
        if node.is_synthesizer:
            color = "#44ff44"
            label += "\\n[SYNTH]"
        lines.append(f'  "{node.alias}" [label="{label}", fillcolor="{color}"];')

    edge_colors = {"aligned": "#34A853", "partial": "#F9AB00", "dissent": "#EA4335"}
    for edge in graph.edges:
        col = edge_colors.get(edge.kind, "#888888")
        lines.append(
            f'  "{edge.source}" -> "{edge.target}" '
            f'[label="{edge.kind} ({edge.similarity_score:.2f})", color="{col}", penwidth=2];'
        )

    lines.append("}")
    return "\n".join(lines) + "\n"
