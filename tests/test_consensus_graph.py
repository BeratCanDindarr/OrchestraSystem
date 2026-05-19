"""Tests for consensus graph visualization."""
from __future__ import annotations

from orchestra.engine.consensus import (
    build_consensus_graph,
    format_consensus_ascii,
    format_consensus_dot,
    _jaccard_similarity,
    _kind_from_score,
    _consensus_level,
)
from orchestra.engine.consensus_graph import (
    ConsensusEdge,
    ConsensusGraph,
    ConsensusNode,
)
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


class TestSimilarity:
    def test_jaccard_identical(self) -> None:
        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_jaccard_disjoint(self) -> None:
        assert _jaccard_similarity("abc def", "ghi jkl") == 0.0

    def test_jaccard_partial(self) -> None:
        score = _jaccard_similarity("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0


class TestKindFromScore:
    def test_aligned(self) -> None:
        assert _kind_from_score(0.9) == "aligned"

    def test_partial(self) -> None:
        assert _kind_from_score(0.7) == "partial"

    def test_dissent(self) -> None:
        assert _kind_from_score(0.3) == "dissent"


class TestConsensusLevel:
    def test_unanimous(self) -> None:
        edges = [
            ConsensusEdge("a", "b", "aligned", 0.9),
            ConsensusEdge("a", "c", "aligned", 0.95),
        ]
        assert _consensus_level(edges, 3) == "unanimous"

    def test_majority(self) -> None:
        edges = [
            ConsensusEdge("a", "b", "aligned", 0.9),
            ConsensusEdge("a", "c", "dissent", 0.3),
        ]
        assert _consensus_level(edges, 3) == "majority"

    def test_split(self) -> None:
        edges = [
            ConsensusEdge("a", "b", "dissent", 0.3),
            ConsensusEdge("a", "c", "dissent", 0.2),
        ]
        assert _consensus_level(edges, 3) == "split"


class TestBuildConsensusGraph:
    def test_empty_run(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="t")
        graph = build_consensus_graph(run)
        assert graph.run_id == "r1"
        assert graph.nodes == {}
        assert graph.edges == []

    def test_single_agent(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="t")
        run.agents.append(AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED, stdout_log="output a"))
        graph = build_consensus_graph(run)
        assert len(graph.nodes) == 1
        assert "cdx-deep" in graph.nodes
        assert graph.consensus_level == "unanimous"

    def test_two_agents_aligned(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="t")
        run.agents.append(AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED, stdout_log="hello world foo bar"))
        run.agents.append(AgentRun(alias="gmn-pro", provider="gemini", model="m", status=AgentStatus.COMPLETED, stdout_log="hello world foo bar"))
        graph = build_consensus_graph(run)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.edges[0].kind == "aligned"
        assert graph.consensus_level == "unanimous"

    def test_two_agents_dissent(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="t")
        run.agents.append(AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED, stdout_log="completely different text here"))
        run.agents.append(AgentRun(alias="gmn-pro", provider="gemini", model="m", status=AgentStatus.COMPLETED, stdout_log="another unrelated thing altogether"))
        graph = build_consensus_graph(run)
        assert graph.edges[0].kind == "dissent"

    def test_red_team_flag(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="t", consensus_challenged=True)
        run.agents.append(AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED, stdout_log="x"))
        run.agents.append(AgentRun(alias="cld-fast", provider="claude", model="m", status=AgentStatus.COMPLETED, stdout_log="y"))
        graph = build_consensus_graph(run)
        # Last agent is flagged as red-team when consensus_challenged=True
        assert graph.nodes[run.agents[-1].alias].is_red_team is True


class TestAsciiExport:
    def test_empty_graph(self) -> None:
        graph = ConsensusGraph(run_id="r1")
        text = format_consensus_ascii(graph)
        assert "empty" in text

    def test_nodes_and_edges(self) -> None:
        graph = ConsensusGraph(run_id="r1")
        graph.nodes["a"] = ConsensusNode(alias="a", model_family="codex", confidence=0.9)
        graph.nodes["b"] = ConsensusNode(alias="b", model_family="gemini", confidence=0.8)
        graph.edges.append(ConsensusEdge("a", "b", "aligned", 0.9))
        text = format_consensus_ascii(graph)
        assert "a (codex)" in text
        assert "b (gemini)" in text
        assert "✓" in text


class TestDotExport:
    def test_valid_dot(self) -> None:
        graph = ConsensusGraph(run_id="r1")
        graph.nodes["a"] = ConsensusNode(alias="a", model_family="codex")
        graph.nodes["b"] = ConsensusNode(alias="b", model_family="gemini")
        graph.edges.append(ConsensusEdge("a", "b", "aligned", 0.9))
        dot = format_consensus_dot(graph)
        assert "digraph consensus" in dot
        assert '"a"' in dot
        assert '"b"' in dot
        assert "#34A853" in dot  # aligned edge color
