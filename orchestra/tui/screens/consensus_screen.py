"""Modal screen showing the consensus graph with interactive node selection."""
from __future__ import annotations

from pathlib import Path

try:
    from textual.app import ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Static, Tree
    from textual.widgets.tree import TreeNode

    TEXTUAL_AVAILABLE = True
except ImportError:
    ComposeResult = object  # type: ignore[assignment, misc]
    Horizontal = Vertical = object  # type: ignore[assignment, misc]
    ModalScreen = object  # type: ignore[assignment, misc]
    Static = Tree = object  # type: ignore[assignment, misc]
    TreeNode = object  # type: ignore[assignment, misc]
    TEXTUAL_AVAILABLE = False


class ConsensusScreen(ModalScreen):
    """A modal that renders the consensus graph as an interactive node tree.

    Selecting a node reveals its metadata and reasoning chain (stdout log).
    """

    def __init__(self, graph, run_path: Path) -> None:
        super().__init__()
        self._graph = graph
        self._run_path = run_path

    def compose(self) -> ComposeResult:
        with Horizontal(id="consensus-container"):
            tree = Tree("Consensus Graph")
            tree.root.expand()
            for node in self._graph.nodes.values():
                label = f"{node.alias} ({node.model_family})"
                if node.is_red_team:
                    label += " [RED]"
                if node.is_synthesizer:
                    label += " [SYNTH]"
                tree.root.add(label, data={"alias": node.alias})
            yield tree
            with Vertical(id="consensus-details"):
                yield Static("Select a node to see reasoning chain.", id="node-meta")
                yield Static("", id="node-reasoning")

    def on_tree_node_selected(self, event) -> None:
        node = event.node
        if not node.data or "alias" not in node.data:
            return
        alias = node.data["alias"]
        consensus_node = self._graph.nodes.get(alias)
        if not consensus_node:
            return

        # Metadata panel
        lines = [
            f"Alias:      {consensus_node.alias}",
            f"Family:     {consensus_node.model_family}",
            f"Confidence: {consensus_node.confidence:.2f}",
            f"Status:     {consensus_node.status}",
            f"Digest:     {consensus_node.output_digest}",
        ]
        if consensus_node.is_red_team:
            lines.append("Role:       RED-TEAM")
        if consensus_node.is_synthesizer:
            lines.append("Role:       SYNTHESIZER")

        # Edges involving this node
        related = [e for e in self._graph.edges if e.source == alias or e.target == alias]
        if related:
            lines.append("\nEdges:")
            for edge in related:
                other = edge.target if edge.source == alias else edge.source
                symbol = "✓" if edge.kind == "aligned" else "~" if edge.kind == "partial" else "✗"
                lines.append(f"  {symbol} {other}  [{edge.kind}] {edge.similarity_score:.3f}")

        self.query_one("#node-meta").update("\n".join(lines))

        # Reasoning chain panel
        log_path = self._run_path / "agents" / f"{alias}.stdout.log"
        if log_path.exists():
            full_text = log_path.read_text(encoding="utf-8")
            reasoning = full_text[:4000]
            if len(full_text) > 4000:
                reasoning += "\n...[truncated]"
        else:
            reasoning = "(no reasoning log found)"
        self.query_one("#node-reasoning").update(reasoning)

    def on_key(self, event) -> None:
        if event.key in ("q", "escape", "g"):
            self.dismiss()
