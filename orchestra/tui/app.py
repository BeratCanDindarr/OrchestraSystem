"""Split-pane Textual TUI for Orchestra runs (claude-esp style)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from orchestra.engine.artifacts import run_dir

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer

    from orchestra.tui.widgets.agent_tree import AgentTree
    from orchestra.tui.widgets.header_bar import HeaderBar
    from orchestra.tui.widgets.stream_view import StreamView
    from orchestra.tui.screens.consensus_screen import ConsensusScreen

    TEXTUAL_AVAILABLE = True
except ImportError:
    App = object  # type: ignore[assignment]
    ComposeResult = Binding = Horizontal = Vertical = Footer = object  # type: ignore[assignment]
    AgentTree = HeaderBar = StreamView = object  # type: ignore[assignment]
    TEXTUAL_AVAILABLE = False


def _format_elapsed(start: str | None, end: str | None = None) -> str:
    if not start:
        return "--:--"
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
        diff = max(0, int((end_dt - start_dt).total_seconds()))
        return f"{diff // 60:02d}:{diff % 60:02d}"
    except ValueError:
        return "--:--"


def _read_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


class OrchestraApp(App):
    """Split-pane TUI: agent tree (30%) + live log stream (70%)."""

    CSS_PATH = "styles.tcss"
    TITLE = "Orchestra"

    BINDINGS = [
        Binding("t", "toggle_thinking", "Thinking", priority=True),
        Binding("i", "toggle_tool", "Tools", priority=True),
        Binding("o", "toggle_output", "Output", priority=True),
        Binding("j", "scroll_down", "↓", show=False),
        Binding("k", "scroll_up", "↑", show=False),
        Binding("tab", "switch_focus", "Switch Pane"),
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_now", "Refresh"),
        Binding("g", "show_consensus", "Graph"),
    ] if TEXTUAL_AVAILABLE else []

    def __init__(self, run_id: str, interval: float = 1.0) -> None:
        super().__init__()
        self.run_id = run_id
        self.interval = interval
        self._run_path = run_dir(run_id)
        self._manifest_path = self._run_path / "manifest.json"
        self._current_agent: str | None = None

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header")
        with Horizontal(id="main-container"):
            yield AgentTree(id="left-pane")
            with Vertical(id="right-pane"):
                yield StreamView(id="stream-view")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#left-pane").focus()
        self.set_interval(1.0, self._poll_manifest)
        self.set_interval(0.5, self._poll_logs)

    # ── Filter actions ────────────────────────────────────────────────────────

    def action_toggle_thinking(self) -> None:
        header = self.query_one(HeaderBar)
        header.t_active = not header.t_active
        self._apply_filters()

    def action_toggle_tool(self) -> None:
        header = self.query_one(HeaderBar)
        header.i_active = not header.i_active
        self._apply_filters()

    def action_toggle_output(self) -> None:
        header = self.query_one(HeaderBar)
        header.o_active = not header.o_active
        self._apply_filters()

    def _apply_filters(self) -> None:
        h = self.query_one(HeaderBar)
        self.query_one(StreamView).set_filters(h.t_active, h.i_active, h.o_active)

    # ── Navigation actions ────────────────────────────────────────────────────

    def action_switch_focus(self) -> None:
        left = self.query_one("#left-pane")
        if left.has_focus:
            self.query_one("#stream-view").focus()
        else:
            left.focus()

    def action_scroll_down(self) -> None:
        self.query_one(StreamView).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one(StreamView).scroll_up()

    def action_refresh_now(self) -> None:
        self._poll_manifest()
        self._poll_logs()

    def action_show_consensus(self) -> None:
        manifest = _read_manifest(self._manifest_path)
        graph_dict = manifest.get("consensus_graph") if manifest else None
        if not graph_dict:
            from textual.widgets import Static
            self.push_screen(ConsensusScreen(ConsensusGraph(), self._run_path))
            return
        from orchestra.engine.consensus import format_consensus_ascii
        from orchestra.engine.consensus_graph import ConsensusGraph, ConsensusNode, ConsensusEdge
        graph = ConsensusGraph(
            run_id=graph_dict.get("run_id", ""),
            consensus_level=graph_dict.get("consensus_level", "split"),
        )
        for n in graph_dict.get("nodes", []):
            graph.nodes[n["alias"]] = ConsensusNode(
                alias=n["alias"],
                model_family=n.get("model_family", ""),
                output_digest=n.get("output_digest", ""),
                confidence=n.get("confidence", 0.0),
                status=n.get("status", ""),
                is_red_team=n.get("is_red_team", False),
                is_synthesizer=n.get("is_synthesizer", False),
            )
        for e in graph_dict.get("edges", []):
            graph.edges.append(ConsensusEdge(
                source=e["source"],
                target=e["target"],
                kind=e["kind"],
                similarity_score=e.get("similarity_score", 0.0),
            ))
        self.push_screen(ConsensusScreen(graph, self._run_path))

    # ── Tree selection ────────────────────────────────────────────────────────

    def on_tree_node_selected(self, event) -> None:
        node = event.node
        if node.data and "alias" in node.data:
            alias = node.data["alias"]
            if alias != self._current_agent:
                self._current_agent = alias
                log_path = self._run_path / "agents" / f"{alias}.stdout.log"
                self.query_one(StreamView).set_log_path(log_path)

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll_manifest(self) -> None:
        manifest = _read_manifest(self._manifest_path)
        if not manifest:
            return

        header = self.query_one(HeaderBar)
        header.run_id = manifest.get("run_id", "?")[:8]
        header.mode = manifest.get("mode", "--")
        status = manifest.get("status", "?")
        created = manifest.get("created_at")
        finished = manifest.get("updated_at") if status in {"completed", "failed", "cancelled"} else None
        header.elapsed = _format_elapsed(created, finished)
        cost = manifest.get("total_cost_usd", 0.0) or 0.0
        header.cost = f"${cost:.4f}"

        self.query_one(AgentTree).update_run(manifest)

        # Auto-select first running agent if none selected
        if self._current_agent is None:
            for agent in manifest.get("agents", []):
                if agent.get("status") in {"running", "started"}:
                    alias = agent.get("alias")
                    if alias:
                        self._current_agent = alias
                        log_path = self._run_path / "agents" / f"{alias}.stdout.log"
                        self.query_one(StreamView).set_log_path(log_path)
                        break

    def _poll_logs(self) -> None:
        self.query_one(StreamView).poll()


def run_tui(run_id: str, interval: float = 1.0):
    """Launch the split-pane Orchestra TUI."""
    if not TEXTUAL_AVAILABLE:
        return None
    return OrchestraApp(run_id=run_id, interval=interval).run()
