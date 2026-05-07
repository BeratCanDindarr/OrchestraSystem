"""Agent tree widget for Orchestra TUI left pane."""
from rich.text import Text
from textual.widgets import Tree
from textual.widgets.tree import TreeNode


_STATUS_ICON: dict[str, tuple[str, str]] = {
    "queued":    ("○", "dim"),
    "pending":   ("○", "dim"),
    "started":   ("●", "cyan"),
    "running":   ("●", "cyan"),
    "completed": ("✓", "green"),
    "done":      ("✓", "green"),
    "failed":    ("✗", "red"),
    "cancelled": ("–", "dim"),
    "timeout":   ("⏱", "yellow"),
}


def _agent_label(alias: str, status: str, model: str, elapsed: str) -> Text:
    icon, color = _STATUS_ICON.get(status, ("?", "white"))
    return Text.assemble(
        (f"{alias:<12}", "bold"),
        (f"{icon} ", color),
        (f"{status:<10}", color),
        (elapsed, "dim"),
    )


class AgentTree(Tree):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("Orchestra", *args, **kwargs)
        self.root.expand()
        self._agent_nodes: dict[str, TreeNode] = {}

    def update_run(self, manifest: dict) -> None:
        run_id = manifest.get("run_id", "?")
        mode = manifest.get("mode", "?")
        self.root.set_label(f"run-{run_id[:8]} [{mode}]")

        agents = manifest.get("agents", [])
        for agent in agents:
            alias = agent.get("alias", "unknown")
            status = agent.get("status", "pending")
            model = agent.get("model", "")
            elapsed = agent.get("elapsed", "--:--")

            label = _agent_label(alias, status, model, elapsed)

            if alias in self._agent_nodes:
                self._agent_nodes[alias].set_label(label)
            else:
                node = self.root.add(label, data={"alias": alias, "model": model})
                self._agent_nodes[alias] = node
