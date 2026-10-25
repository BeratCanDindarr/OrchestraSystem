"""Rich status-table rendering for parallel runs."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from orchestra.models import AgentStatus, OrchestraRun

console = Console()


def _make_table(run: OrchestraRun) -> Table:
    table = Table(
        title=f"[bold cyan]run {run.run_id}[/bold cyan]  [dim]{run.mode}[/dim]",
        box=None,
        pad_edge=False,
        show_header=True,
    )
    table.add_column("AGENT", style="cyan", width=12)
    table.add_column("MODEL", style="white", width=22)
    table.add_column("STATE", width=11)
    table.add_column("ELAPSED", width=8)
    table.add_column("LAST", style="dim", no_wrap=True)

    colors = {
        AgentStatus.QUEUED: "yellow",
        AgentStatus.STARTED: "blue",
        AgentStatus.COMPLETED: "green",
        AgentStatus.FAILED: "red",
        AgentStatus.CANCELLED: "dim",
    }
    for agent in run.agents:
        color = colors.get(agent.status, "white")
        table.add_row(
            agent.alias,
            agent.model,
            f"[{color}]{agent.status.value}[/{color}]",
            agent.elapsed,
            agent.last_line,
        )
    return table

