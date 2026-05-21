from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Static
from textual.containers import Container
from orchestra.storage.db import get_db

class FleetDashboard(App):
    CSS = """
    DataTable {
        height: 100%;
    }
    """
    BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="runs_table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Run ID", "Task", "Status", "Created", "Cost")
        self.refresh_data()

    def action_refresh(self) -> None:
        self.refresh_data()

    def refresh_data(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        
        conn = get_db()
        rows = conn.execute(
            "SELECT run_id, task, status, created_at, total_cost_usd FROM runs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        conn.close()
        
        for r in rows:
            cost = r["total_cost_usd"] or 0
            task = r["task"][:40] + "..." if len(r["task"]) > 40 else r["task"]
            table.add_row(r["run_id"], task, r["status"], r["created_at"], f"${cost:.3f}")
            
def launch_dashboard():
    app = FleetDashboard()
    app.run()
