"""Session monitor widget for Orchestra TUI v2."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, List

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static


class MonitorWidget(Vertical):
    """Right column cost tracker and recent Cue automation feed."""

    DEFAULT_CSS = """
    MonitorWidget {
        height: 100%;
    }
    #cue-feed {
        height: 1fr;
        overflow-y: scroll;
    }
    """

    run_id: reactive[str] = reactive("")
    run_status: reactive[str] = reactive("idle")
    total_cost_usd: reactive[float] = reactive(0.0)
    active_agents: reactive[int] = reactive(0)
    completed_agents: reactive[int] = reactive(0)
    failed_agents: reactive[int] = reactive(0)
    budget_usd: reactive[float | None] = reactive(None)

    class BudgetExceeded(Message):
        def __init__(self, total_cost_usd: float, budget_usd: float) -> None:
            super().__init__()
            self.total_cost_usd = total_cost_usd
            self.budget_usd = budget_usd

    def __init__(self, *, max_events: int = 20, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_events = max_events
        self._cue_events: Deque[dict[str, Any]] = deque(maxlen=max_events)

    def compose(self) -> ComposeResult:
        yield Static("Monitor", classes="pane-title")
        yield Static("$0.0000", id="total-cost", classes="stat-value")
        yield Static("idle  0 active  0 done  0 failed", id="fleet-stats", classes="stat-label")
        yield Static("", id="monitor-budget", classes="stat-label")
        yield Static("Cue Events", classes="pane-title")
        yield Static("no cue events", id="cue-feed")

    def watch_total_cost_usd(self, value: float) -> None:
        if not self.is_mounted: return
        self.query_one("#total-cost", Static).update(f"${value:.4f}")
        if self.budget_usd is not None and value > self.budget_usd:
            self.post_message(self.BudgetExceeded(value, self.budget_usd))
        self._update_budget()

    def watch_active_agents(self, _value: int) -> None: self._update_agents()
    def watch_completed_agents(self, _value: int) -> None: self._update_agents()
    def watch_failed_agents(self, _value: int) -> None: self._update_agents()
    def watch_run_status(self, _value: str) -> None: self._update_agents()
    def watch_budget_usd(self, _value: float | None) -> None: self._update_budget()

    def update_from_manifest(self, manifest: dict[str, Any]) -> None:
        """Update cost and agent counters from a run manifest."""
        self.run_id = str(manifest.get("run_id") or "")
        self.run_status = str(manifest.get("status") or "unknown")
        agents = list(manifest.get("agents") or [])
        
        manifest_cost = manifest.get("total_cost_usd")
        if manifest_cost is None:
            manifest_cost = sum(float(agent.get("estimated_cost_usd") or 0.0) for agent in agents)
        
        self.total_cost_usd = float(manifest_cost or 0.0)
        self.active_agents = sum(1 for a in agents if str(a.get("status") or "").lower() in {"running", "started", "queued", "pending"})
        self.completed_agents = sum(1 for a in agents if str(a.get("status") or "").lower() in {"completed", "done", "success"})
        self.failed_agents = sum(1 for a in agents if str(a.get("status") or "").lower() in {"failed", "cancelled"})
        
        budget = manifest.get("budget_usd") or manifest.get("daily_budget_usd")
        if budget is not None:
            self.budget_usd = float(budget)

    def clear_feed(self) -> None:
        """Clear the Cue mini-feed for a newly selected run."""
        self._cue_events.clear()
        if self.is_mounted:
            self._render_feed()

    def add_events(self, events: List[dict[str, Any]]) -> None:
        """Batch add multiple automation events."""
        changed = False
        for event in events:
            event_type = str(event.get("event") or event.get("event_type") or "")
            source = str(event.get("source", "")).lower()
            if "cue" in event_type.lower() or "cue" in source or event_type in ("run_started", "agent_started", "checkpoint_written"):
                self._cue_events.appendleft(event)
                changed = True
        
        if changed and self.is_mounted:
            self._render_feed()

    def _update_agents(self) -> None:
        if self.is_mounted:
            self.query_one("#fleet-stats", Static).update(
                f"{self.run_status}  "
                f"{self.active_agents} active  "
                f"{self.completed_agents} done  "
                f"{self.failed_agents} failed"
            )

    def _update_budget(self) -> None:
        if not self.is_mounted: return
        if self.budget_usd is None:
            text = ""
        else:
            remaining = self.budget_usd - self.total_cost_usd
            text = f"budget ${self.budget_usd:.2f}  rem ${remaining:.4f}"
        self.query_one("#monitor-budget", Static).update(text)

    def _render_feed(self) -> None:
        lines: list[Text] = []
        for event in self._cue_events:
            ts = str(event.get("ts") or event.get("timestamp") or "")
            stamp = ts[11:19] if len(ts) >= 19 else "now"
            name = str(event.get("rule") or event.get("event") or event.get("event_type") or "event")
            
            style = "yellow"
            if "fail" in name.lower() or "error" in name.lower(): style = "red"
            elif "success" in name.lower() or "finish" in name.lower(): style = "green"
            
            lines.append(Text.assemble((f"{stamp} ", "dim"), (f"{name:<15}", style)))
            
        self.query_one("#cue-feed", Static).update(Text("\n").join(lines) if lines else "no events")
