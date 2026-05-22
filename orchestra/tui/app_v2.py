"""The Orchestrator's Console main Textual application."""
from __future__ import annotations

import shlex
import threading
import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

from rich.text import Text

from orchestra.engine.runners import artifacts as runner_artifacts
from orchestra.engine.runners.ask_runner import run_ask
from orchestra.engine.runners.critical_runner import run_critical
from orchestra.engine.runners.dual_runner import run_dual
from orchestra.engine.runners.planned_runner import run_planned
from orchestra.engine.runners.resume_runner import resume_run
from orchestra import config

logger = logging.getLogger(__name__)

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Input, Static, Header, Footer
    from textual import work

    from orchestra.tui.widgets.v2.stream import AgentStreamWidget
    from orchestra.tui.widgets.v2.monitor import MonitorWidget
    from orchestra.tui.widgets.v2.navigator import NavigatorWidget
    from orchestra.tui.widgets.v3 import SmartCommandBar, SmartSubmission
    from orchestra.tui.screens.diff_modal import DiffModal

    TEXTUAL_AVAILABLE = True
except ImportError:
    # Dummy classes for type hints and structural support
    class App: pass
    class Vertical: pass
    class Horizontal: pass
    class Input: 
        class Submitted: pass
    class Static: pass
    class Header: pass
    class Footer: pass
    def work(*args, **kwargs): return lambda x: x
    AgentStreamWidget = MonitorWidget = NavigatorWidget = SmartCommandBar = object
    TEXTUAL_AVAILABLE = False


RunCallable = Callable[..., Any]
RUNNER_ACTIONS: dict[str, RunCallable] = {
    "ask": run_ask,
    "dual": run_dual,
    "critical": run_critical,
    "planned": run_planned,
    "resume": resume_run,
}
_ACTIVE_AGENT_STATUSES = {"running", "started", "queued", "pending"}


class ArenaPane(Vertical):
    """Center pane that keeps up to two agent streams visible side by side."""

    def __init__(self, *, stream_count: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stream_count = max(1, min(2, stream_count))
        self.run_id: str | None = None
        self._manifest: dict[str, Any] | None = None
        self._manual_aliases: tuple[str, ...] | None = None

    def compose(self) -> ComposeResult:
        yield Static("Arena", classes="pane-title", id="arena-title")
        with Horizontal(id="arena-streams"):
            with Vertical(id="arena-panel-1", classes="arena-panel"):
                yield Static("No agent selected", id="arena-panel-title-1", classes="arena-panel-title")
                yield AgentStreamWidget(id="arena-stream-1")
            with Vertical(id="arena-panel-2", classes="arena-panel"):
                yield Static("Awaiting second agent", id="arena-panel-title-2", classes="arena-panel-title")
                yield AgentStreamWidget(id="arena-stream-2")
        yield Static(
            "Select a run in Navigator or launch one from the command bar.",
            id="arena-status",
            classes="arena-status",
        )

    def clear_run(self, message: str = "Select a run in Navigator or launch one from the command bar.") -> None:
        self.run_id = None
        self._manifest = None
        self._manual_aliases = None
        if self.is_mounted:
            self.query_one("#arena-title", Static).update("Arena")
            self.query_one("#arena-status", Static).update(message)
            self._show_placeholder(1, "No agent selected")
            self._show_placeholder(2, "Awaiting second agent")
            self.query_one("#arena-panel-2", Vertical).display = self.stream_count > 1

    def set_run(self, run_id: str, manifest: dict[str, Any], *, reset_aliases: bool = True) -> None:
        self.run_id = run_id
        self._manifest = manifest
        if reset_aliases:
            self._manual_aliases = None
        if self.is_mounted:
            self.update_manifest(manifest)

    def update_manifest(self, manifest: dict[str, Any]) -> None:
        self._manifest = manifest
        run_id = str(manifest.get("run_id") or self.run_id or "")
        if not run_id:
            self.clear_run()
            return
        self.run_id = run_id
        aliases = tuple(self._resolve_aliases(manifest))
        if not self.is_mounted:
            return

        status = str(manifest.get("status") or "--")
        mode = str(manifest.get("mode") or "--")
        cost = float(manifest.get("total_cost_usd") or 0.0)
        self.query_one("#arena-title", Static).update(f"Arena  {run_id[:12]}  {mode}  {status}")
        if aliases:
            displayed = ", ".join(aliases)
            self.query_one("#arena-status", Static).update(f"displaying {displayed}  cost ${cost:.4f}")
        else:
            self.query_one("#arena-status", Static).update(f"waiting for agent logs  cost ${cost:.4f}")
        self._sync_panels()

    def set_stream_count(self, count: int) -> int:
        self.stream_count = max(1, min(2, count))
        if self.is_mounted:
            self._sync_panels()
        return self.stream_count

    def set_manual_aliases(self, aliases: Sequence[str]) -> tuple[str, ...]:
        cleaned = tuple(alias for alias in aliases if alias)
        self._manual_aliases = cleaned or None
        if self.is_mounted:
            self._sync_panels()
        return cleaned

    def poll_logs(self) -> None:
        if not self.is_mounted:
            return
        for index in (1, 2):
            if index == 2 and self.stream_count == 1:
                continue
            if self.query_one(f"#arena-panel-{index}", Vertical).display:
                self.query_one(f"#arena-stream-{index}", AgentStreamWidget).poll()

    def _sync_panels(self) -> None:
        manifest = self._manifest or {}
        aliases = tuple(self._resolve_aliases(manifest))
        self._sync_panel(1, aliases[0] if aliases else None)
        show_second = self.stream_count > 1
        self.query_one("#arena-panel-2", Vertical).display = show_second
        if show_second:
            self._sync_panel(2, aliases[1] if len(aliases) > 1 else None)

    def _sync_panel(self, index: int, alias: str | None) -> None:
        panel = self.query_one(f"#arena-panel-{index}", Vertical)
        title = self.query_one(f"#arena-panel-title-{index}", Static)
        stream = self.query_one(f"#arena-stream-{index}", AgentStreamWidget)

        panel.display = True
        if alias is None:
            placeholder = "No matching agent" if self._manual_aliases else "Waiting for agent log"
            if index == 1 and self.run_id is None:
                placeholder = "No agent selected"
            if index == 2:
                placeholder = "Awaiting second agent"
            title.update(placeholder)
            stream.set_agent(None, None)
            stream.set_markdown(placeholder)
            return

        title.update(alias)
        stream.set_agent(alias, self._log_path(alias))

    def _resolve_aliases(self, manifest: dict[str, Any]) -> list[str]:
        agents = list(manifest.get("agents") or [])
        available = [str(agent.get("alias") or "") for agent in agents if agent.get("alias")]
        if not available:
            return []

        if self._manual_aliases:
            selected = [alias for alias in self._manual_aliases if alias in available]
            if selected:
                return selected[: self.stream_count]

        active: list[str] = []
        inactive: list[str] = []
        for agent in agents:
            alias = str(agent.get("alias") or "")
            if not alias: continue
            status = str(agent.get("status") or "").lower()
            bucket = active if status in _ACTIVE_AGENT_STATUSES else inactive
            if alias not in bucket:
                bucket.append(alias)
        return (active + inactive)[: self.stream_count]

    def _log_path(self, alias: str) -> Path:
        if self.run_id is None:
            raise RuntimeError("Cannot build a log path without a selected run")
        return runner_artifacts.run_dir(self.run_id) / "agents" / f"{alias}.stdout.log"

    def _show_placeholder(self, index: int, message: str) -> None:
        title = self.query_one(f"#arena-panel-title-{index}", Static)
        stream = self.query_one(f"#arena-stream-{index}", AgentStreamWidget)
        title.update(message)
        stream.set_agent(None, None)
        stream.set_markdown(message)


class OrchestraAppV2(App):
    """Three-column Orchestrator's Console application (Maestro Edition)."""

    CSS_PATH = "styles_v2.tcss"
    TITLE = "The Orchestrator's Console"
    SUB_TITLE = "TUI v3 (Maestro Edition)"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh_now", "Refresh", priority=True),
        Binding("tab", "cycle_focus", "Focus"),
        Binding("f", "toggle_focus", "Focus Mode"),
        Binding("h", "toggle_history", "History"),
    ] if TEXTUAL_AVAILABLE else []

    def __init__(self, run_id: str | None = None, interval: float = 1.0) -> None:
        super().__init__()
        self.initial_run_id = run_id
        self.interval = interval
        self.selected_run_id: str | None = None
        self._event_offsets: dict[str, int] = {}
        self._last_manifest: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-shell"):
            with Horizontal(id="main-layout"):
                yield NavigatorWidget(id="left-pane", classes="console-pane")
                yield ArenaPane(id="arena-pane", classes="console-pane")
                yield MonitorWidget(id="right-pane", classes="console-pane")
            yield SmartCommandBar(id="command-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(SmartCommandBar).focus_input()
        self.set_interval(self.interval, self._poll_selected_run)
        self.set_interval(max(self.interval * 10, 10.0), self._refresh_navigator)
        if self.initial_run_id:
            self._try_select_run(self.initial_run_id)

    def action_refresh_now(self) -> None:
        self._refresh_navigator()
        self._poll_selected_run()

    def action_cycle_focus(self) -> None:
        candidates = [
            self.query_one("#navigator-search", Input),
            self.query_one("#smart-bar-input", Input),
            self.query_one("#arena-stream-1", AgentStreamWidget),
        ]
        if self.query_one("#arena-panel-2", Vertical).display:
            candidates.append(self.query_one("#arena-stream-2", AgentStreamWidget))

        focus_index = 0
        for index, widget in enumerate(candidates):
            if widget.has_focus:
                focus_index = (index + 1) % len(candidates)
                break
        candidates[focus_index].focus()

    def action_toggle_focus(self) -> None:
        """Toggle focus mode (hide sidebars)."""
        left = self.query_one("#left-pane")
        right = self.query_one("#right-pane")
        if left.display:
            left.display = False
            right.display = False
            self.query_one("#arena-pane").styles.width = "1fr"
        else:
            left.display = True
            right.display = True
            self.query_one("#arena-pane").styles.width = "6fr"

    def action_toggle_history(self) -> None:
        """Toggle between Shells and History in the Navigator."""
        nav = self.query_one(NavigatorWidget)
        nav.nav_mode = "history" if nav.nav_mode == "models" else "models"

    def on_agent_stream_widget_tool_use_detected(self, message: Any) -> None:
        self.notify(f"Agent {message.agent_alias} using tool: {message.line[:40]}...", title="Skill Active")

    def on_monitor_widget_budget_exceeded(self, message: Any) -> None:
        self.notify(f"Budget exceeded! Cost: ${message.total_cost_usd:.2f}", 
                    title="Budget Warning", severity="error")

    def on_navigator_widget_model_selected(self, message: Any) -> None:
        alias = message.alias
        self.notify(f"Focused on shell: {alias}", title="Arena")
        self.query_one(ArenaPane).clear_run(message=f"Ready to run task with: {alias}")
        self.query_one(SmartCommandBar).value = f"/ask {alias} "
        self.query_one(SmartCommandBar).focus_input()

    def on_navigator_widget_run_selected(self, message: Any) -> None:
        self._select_run(message.run_id, message.manifest)

    def on_navigator_widget_worktree_selected(self, message: Any) -> None:
        self.notify(f"Switching to worktree: {message.name}", title="Navigator")

    def on_navigator_widget_playbook_selected(self, message: Any) -> None:
        self.notify(f"Starting playbook: {message.name}", title="Playbook")
        self._start_runner("playbook", self._run_playbook_logic, message.path)

    def on_smart_command_bar_submitted(self, message: SmartCommandBar.Submitted) -> None:
        submission = message.submission
        if submission.is_command:
            self._handle_command(submission.command, submission.args)
        else:
            self._start_runner("auto", self._run_auto_logic, submission.task)

    def _handle_command(self, cmd: str, args: tuple[str, ...]) -> None:
        if cmd == "open" and args:
            self._try_select_run(args[0])
        elif cmd == "diff" and len(args) >= 2:
            self.query_one(ArenaPane).set_manual_aliases(args[:2])
            self.notify(f"Diff view: {args[0]} vs {args[1]}", title="Arena")
        elif cmd == "playbook" and args:
            self._start_runner("playbook", self._run_playbook_logic, args[0])
        elif cmd == "ask" and len(args) >= 2:
            self._start_runner("ask", RUNNER_ACTIONS["ask"], args[0], " ".join(args[1:]))
        elif cmd in RUNNER_ACTIONS:
            self._start_runner(cmd, RUNNER_ACTIONS[cmd], " ".join(args))
        elif cmd == "help":
            self.notify("open | diff | playbook | ask | dual | critical | planned | resume", title="Commands")
        else:
            self.notify(f"Unknown command: {cmd}", severity="error")

    def _run_auto_logic(self, task: str) -> Any:
        from orchestra.router import task_to_mode
        routed_mode = task_to_mode(task)
        if routed_mode.startswith("ask "):
            return run_ask(routed_mode.split()[1], task)
        elif routed_mode == "dual":
            return run_dual(task)
        else:
            return run_critical(task)

    def _run_playbook_logic(self, path: str):
        from orchestra.engine.playbook import PlaybookRunner
        runner = PlaybookRunner()
        runner.run(path)

    def _start_runner(self, label: str, runner_fn: RunCallable, *args: Any, **kwargs: Any) -> None:
        self.notify(f"starting {label} run", title="Runner")
        self._run_runner_worker(label, runner_fn, args, kwargs)

    @work(thread=True, group="runner-launches", exit_on_error=False)
    def _run_runner_worker(self, label: str, runner_fn: RunCallable, args: tuple, kwargs: dict) -> None:
        try:
            run = runner_fn(*args, **kwargs)
            self.app.call_from_thread(self._handle_runner_result, label, run)
        except Exception as exc:
            self.app.call_from_thread(self._notify_runner_error, label, exc)

    def _handle_runner_result(self, label: str, run: Any) -> None:
        manifest = self._manifest_from_run(run)
        run_id = str(manifest.get("run_id") or getattr(run, "run_id", "") or "")
        if run_id:
            self._select_run(run_id, manifest)
            self.notify(f"{label} finished", title="Runner")

    def _notify_runner_error(self, label: str, exc: Exception) -> None:
        self.notify(f"{label} failed: {exc}", title="Runner", severity="error", timeout=8)

    @work(exclusive=True, thread=True)
    def _poll_selected_run(self) -> None:
        run_id = self.selected_run_id
        if not run_id:
            return
        manifest = runner_artifacts.load_manifest(run_id)
        if not manifest:
            return
            
        if manifest != self._last_manifest:
            self._last_manifest = manifest
            self.app.call_from_thread(self._update_selected_run, manifest)

        monitor = self.query_one(MonitorWidget)
        events = runner_artifacts.read_events(run_id)
        start = self._event_offsets.get(run_id, 0)
        if start < len(events):
            self.app.call_from_thread(monitor.add_events, events[start:])
            self._event_offsets[run_id] = len(events)
        
        self.app.call_from_thread(self.query_one(ArenaPane).poll_logs)

    @work(exclusive=True, thread=True)
    def _refresh_navigator(self) -> None:
        self.app.call_from_thread(self.query_one(NavigatorWidget).refresh_entries)

    def _try_select_run(self, run_id: str) -> None:
        self._try_select_run_worker(run_id)

    @work(thread=True, group="run-selection", exclusive=True, exit_on_error=False)
    def _try_select_run_worker(self, run_id: str) -> None:
        try:
            manifest = runner_artifacts.load_manifest(run_id)
            if manifest is None:
                self.app.call_from_thread(self.notify, f"run not found: {run_id}", title="Navigator", severity="warning")
            else:
                self.app.call_from_thread(self._select_run, run_id, manifest)
        except Exception:
            pass

    def _select_run(self, run_id: str, manifest: dict[str, Any]) -> None:
        self.selected_run_id = run_id
        self._event_offsets[run_id] = 0
        self._last_manifest = manifest
        self._update_selected_run(manifest, reset_aliases=True)

    def _update_selected_run(self, manifest: dict[str, Any], *, reset_aliases: bool = False) -> None:
        run_id = str(manifest.get("run_id") or self.selected_run_id or "")
        if not run_id:
            return
        self.selected_run_id = run_id

        arena = self.query_one(ArenaPane)
        if reset_aliases or arena.run_id != run_id:
            arena.set_run(run_id, manifest, reset_aliases=reset_aliases or arena.run_id != run_id)
        else:
            arena.update_manifest(manifest)

        monitor = self.query_one(MonitorWidget)
        if reset_aliases:
            monitor.clear_feed()
        monitor.update_from_manifest(manifest)

    @staticmethod
    def _manifest_from_run(run: Any) -> dict[str, Any]:
        if isinstance(run, dict): return run
        run_id = getattr(run, "run_id", None)
        if run_id:
            manifest = runner_artifacts.load_manifest(run_id)
            if manifest is not None: return manifest
        return {}


def run_tui_v2(run_id: str | None = None, interval: float = 1.0) -> Any:
    """Launch the Orchestrator's Console TUI."""
    if not TEXTUAL_AVAILABLE:
        print("Textual library is not available. Please install it with: pip install textual")
        return None
    return OrchestraAppV2(run_id=run_id, interval=interval).run()


if __name__ == "__main__":
    run_tui_v2()
