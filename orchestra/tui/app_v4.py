"""The Conversational Console (TUI v4) — Maestro-class chat interface."""
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

    from orchestra.tui.widgets.v4.chat import UnifiedChatWidget
    from orchestra.tui.widgets.v3 import SmartCommandBar, SmartSubmission
    from orchestra.tui.screens.diff_modal import DiffModal

    TEXTUAL_AVAILABLE = True
except ImportError:
    # Dummy classes
    class App: pass
    class Vertical: pass
    class Horizontal: pass
    class Input: pass
    class Static: pass
    class Header: pass
    class Footer: pass
    def work(*args, **kwargs): return lambda x: x
    UnifiedChatWidget = SmartCommandBar = object
    TEXTUAL_AVAILABLE = False


RunCallable = Callable[..., Any]
RUNNER_ACTIONS: dict[str, RunCallable] = {
    "ask": run_ask,
    "dual": run_dual,
    "critical": run_critical,
    "planned": run_planned,
    "resume": resume_run,
}

class OrchestraAppV4(App):
    """Unified Conversational Console application (TUI v4)."""

    CSS_PATH = "styles_v2.tcss"
    TITLE = "The Conversational Console"
    SUB_TITLE = "TUI v4 (Maestro Mode)"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("tab", "cycle_focus", "Focus"),
        Binding("?", "show_help", "Help"),
    ] if TEXTUAL_AVAILABLE else []

    def __init__(self, run_id: str | None = None, interval: float = 1.0) -> None:
        super().__init__()
        self.interval = interval
        self.last_agent_alias: str | None = None
        self._current_run_id: str | None = None
        self._poll_timer: Any = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield UnifiedChatWidget(id="chat-stream")
        yield SmartCommandBar(id="command-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(SmartCommandBar).focus_input()
        self.query_one(UnifiedChatWidget).add_system_message("Ready for your task. I'll automatically route to the best agent.")

    def action_show_help(self) -> None:
        self.notify("TAB: Cycle Focus | CTRL+C: Quit", title="Shortcuts")

    def action_cycle_focus(self) -> None:
        chat = self.query_one(UnifiedChatWidget)
        bar = self.query_one(SmartCommandBar)
        if bar.has_focus:
            chat.focus()
        else:
            bar.focus_input()

    
    def on_smart_command_bar_submitted(self, message: SmartCommandBar.Submitted) -> None:
        submission = message.submission
        chat = self.query_one("#chat-stream", UnifiedChatWidget)
        
        if submission.is_command:
            chat.add_system_message(f"Command: {submission.command} {' '.join(submission.args)}")
            self._handle_command(submission.command, submission.args)
        else:
            chat.add_user_message(submission.task)
            # 🛡️ Instant Feedback: Show placeholder immediately
            chat.start_agent_thinking("ROUTING...")
            self._route_and_start_task(submission.task)

    @work(thread=True, group="task-routing", exclusive=True)
    def _route_and_start_task(self, task: str) -> None:
        from orchestra.router import route_task
        try:
            # This can involve blocking IO (Ollama/Semantic search)
            decision = route_task(task, previous_alias=self.last_agent_alias)
            display_name = decision.alias or decision.mode.upper()
            
            def update_ui_and_run():
                chat = self.query_one("#chat-stream", UnifiedChatWidget)
                chat.start_agent_thinking(display_name)
                self._start_runner("auto", self._run_auto_logic, task)
            
            self.app.call_from_thread(update_ui_and_run)
        except Exception as exc:
            self.app.call_from_thread(self.notify, f"Routing failed: {exc}", severity="error")

            self._start_runner("auto", self._run_auto_logic, submission.task)

    def _handle_command(self, cmd: str, args: tuple[str, ...]) -> None:
        if cmd == "open" and args:
            self._load_past_run(args[0])
        elif cmd == "diff" and len(args) >= 2:
            self.notify("Use TUI v3 for side-by-side diff. Modal integration coming soon.", severity="warning")
        elif cmd == "ask" and len(args) >= 2:
            self._start_runner("ask", RUNNER_ACTIONS["ask"], args[0], " ".join(args[1:]))
        elif cmd in RUNNER_ACTIONS:
            self._start_runner(cmd, RUNNER_ACTIONS[cmd], " ".join(args))
        elif cmd == "help":
            self.notify("open | ask | dual | critical | planned | resume", title="Commands")
        else:
            self.notify(f"Unknown command: {cmd}", severity="error")

    def _run_auto_logic(self, task: str) -> Any:
        from orchestra.service import run_task
        # Pass previous_alias for context continuity (stickiness)
        result = run_task(
            mode="auto",
            task=task,
            previous_alias=self.last_agent_alias
        )
        return result

    def _start_runner(self, label: str, runner_fn: RunCallable, *args: Any, **kwargs: Any) -> None:
        self._run_runner_worker(label, runner_fn, args, kwargs)

    @work(thread=True, group="runner-launches", exit_on_error=False)
    def _run_runner_worker(self, label: str, runner_fn: RunCallable, args: tuple, kwargs: dict) -> None:
        try:
            # We don't know the alias yet if it's auto-routed, so we'll catch it in the results
            self.app.call_from_thread(self._on_run_started, label)
            result = runner_fn(*args, **kwargs)
            self.app.call_from_thread(self._handle_runner_result, label, result)
        except Exception as exc:
            self.app.call_from_thread(self._notify_runner_error, label, exc)

    def _on_run_started(self, label: str) -> None:
        # Placeholder while routing
        pass

    def _handle_runner_result(self, label: str, result: Any) -> None:
        manifest = result["run"] if isinstance(result, dict) and "run" in result else getattr(result, "to_manifest", lambda: {})()
        run_id = manifest.get("run_id")
        if not run_id: return

        self._current_run_id = run_id
        
        # Determine which agent was active
        agents = manifest.get("agents", [])
        if agents:
            # For v4, we focus on the main agent response
            active_agent = agents[0].get("alias")
            self.last_agent_alias = active_agent
            
            chat = self.query_one(UnifiedChatWidget)
            # If it's a handoff, mention it
            routed_mode = result.get("routed_mode")
            if routed_mode:
                chat.add_system_message(f"Routed to {active_agent} ({routed_mode})")
            
            chat.start_agent_thinking(active_agent)
            
            # Start polling logs
            if self._poll_timer:
                self._poll_timer.stop()
            self._poll_timer = self.set_interval(self.interval, self._poll_logs)

    def _notify_runner_error(self, label: str, exc: Exception) -> None:
        self.query_one(UnifiedChatWidget).add_system_message(f"Error: {exc}")

    @work(exclusive=True, thread=True)
    def _poll_logs(self) -> None:
        run_id = self._current_run_id
        alias = self.last_agent_alias
        if not run_id or not alias: return
        
        # Simple tail of the log
        log_path = runner_artifacts.run_dir(run_id) / "agents" / f"{alias}.stdout.log"
        if not log_path.exists(): return
        
        # We use the same tailing logic but simplified for unified chat
        # For brevity, I'll use a class-level offset
        if not hasattr(self, "_log_offsets"): self._log_offsets = {}
        offset = self._log_offsets.get(run_id, 0)
        
        size = log_path.stat().st_size
        if size > offset:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                chunk = f.read(16384)
                self._log_offsets[run_id] = f.tell()
                self.app.call_from_thread(self._append_agent_chunk, chunk)
        
        # Check if run is finished
        manifest = runner_artifacts.load_manifest(run_id)
        if manifest and manifest.get("status") in {"completed", "failed", "cancelled"}:
            if self._poll_timer:
                self.app.call_from_thread(self._finish_agent_response)

    def _append_agent_chunk(self, chunk: str) -> None:
        self.query_one(UnifiedChatWidget).update_agent_message(chunk)

    def _finish_agent_response(self) -> None:
        if self._poll_timer:
            self._poll_timer.stop()
            self._poll_timer = None
        self.query_one(UnifiedChatWidget).update_agent_message("", final=True)

    def _load_past_run(self, run_id: str) -> None:
        manifest = runner_artifacts.load_manifest(run_id)
        if not manifest:
            self.notify(f"Run {run_id} not found.", severity="error")
            return
        
        chat = self.query_one(UnifiedChatWidget)
        chat.add_system_message(f"Loading history for run {run_id}...")
        
        # Show the task
        chat.add_user_message(manifest.get("task", "(empty task)"))
        
        # Show agent logs
        for agent in manifest.get("agents", []):
            alias = agent.get("alias")
            log = runner_artifacts.read_agent_log(run_id, alias)
            if log:
                chat.start_agent_thinking(alias)
                chat.update_agent_message(log, final=True)

def run_tui_v4(run_id: str | None = None, interval: float = 1.0) -> Any:
    """Launch the Conversational Console TUI."""
    if not TEXTUAL_AVAILABLE:
        print("Textual library is not available.")
        return None
    return OrchestraAppV4(run_id=run_id, interval=interval).run()

if __name__ == "__main__":
    run_tui_v4()
