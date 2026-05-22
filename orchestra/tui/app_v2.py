"""The Orchestrator's Console main Textual application."""
from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Sequence

from rich.text import Text

from orchestra import config
from orchestra.engine import artifacts as runner_artifacts

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, Static

    from orchestra.tui.widgets.v2 import AgentStreamWidget, MonitorWidget, NavigatorWidget

    TEXTUAL_AVAILABLE = True
except ImportError:
    def work(*args: Any, **kwargs: Any):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(method: Any) -> Any:
            return method

        return decorator

    class App:  # type: ignore[no-redef]
        pass

    class Vertical:  # type: ignore[no-redef]
        pass

    class Horizontal:  # type: ignore[no-redef]
        pass

    class Input:  # type: ignore[no-redef]
        class Submitted:
            pass

    class Static:  # type: ignore[no-redef]
        pass

    class Header:  # type: ignore[no-redef]
        pass

    class Footer:  # type: ignore[no-redef]
        pass

    class AgentStreamWidget:
        pass

    class MonitorWidget:
        pass

    class NavigatorWidget:
        pass
    TEXTUAL_AVAILABLE = False


RunCallable = Callable[..., Any]
_ACTIVE_AGENT_STATUSES = {"running", "started", "queued", "pending"}
_TOOL_NOTICE_DEBOUNCE_SECONDS = 1.0


def _lazy_runner(module_name: str, func_name: str) -> RunCallable:
    def _runner(*args: Any, **kwargs: Any) -> Any:
        module = import_module(module_name)
        return getattr(module, func_name)(*args, **kwargs)

    return _runner


RUNNER_ACTIONS: dict[str, RunCallable] = {
    "ask": _lazy_runner("orchestra.engine.runners.ask_runner", "run_ask"),
    "dual": _lazy_runner("orchestra.engine.runners.dual_runner", "run_dual"),
    "critical": _lazy_runner("orchestra.engine.runners.critical_runner", "run_critical"),
    "planned": _lazy_runner("orchestra.engine.runners.planned_runner", "run_planned"),
    "resume": _lazy_runner("orchestra.engine.runners.resume_runner", "resume_run"),
}


@dataclass(frozen=True)
class EventCursor:
    source: str = ""
    position: int = 0
    count: int = 0


class CommandBar(Input):
    """Bottom command bar for TUI actions and run launches."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            placeholder=(
                "Commands: open <run_id>, playbook <path>, ask <alias> <prompt>, dual <prompt>, "
                "critical <prompt>, planned <prompt>, resume <run_id> [approve|reject]"
            ),
            **kwargs,
        )


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
        if not self.is_mounted:
            return
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
            self.query_one("#arena-status", Static).update(
                f"displaying {', '.join(aliases)}  cost ${cost:.4f}"
            )
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
        self._update_status_line(manifest, aliases)
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
            stream.clear()
            stream.write(Text(placeholder, style="dim"))
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
            if not alias:
                continue
            status = str(agent.get("status") or "").lower()
            bucket = active if status in _ACTIVE_AGENT_STATUSES else inactive
            if alias not in bucket:
                bucket.append(alias)
        return (active + inactive)[: self.stream_count]

    def _log_path(self, alias: str) -> Path:
        if self.run_id is None:
            raise RuntimeError("Cannot build a log path without a selected run")
        return config.artifact_root() / self.run_id / "agents" / f"{alias}.stdout.log"

    def _show_placeholder(self, index: int, message: str) -> None:
        title = self.query_one(f"#arena-panel-title-{index}", Static)
        stream = self.query_one(f"#arena-stream-{index}", AgentStreamWidget)
        title.update(message)
        stream.set_agent(None, None)
        stream.clear()
        stream.write(Text(message, style="dim"))

    def _update_status_line(self, manifest: dict[str, Any], aliases: Sequence[str]) -> None:
        if not self.is_mounted:
            return
        cost = float(manifest.get("total_cost_usd") or 0.0)
        if aliases:
            self.query_one("#arena-status", Static).update(
                f"displaying {', '.join(aliases)}  cost ${cost:.4f}"
            )
        else:
            self.query_one("#arena-status", Static).update(f"waiting for agent logs  cost ${cost:.4f}")


class OrchestraAppV2(App):
    """Three-column Orchestrator's Console application."""

    CSS_PATH = "styles_v2.tcss"
    TITLE = "The Orchestrator's Console"
    SUB_TITLE = "TUI v2"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh_now", "Refresh", priority=True),
        Binding("tab", "cycle_focus", "Focus"),
    ] if TEXTUAL_AVAILABLE else []

    def __init__(self, run_id: str | None = None, interval: float = 1.0) -> None:
        super().__init__()
        self.initial_run_id = run_id
        self.interval = interval
        self.selected_run_id: str | None = None
        self._event_cursors: dict[str, EventCursor] = {}
        self._run_refresh_inflight = False
        self._run_refresh_queued = False
        self._last_manifest: dict[str, Any] | None = None
        self._last_tool_notice_at = 0.0
        self._last_tool_notice_key: tuple[str | None, str] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="app-shell"):
            with Horizontal(id="main-layout"):
                yield NavigatorWidget(id="left-pane", classes="console-pane")
                yield ArenaPane(id="arena-pane", classes="console-pane")
                yield MonitorWidget(id="right-pane", classes="console-pane")
            yield CommandBar(id="command-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#command-bar", Input).focus()
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
            self.query_one("#command-bar", Input),
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

    def on_agent_stream_widget_tool_use_detected(self, message: Any) -> None:
        snippet = message.line[:40]
        key = (message.agent_alias, snippet)
        now = time.monotonic()
        if key == self._last_tool_notice_key and now - self._last_tool_notice_at < _TOOL_NOTICE_DEBOUNCE_SECONDS:
            return
        if now - self._last_tool_notice_at < _TOOL_NOTICE_DEBOUNCE_SECONDS / 2:
            return
        self._last_tool_notice_key = key
        self._last_tool_notice_at = now
        self.notify(f"Agent {message.agent_alias} is using a tool: {snippet}...", title="Skill Active")

    def on_agent_stream_widget_stream_line_appended(self, message: Any) -> None:
        pass

    def on_monitor_widget_budget_exceeded(self, message: Any) -> None:
        self.notify(
            f"Budget exceeded! Cost: ${message.total_cost_usd:.2f} / Limit: ${message.budget_usd:.2f}",
            title="Budget Warning",
            severity="error",
        )

    def on_navigator_widget_run_selected(self, message: Any) -> None:
        self._select_run(message.run_id, message.manifest)

    def on_navigator_widget_worktree_selected(self, message: Any) -> None:
        detail = str(message.path) if message.path is not None else "path unavailable"
        self.notify(f"worktree {message.name}: {detail}", title="Navigator")

    def on_navigator_widget_playbook_selected(self, message: Any) -> None:
        self.notify(f"Starting playbook: {message.path}", title="Playbook")
        self._start_runner("playbook", self._run_playbook_logic, message.path)

    def on_navigator_widget_entries_refreshed(self, message: Any) -> None:
        if self.selected_run_id:
            return
        target_run = self.initial_run_id
        for entry in message.entries:
            if entry.kind != "run" or entry.payload is None:
                continue
            if target_run and entry.key != target_run:
                continue
            self._select_run(entry.key, entry.payload)
            break

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command-bar":
            return
        raw = event.value.strip()
        event.input.value = ""
        if raw:
            self._dispatch_command(raw)

    def _dispatch_command(self, raw: str) -> None:
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self.notify(str(exc), title="Command Error", severity="error")
            return
        if not parts:
            return

        command = parts[0].lower()
        if command == "help":
            self.notify(
                "open <run_id> | playbook <path> | ask <alias> <prompt> | dual <prompt> | critical <prompt>",
                title="Commands",
                timeout=8,
            )
            return
        if command == "open" and len(parts) == 2:
            self._try_select_run(parts[1])
            return
        if command == "playbook" and len(parts) == 2:
            self._start_runner("playbook", self._run_playbook_logic, parts[1])
            return
        if command == "ask":
            if len(parts) < 3:
                self.notify("ask expects <alias> <prompt>", title="Command Error", severity="error")
                return
            self._start_runner("ask", RUNNER_ACTIONS["ask"], parts[1], " ".join(parts[2:]))
            return
        if command in {"dual", "critical", "planned"}:
            if len(parts) < 2:
                self.notify(f"{command} expects a prompt", title="Command Error", severity="error")
                return
            self._start_runner(command, RUNNER_ACTIONS[command], " ".join(parts[1:]))
            return
        if command == "resume":
            decision = parts[2] if len(parts) == 3 else "approve"
            self._start_runner("resume", RUNNER_ACTIONS["resume"], parts[1], approval_decision=decision)
            return
        self.notify(f"unknown command: {command}", title="Command Error", severity="error")

    def _run_playbook_logic(self, path: str) -> None:
        from orchestra.engine.playbook import PlaybookRunner

        PlaybookRunner().run(path)

    def _start_runner(self, label: str, runner_fn: RunCallable, *args: Any, **kwargs: Any) -> None:
        self.notify(f"starting {label} run", title="Runner")
        self._run_runner_worker(label, runner_fn, args, kwargs)

    @work(thread=True, group="runner-launches", exit_on_error=False)
    def _run_runner_worker(
        self,
        label: str,
        runner_fn: RunCallable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        try:
            run = runner_fn(*args, **kwargs)
            self.call_from_thread(self._handle_runner_result, label, run)
        except Exception as exc:
            self.call_from_thread(self._notify_runner_error, label, exc)

    def _handle_runner_result(self, label: str, run: Any) -> None:
        manifest = self._manifest_from_run(run)
        run_id = str(manifest.get("run_id") or getattr(run, "run_id", "") or "")
        if not run_id:
            return
        if manifest:
            self._select_run(run_id, manifest)
        else:
            self._try_select_run(run_id)
        self.notify(f"{label} finished", title="Runner")

    def _notify_runner_error(self, label: str, exc: Exception) -> None:
        self.notify(f"{label} failed: {exc}", title="Runner", severity="error", timeout=8)

    @work(exclusive=True, thread=True)
    def _poll_selected_run(self) -> None:
        run_id = self.selected_run_id
        if not run_id:
            return
        if self._run_refresh_inflight:
            self._run_refresh_queued = True
            return
        self._run_refresh_inflight = True
        self._poll_selected_run_worker(run_id, self._event_cursors.get(run_id, EventCursor()))

    @work(thread=True, group="selected-run-refresh", exit_on_error=False)
    def _poll_selected_run_worker(self, run_id: str, cursor: EventCursor) -> None:
        try:
            manifest = runner_artifacts.load_manifest(run_id)
            next_cursor, events = self._read_event_delta(run_id, cursor)
        except Exception:
            manifest = None
            next_cursor, events = cursor, []
        self.call_from_thread(self._complete_selected_run_poll, run_id, manifest, next_cursor, events)

    def _complete_selected_run_poll(
        self,
        run_id: str,
        manifest: dict[str, Any] | None,
        cursor: EventCursor,
        events: list[dict[str, Any]],
    ) -> None:
        self._run_refresh_inflight = False
        if run_id == self.selected_run_id:
            if manifest:
                self._event_cursors[run_id] = cursor
                if manifest != self._last_manifest:
                    self._last_manifest = manifest
                    self._update_selected_run(manifest)
                else:
                    self.query_one(ArenaPane).poll_logs()
                if events:
                    self.query_one(MonitorWidget).add_events(events)
            if self._run_refresh_queued:
                self._run_refresh_queued = False
                self._poll_selected_run()
                return
        self._run_refresh_queued = False

    @work(exclusive=True, thread=True)
    def _refresh_navigator(self) -> None:
        self.query_one(NavigatorWidget).refresh_entries()

    def _try_select_run(self, run_id: str) -> None:
        self._try_select_run_worker(run_id)

    @work(thread=True, group="run-selection", exclusive=True, exit_on_error=False)
    def _try_select_run_worker(self, run_id: str) -> None:
        try:
            manifest = runner_artifacts.load_manifest(run_id)
        except Exception:
            manifest = None
        self.call_from_thread(self._complete_try_select_run, run_id, manifest)

    def _complete_try_select_run(self, run_id: str, manifest: dict[str, Any] | None) -> None:
        if manifest is None:
            self.notify(f"run not found: {run_id}", title="Navigator", severity="warning")
            return
        self._select_run(run_id, manifest)

    def _select_run(self, run_id: str, manifest: dict[str, Any]) -> None:
        self.selected_run_id = run_id
        self._event_cursors[run_id] = EventCursor()
        self._last_manifest = manifest
        self._update_selected_run(manifest, reset_aliases=True)
        self._poll_selected_run()

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
        arena.poll_logs()

        monitor = self.query_one(MonitorWidget)
        if reset_aliases:
            monitor.clear_feed()
        monitor.update_from_manifest(manifest)

    @staticmethod
    def _manifest_from_run(run: Any) -> dict[str, Any]:
        if isinstance(run, dict):
            return run
        return {}

    @staticmethod
    def _read_event_delta(run_id: str, cursor: EventCursor) -> tuple[EventCursor, list[dict[str, Any]]]:
        run_path = config.artifact_root() / run_id
        live_path = run_path / "events.jsonl"
        gzip_path = run_path / "events.jsonl.gz"

        if live_path.exists():
            return OrchestraAppV2._read_live_event_delta(live_path, cursor)
        if gzip_path.exists():
            events = runner_artifacts.read_events(run_id)
            start = cursor.count if cursor.source == "events.jsonl.gz" else 0
            return EventCursor("events.jsonl.gz", 0, len(events)), events[start:]
        return EventCursor(), []

    @staticmethod
    def _read_live_event_delta(path: Path, cursor: EventCursor) -> tuple[EventCursor, list[dict[str, Any]]]:
        start = cursor.position if cursor.source == "events.jsonl" else 0
        try:
            size = path.stat().st_size
            if size < start:
                start = 0
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(start)
                lines = handle.readlines()
                end = handle.tell()
        except OSError:
            return cursor, []

        events: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return EventCursor("events.jsonl", end, cursor.count + len(events)), events


def run_tui_v2(run_id: str | None = None, interval: float = 1.0) -> Any:
    """Launch the Orchestrator's Console TUI."""
    if not TEXTUAL_AVAILABLE:
        print("Textual library is not available. Please install it with: pip install textual")
        return None
    return OrchestraAppV2(run_id=run_id, interval=interval).run()


if __name__ == "__main__":
    run_tui_v2()
