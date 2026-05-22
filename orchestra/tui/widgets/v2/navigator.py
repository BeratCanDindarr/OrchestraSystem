"""Navigator widget for Orchestra TUI v2.

Left column widget for selecting historical runs, active Orchestra git
worktrees, and automated playbooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual import work
from textual.widgets import Input, ListItem, ListView, Static

from orchestra.engine import artifacts
from orchestra.engine.worktree_manager import WorktreeManager
from orchestra import config


@dataclass(frozen=True)
class NavigatorEntry:
    kind: str
    key: str
    label: str
    detail: str = ""
    path: Path | None = None
    payload: dict[str, Any] | None = None


class NavigatorListItem(ListItem):
    """List item carrying run/worktree/playbook metadata."""

    def __init__(self, entry: NavigatorEntry) -> None:
        super().__init__(Static(_entry_text(entry)))
        self.entry = entry


def _entry_text(entry: NavigatorEntry) -> Text:
    icons = {"run": "◉", "worktree": "⌁", "playbook": "📝", "status": "ℹ"}
    styles = {"run": "cyan", "worktree": "magenta", "playbook": "green", "status": "dim"}
    
    icon = icons.get(entry.kind, "•")
    style = styles.get(entry.kind, "white")
    return Text.assemble((f"{icon} ", style), (entry.label, "bold"), ("  " + entry.detail, "dim"))


def _format_run(manifest: dict[str, Any]) -> NavigatorEntry:
    run_id = str(manifest.get("run_id") or "?")
    mode = str(manifest.get("mode") or "--")
    status = str(manifest.get("status") or "--")
    cost = float(manifest.get("total_cost_usd") or 0.0)
    created = str(manifest.get("created_at") or manifest.get("started_at") or "")
    label = f"{run_id[:12]}"
    detail = f"{mode} {status} ${cost:.4f}"
    if created:
        detail = f"{detail} {created[:16]}"
    return NavigatorEntry(kind="run", key=run_id, label=label, detail=detail, payload=manifest)


class NavigatorWidget(Vertical):
    """Searchable list of past runs, git worktrees, and playbooks."""

    DEFAULT_CSS = """
    NavigatorWidget {
        height: 100%;
    }
    NavigatorWidget Input {
        margin-bottom: 1;
    }
    NavigatorWidget ListView {
        height: 1fr;
    }
    """

    search: reactive[str] = reactive("")
    selected_kind: reactive[str | None] = reactive(None)
    selected_key: reactive[str | None] = reactive(None)

    class RunSelected(Message):
        def __init__(self, run_id: str, manifest: dict[str, Any]) -> None:
            super().__init__()
            self.run_id = run_id
            self.manifest = manifest

    class WorktreeSelected(Message):
        def __init__(self, name: str, path: Path | None) -> None:
            super().__init__()
            self.name = name
            self.path = path

    class PlaybookSelected(Message):
        def __init__(self, name: str, path: str) -> None:
            super().__init__()
            self.name = name
            self.path = path

    class EntriesRefreshed(Message):
        def __init__(self, entries: list[NavigatorEntry]) -> None:
            super().__init__()
            self.entries = entries

    def __init__(self, *, worktree_manager: WorktreeManager | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.worktree_manager = worktree_manager
        self._entries: list[NavigatorEntry] = []

    def compose(self) -> ComposeResult:
        yield Static("Navigator", classes="pane-title")
        yield Input(placeholder="Search runs, worktrees, playbooks", id="navigator-search")
        yield ListView(id="navigator-list")

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_entries)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "navigator-search":
            self.search = event.value

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if not isinstance(item, NavigatorListItem):
            return
        entry = item.entry
        self.selected_kind = entry.kind
        self.selected_key = entry.key
        if entry.kind == "run":
            self.post_message(self.RunSelected(entry.key, entry.payload or {}))
        elif entry.kind == "worktree":
            self.post_message(self.WorktreeSelected(entry.key, entry.path))
        elif entry.kind == "playbook":
            self.post_message(self.PlaybookSelected(entry.label, entry.key))

    def watch_search(self, _value: str) -> None:
        if self.is_mounted:
            self._render_entries()

    @work(exclusive=True, thread=True)
    def refresh_entries(self) -> None:
        """Reload run, worktree, and playbook data from disk."""
        previous_selection = (self.selected_kind, self.selected_key)
        entries: list[NavigatorEntry] = []
        
        # 1. Runs
        try:
            entries.extend(_format_run(manifest) for manifest in artifacts.list_runs())
        except Exception:
            entries.append(NavigatorEntry("status", "runs-error", "Runs unavailable", "artifact read failed"))

        # 2. Playbooks
        try:
            # Look in project root and docs/plans for .md files with checkboxes
            proj_root = config.project_root()
            for pattern in ["*.md", "docs/plans/*.md"]:
                for path in proj_root.glob(pattern):
                    # Quick check if it looks like a playbook
                    content = path.read_text(encoding="utf-8")[:1000]
                    if "- [ ]" in content or "- [x]" in content:
                        entries.append(NavigatorEntry(
                            kind="playbook",
                            key=str(path),
                            label=path.name,
                            detail=f"Automated playbook: {path.relative_to(proj_root)}"
                        ))
        except Exception:
            pass

        
        # 2. Playbooks
        try:
            from orchestra import config as _cfg
            proj_root = _cfg.project_root()
            for pattern in ["*.md", "docs/plans/*.md"]:
                for path in proj_root.glob(pattern):
                    try:
                        text = path.read_text(encoding="utf-8")[:1000]
                        if "- [ ]" in text or "- [x]" in text:
                            entries.append(NavigatorEntry(
                                kind="playbook",
                                key=str(path),
                                label=path.name,
                                detail=f"Playbook: {path.name}"
                            ))
                    except: continue
        except Exception:
            pass

        # 3. Worktrees
        try:
            manager = self.worktree_manager or WorktreeManager()
            root = manager.worktrees_root
            for name in manager.list_worktrees():
                entries.append(
                    NavigatorEntry(
                        kind="worktree",
                        key=name,
                        label=name,
                        detail=str(root / name),
                        path=root / name,
                    )
                )
        except Exception:
            entries.append(NavigatorEntry("status", "worktrees-error", "Worktrees unavailable", "git read failed"))

        if not entries:
            entries.append(NavigatorEntry("status", "empty", "No data found", "launch a task to populate"))

        self._entries = entries
        if self.is_mounted:
            self.app.call_from_thread(self._render_entries, previous_selection=previous_selection)
        self.app.call_from_thread(self.post_message, self.EntriesRefreshed(entries))

    def _render_entries(self, previous_selection: tuple[str | None, str | None] | None = None) -> None:
        list_view = self.query_one("#navigator-list", ListView)
        list_view.clear()
        needle = self.search.strip().lower()
        visible_entries: list[NavigatorEntry] = []
        for entry in self._entries:
            haystack = f"{entry.kind} {entry.key} {entry.label} {entry.detail}".lower()
            if needle and needle not in haystack:
                continue
            visible_entries.append(entry)

        if not visible_entries:
            visible_entries.append(NavigatorEntry("status", "no-match", "No matches", f"search: {self.search}"))

        for entry in visible_entries:
            list_view.append(NavigatorListItem(entry))

        target_kind, target_key = previous_selection or (self.selected_kind, self.selected_key)
        if target_kind is None or target_key is None:
            return

        for index, entry in enumerate(visible_entries):
            if entry.kind == target_kind and entry.key == target_key:
                list_view.index = index
                break
