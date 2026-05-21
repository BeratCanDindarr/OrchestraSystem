"""Focused tests for Orchestra TUI v2 widgets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListView, Static

from orchestra.tui.app_v2 import OrchestraAppV2
from orchestra.tui.widgets.v2 import AgentStreamWidget, MonitorWidget, NavigatorWidget
from orchestra.engine.worktree_manager import WorktreeManager


class WidgetHarness(App[None]):
    """Minimal Textual app used to mount a single widget under test."""

    def __init__(self, widget) -> None:
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


@pytest.fixture
def orchestra_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Route Orchestra artifacts and worktrees into a temporary sandbox."""
    orchestra_root = tmp_path / ".orchestra"
    runs_root = orchestra_root / "runs"
    worktrees_root = orchestra_root / "worktrees"
    runs_root.mkdir(parents=True)
    worktrees_root.mkdir(parents=True)

    monkeypatch.setattr("orchestra.engine.artifacts.config.artifact_root", lambda: runs_root)
    monkeypatch.setattr("orchestra.engine.worktree_manager.config.orchestra_root", lambda: orchestra_root)
    monkeypatch.setattr("orchestra.engine.worktree_manager.config.project_root", lambda: tmp_path)
    return {
        "orchestra_root": orchestra_root,
        "runs_root": runs_root,
        "worktrees_root": worktrees_root,
    }


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "agents").mkdir(exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.asyncio
async def test_navigator_widget_lists_runs_and_worktrees(orchestra_roots: dict[str, Path]) -> None:
    runs_root = orchestra_roots["runs_root"]
    worktrees_root = orchestra_roots["worktrees_root"]

    _write_manifest(
        runs_root / "run-001",
        {
            "run_id": "run-001",
            "mode": "ask",
            "status": "completed",
            "total_cost_usd": 0.42,
            "created_at": "2026-05-22T10:15:00+00:00",
            "agents": [],
        },
    )
    (worktrees_root / "feature-alpha").mkdir()

    manager = WorktreeManager(repo_root=orchestra_roots["orchestra_root"].parent)
    manager.worktrees_root = worktrees_root
    widget = NavigatorWidget(worktree_manager=manager)

    async with WidgetHarness(widget).run_test() as pilot:
        await pilot.pause()
        widget.refresh_entries()
        await pilot.pause()

        list_view = widget.query_one("#navigator-list", ListView)
        assert len(list_view.children) == 2

        widget.search = "feature-alpha"
        await pilot.pause()
        assert len(list_view.children) == 1

        widget.search = "missing"
        await pilot.pause()
        assert len(list_view.children) == 1
        placeholder = list_view.children[0]
        assert getattr(placeholder, "entry").key == "no-match"


@pytest.mark.asyncio
async def test_monitor_widget_tracks_cost_and_filters_cue_events() -> None:
    widget = MonitorWidget()

    async with WidgetHarness(widget).run_test() as pilot:
        widget.update_from_manifest(
            {
                "run_id": "run-xyz",
                "status": "running",
                "budget_usd": 2.0,
                "agents": [
                    {"alias": "a", "status": "running", "estimated_cost_usd": 0.25},
                    {"alias": "b", "status": "completed", "estimated_cost_usd": 0.50},
                    {"alias": "c", "status": "failed", "estimated_cost_usd": 0.10},
                ],
            }
        )
        widget.add_event({"event": "cue_triggered", "rule": "nightly-sync", "ts": "2026-05-22T10:20:30+00:00"})
        widget.add_event({"event": "agent_output", "name": "ignored"})
        await pilot.pause()

        assert widget.total_cost_usd == pytest.approx(0.85)
        assert widget.active_agents == 1
        assert widget.completed_agents == 1
        assert widget.failed_agents == 1
        assert len(widget._cue_events) == 1

        feed = widget.query_one("#cue-feed", Static).render()
        assert "nightly-sync" in (feed.plain if hasattr(feed, "plain") else str(feed))


@pytest.mark.asyncio
async def test_app_v2_stream_widget_loads_agent_logs(orchestra_roots: dict[str, Path]) -> None:
    run_dir = orchestra_roots["runs_root"] / "run-live"
    manifest = {
        "run_id": "run-live",
        "mode": "ask",
        "status": "running",
        "total_cost_usd": 1.25,
        "budget_usd": 3.0,
        "agents": [
            {"alias": "agent-a", "status": "running"},
        ],
    }
    _write_manifest(run_dir, manifest)
    (run_dir / "agents" / "agent-a.stdout.log").write_text(
        "\u001b[32mready\u001b[0m\nexec_command started\npartial",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps({"event": "cue_triggered", "rule": "morning-check", "ts": "2026-05-22T11:00:00+00:00"}) + "\n",
        encoding="utf-8",
    )

    app = OrchestraAppV2(run_id="run-live", interval=60)
    async with app.run_test() as pilot:
        await pilot.pause()

        assert app.selected_run_id == "run-live"

        stream = app.query_one("#arena-stream-1", AgentStreamWidget)
        assert stream.agent_alias == "agent-a"
        assert stream.line_count == 2
        assert stream._buffer == ["\u001b[32mready\u001b[0m", "exec_command started"]

        (run_dir / "agents" / "agent-a.stdout.log").write_text(
            "\u001b[32mready\u001b[0m\nexec_command started\npartial line\n",
            encoding="utf-8",
        )
        stream.poll()
        assert stream.line_count == 3
        assert stream._buffer[-1] == "partial line"

        monitor = app.query_one(MonitorWidget)
        assert monitor.total_cost_usd == pytest.approx(1.25)
        assert len(monitor._cue_events) == 1
