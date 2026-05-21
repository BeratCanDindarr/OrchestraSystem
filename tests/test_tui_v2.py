"""Focused regression tests for the Orchestrator's Console TUI v2."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.widgets import Static

from orchestra.tui.app_v2 import OrchestraAppV2


def _static_text(widget: Static) -> str:
    content = widget.content
    return content.plain if hasattr(content, "plain") else str(content)


def _write_run(
    root: Path,
    run_id: str,
    manifest: dict,
    *,
    events: list[dict] | None = None,
    logs: dict[str, str] | None = None,
) -> None:
    run_path = root / run_id
    agents_path = run_path / "agents"
    agents_path.mkdir(parents=True, exist_ok=True)
    (run_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if events is not None:
        payload = "\n".join(json.dumps(event) for event in events)
        (run_path / "events.jsonl").write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    for alias, content in (logs or {}).items():
        (agents_path / f"{alias}.stdout.log").write_text(content, encoding="utf-8")


@pytest.fixture
def mock_v2_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("orchestra.config.artifact_root", lambda: tmp_path)

    class DummyWorktreeManager:
        def __init__(self, *_args, **_kwargs) -> None:
            self.worktrees_root = tmp_path / "worktrees"
            self.worktrees_root.mkdir(exist_ok=True)

        def list_worktrees(self) -> list[str]:
            return []

    monkeypatch.setattr("orchestra.tui.widgets.v2.navigator.WorktreeManager", DummyWorktreeManager)
    return tmp_path


@pytest.mark.asyncio
async def test_v2_selection_hydrates_arena_and_monitor(mock_v2_artifacts: Path) -> None:
    manifest = {
        "run_id": "run-v2",
        "mode": "dual",
        "status": "running",
        "total_cost_usd": 1.2345,
        "agents": [
            {"alias": "alpha", "status": "running"},
            {"alias": "beta", "status": "queued"},
        ],
    }
    _write_run(
        mock_v2_artifacts,
        "run-v2",
        manifest,
        events=[{"event": "cue_applied", "rule": "auto-triage"}],
        logs={
            "alpha": "alpha line 1\n",
            "beta": "beta line 1\n",
        },
    )

    app = OrchestraAppV2(interval=60.0)
    async with app.run_test():
        app._select_run("run-v2", manifest)

        arena_status = _static_text(app.query_one("#arena-status", Static))
        assert "displaying alpha, beta" in arena_status
        assert "$1.2345" in arena_status

        cue_feed = _static_text(app.query_one("#cue-feed", Static))
        assert "auto-triage" in cue_feed

        app.query_one("#arena-pane").set_manual_aliases(["beta"])
        arena_status = _static_text(app.query_one("#arena-status", Static))
        assert "displaying beta" in arena_status
        assert "alpha, beta" not in arena_status


@pytest.mark.asyncio
async def test_v2_poll_updates_cost_status_and_new_cue_events(mock_v2_artifacts: Path) -> None:
    manifest = {
        "run_id": "run-poll",
        "mode": "critical",
        "status": "running",
        "total_cost_usd": 0.2500,
        "agents": [
            {"alias": "critic-a", "status": "running"},
        ],
    }
    _write_run(
        mock_v2_artifacts,
        "run-poll",
        manifest,
        events=[{"event": "cue_started", "rule": "bootstrap"}],
        logs={"critic-a": "initial line\n"},
    )

    app = OrchestraAppV2(interval=60.0)
    async with app.run_test():
        app._select_run("run-poll", manifest)
        first_feed = _static_text(app.query_one("#cue-feed", Static))
        assert "bootstrap" in first_feed

        updated_manifest = {
            **manifest,
            "status": "completed",
            "total_cost_usd": 0.8750,
            "agents": [{"alias": "critic-a", "status": "completed"}],
        }
        _write_run(
            mock_v2_artifacts,
            "run-poll",
            updated_manifest,
            events=[
                {"event": "cue_started", "rule": "bootstrap"},
                {"event": "cue_finished", "rule": "wrap-up"},
            ],
            logs={"critic-a": "initial line\ncompleted line\n"},
        )

        app._poll_selected_run()

        total_cost = _static_text(app.query_one("#total-cost", Static))
        fleet_stats = _static_text(app.query_one("#fleet-stats", Static))
        cue_feed = _static_text(app.query_one("#cue-feed", Static))

        assert total_cost == "$0.8750"
        assert fleet_stats.startswith("completed")
        assert "wrap-up" in cue_feed
        assert cue_feed.count("bootstrap") == 1
