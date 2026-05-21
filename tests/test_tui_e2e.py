"""End-to-end tests for Orchestra TUI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestra.tui.app import OrchestraApp
from orchestra.tui.widgets.agent_tree import AgentTree
from orchestra.tui.widgets.stream_view import StreamView


@pytest.fixture
def mock_run_dir(tmp_path: Path, monkeypatch):
    """Patch run_dir so the TUI reads/writes under tmp_path."""

    def _make_run_dir(run_id: str) -> Path:
        d = tmp_path / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "agents").mkdir(exist_ok=True)
        return d

    monkeypatch.setattr("orchestra.tui.app.run_dir", _make_run_dir)
    return tmp_path


@pytest.mark.asyncio
async def test_app_startup(mock_run_dir: Path) -> None:
    """Mount the app and verify the header bar and agent tree exist."""
    app = OrchestraApp(run_id="test-run")
    async with app.run_test() as pilot:
        assert app.title == "Orchestra"

        header = app.query_one("#header")
        assert header is not None

        agent_tree = app.query_one(AgentTree)
        assert agent_tree is not None
        assert "Orchestra" in str(agent_tree.root.label)


@pytest.mark.asyncio
async def test_agent_tree_update(mock_run_dir: Path) -> None:
    """Inject a manifest with two agents and verify tree updates reflect status changes."""
    app = OrchestraApp(run_id="test-run")
    manifest = {
        "run_id": "test-run",
        "mode": "ask",
        "status": "running",
        "created_at": "2024-01-01T00:00:00+00:00",
        "agents": [
            {"alias": "agent-a", "status": "running", "model": "gpt-4", "elapsed": "00:12"},
            {"alias": "agent-b", "status": "pending", "model": "gpt-3.5", "elapsed": "--:--"},
        ],
    }
    (app._run_path / "manifest.json").write_text(json.dumps(manifest))

    async with app.run_test() as pilot:
        agent_tree = app.query_one(AgentTree)

        # Initial population
        app._poll_manifest()
        assert len(agent_tree.root.children) == 2
        labels = [child.label.plain for child in agent_tree.root.children]
        assert any("agent-a" in label for label in labels)
        assert any("agent-b" in label for label in labels)

        # Update one agent to completed
        manifest["agents"][0]["status"] = "completed"
        (app._run_path / "manifest.json").write_text(json.dumps(manifest))
        app._poll_manifest()

        labels = [child.label.plain for child in agent_tree.root.children]
        a_label = next(label for label in labels if "agent-a" in label)
        assert "completed" in a_label


@pytest.mark.asyncio
async def test_stream_view(mock_run_dir: Path) -> None:
    """Inject mock log lines and verify the stream view contains them."""
    app = OrchestraApp(run_id="test-run")
    manifest = {
        "run_id": "test-run",
        "mode": "ask",
        "status": "running",
        "created_at": "2024-01-01T00:00:00+00:00",
        "agents": [
            {"alias": "agent-a", "status": "running", "model": "gpt-4", "elapsed": "00:12"},
        ],
    }
    (app._run_path / "manifest.json").write_text(json.dumps(manifest))

    log_file = app._run_path / "agents" / "agent-a.stdout.log"
    log_file.write_text("Hello from agent-a\nSecond line\n")

    async with app.run_test() as pilot:
        # Trigger manifest poll so auto-selection loads agent-a's log
        app._poll_manifest()

        stream_view = app.query_one(StreamView)
        assert stream_view._log_path == log_file

        texts = [line for line, _cat, _color in stream_view._buffer]
        assert "Hello from agent-a" in texts
        assert "Second line" in texts
