import pytest
from textual.app import App
from orchestra.tui.app_v2 import OrchestraAppV2
import os
import json
from pathlib import Path

@pytest.mark.asyncio
async def test_console_boots(tmp_path, monkeypatch):
    # Mock environment
    monkeypatch.setattr("orchestra.config.artifact_root", lambda: tmp_path)
    
    # Create a dummy run
    run_id = "test-run-001"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "agents").mkdir()
    manifest = {
        "run_id": run_id,
        "mode": "ask",
        "status": "completed",
        "task": "hello",
        "agents": []
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    
    app = OrchestraAppV2()
    async with app.run_test() as pilot:
        # Check if basic elements are composed
        assert app.query_one("#left-pane") is not None
        assert app.query_one("#arena-pane") is not None
        assert app.query_one("#right-pane") is not None
        assert app.query_one("#command-bar") is not None
        
        # Test command bar input
        await pilot.press(*"help")
        await pilot.press("enter")
        # Notification should appear (difficult to test exactly but confirms no crash)
