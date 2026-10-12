"""Integration tests for CLI live commands: ps, logs, watch, cancel, config."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner
from datetime import datetime, timezone

from orchestra.cli import app

runner = CliRunner()

def test_ps_basic():
    mock_runs = [
        {
            "run_id": "r1",
            "mode": "ask",
            "status": "completed",
            "created_at": "2026-05-21T10:00:00Z",
            "task": "test task"
        }
    ]
    with patch("orchestra.engine.artifacts.list_runs", return_value=mock_runs):
        result = runner.invoke(app, ["ps"])
        assert result.exit_code == 0
        assert "r1" in result.output
        assert "completed" in result.output

def test_ps_json():
    mock_runs = [{"run_id": "r1", "status": "completed"}]
    with patch("orchestra.engine.artifacts.list_runs", return_value=mock_runs):
        result = runner.invoke(app, ["ps", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["runs"][0]["run_id"] == "r1"

def test_logs_not_found():
    with patch("orchestra.engine.artifacts.run_dir", return_value=Path("/tmp/nonexistent")):
        result = runner.invoke(app, ["logs", "r1"])
        assert result.exit_code == 1
        assert "Run r1 not found" in result.output

def test_logs_success(tmp_path):
    run_id = "r1"
    agents_dir = tmp_path / run_id / "agents"
    agents_dir.mkdir(parents=True)
    log_file = agents_dir / "cdx-deep.stdout.log"
    log_file.write_text("Hello Log", encoding="utf-8")

    with patch("orchestra.config.artifact_root", return_value=tmp_path):
        result = runner.invoke(app, ["logs", run_id])
        assert result.exit_code == 0
        assert "cdx-deep" in result.output
        assert "Hello Log" in result.output

def test_cancel_not_found():
    with patch("orchestra.engine.artifacts.load_manifest", return_value=None):
        result = runner.invoke(app, ["cancel", "r1"])
        assert result.exit_code == 1
        assert "Run r1 not found" in result.output

def test_cancel_success(tmp_path):
    """The command cancels through the service, which also stops the agents and the steps."""
    (tmp_path / "r1").mkdir()
    cancelled = {"run": {"run_id": "r1", "status": "cancelled"}, "already_finished": False}
    with patch("orchestra.config.artifact_root", return_value=tmp_path), \
         patch("orchestra.service.cancel_run", return_value=cancelled) as cancel_run:
        result = runner.invoke(app, ["cancel", "r1"])
        assert result.exit_code == 0
        assert "cancelled" in result.output
        cancel_run.assert_called_once_with("r1")

def test_watch_not_found():
    with patch("orchestra.engine.artifacts.run_dir", return_value=Path("/tmp/nonexistent")):
        result = runner.invoke(app, ["watch", "r1"])
        assert result.exit_code == 1
        assert "Run r1 not found" in result.output

def test_aliases_basic():
    with patch("orchestra.providers.fallback.available_aliases", return_value=["✓ cdx-deep", "✗ gmn-pro"]):
        result = runner.invoke(app, ["aliases"])
        assert result.exit_code == 0
        assert "cdx-deep" in result.output
        assert "gmn-pro" in result.output

def test_config_get():
    with patch("orchestra.cli._get_config_tree", return_value={"a": {"b": 1}}):
        result = runner.invoke(app, ["config", "get", "a.b"])
        assert result.exit_code == 0
        assert "1" in result.output

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Orchestra v" in result.output
