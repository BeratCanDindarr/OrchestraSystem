"""Integration tests for CLI: run, ask, mcp."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from orchestra.cli import app
from orchestra.models import OrchestraRun, AgentRun, AgentStatus

runner = CliRunner()

def test_ask_basic():
    mock_run = OrchestraRun(mode="ask", task="hello", run_id="r1")
    mock_run.agents = [AgentRun(alias="a1", provider="p1", model="m1", status=AgentStatus.COMPLETED, stdout_log="Hi")]
    
    # Patch the correct entry point used by CLI
    with patch("orchestra.engine.runners.run_ask", return_value=mock_run):
        result = runner.invoke(app, ["ask", "a1", "hello", "--raw"])
        assert result.exit_code == 0
        assert "Hi" in result.output

def test_run_auto():
    mock_run = OrchestraRun(mode="auto", task="hello", run_id="r1")
    mock_run.summary = "Success"
    
    with patch("orchestra.router.task_to_mode", return_value="ask a1"), \
         patch("orchestra.engine.runners.run_ask", return_value=mock_run):
        result = runner.invoke(app, ["run", "auto", "hello"])
        assert result.exit_code == 0

def test_run_dual():
    mock_run = OrchestraRun(mode="dual", task="hello", run_id="r1")
    mock_run.summary = "Dual Success"
    
    with patch("orchestra.engine.runners.run_dual", return_value=mock_run):
        result = runner.invoke(app, ["run", "dual", "hello"])
        assert result.exit_code == 0
