"""Tests for the `orchestra resume` CLI command (approve / --reject across modes)."""
from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from orchestra.cli import app

runner = CliRunner()


def test_resume_defaults_to_approve():
    with patch("orchestra.service.resume_run") as mock_resume:
        result = runner.invoke(app, ["resume", "run-123"])
    assert result.exit_code == 0
    mock_resume.assert_called_once_with("run-123", approval_decision="approve")


def test_resume_reject_flag_passes_reject():
    with patch("orchestra.service.resume_run") as mock_resume:
        result = runner.invoke(app, ["resume", "run-123", "--reject"])
    assert result.exit_code == 0
    mock_resume.assert_called_once_with("run-123", approval_decision="reject")
