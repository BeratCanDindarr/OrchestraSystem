"""Test for the `orchestra dashboard` command (starts server, stops on interrupt)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from orchestra.cli import app

runner = CliRunner()


def test_dashboard_cmd_starts_on_port_and_stops_on_interrupt():
    with patch("orchestra.monitoring.dashboard_app.run_app",
               side_effect=KeyboardInterrupt) as mock_run:
        result = runner.invoke(app, ["dashboard", "--port", "9099"])
    assert result.exit_code == 0
    mock_run.assert_called_once_with(host="127.0.0.1", port=9099)
