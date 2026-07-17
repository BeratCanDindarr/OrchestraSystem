"""Test for the `orchestra dashboard` command (starts server, stops on interrupt)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from orchestra.cli import app

runner = CliRunner()


def test_dashboard_cmd_starts_on_port_and_stops_on_interrupt():
    fake_server = MagicMock()
    with patch("orchestra.monitoring.web_dashboard.start_web_dashboard",
               return_value=fake_server) as mock_start, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        result = runner.invoke(app, ["dashboard", "--port", "9099"])
    assert result.exit_code == 0
    mock_start.assert_called_once_with(9099)
    fake_server.shutdown.assert_called_once()
