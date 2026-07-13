from unittest.mock import patch
from typer.testing import CliRunner
from orchestra.cli import app

runner = CliRunner()


def test_nodes_lists_configured():
    cfg = {"windows": {"url": "http://192.168.1.42:7070", "token": "s"}}
    with patch("orchestra.config.remote_nodes_config", return_value=cfg), \
         patch("orchestra.remote_nodes.get_node") as mock_get:
        mock_get.return_value.is_available.return_value = True
        result = runner.invoke(app, ["nodes"])
    assert result.exit_code == 0
    assert "windows" in result.output


def test_nodes_scan_auto_adds(tmp_path):
    found = [{"ip": "192.168.1.42", "host": "winbox", "port": 7070}]
    written = {}
    with patch("orchestra.remote_nodes.default_subnet_prefix", return_value="192.168.1"), \
         patch("orchestra.remote_nodes.scan_subnet", return_value=found), \
         patch("orchestra.cli._set_node_config", side_effect=lambda n, u, t: written.update({n: (u, t)})):
        result = runner.invoke(app, ["nodes-scan", "--token", "s"])
    assert result.exit_code == 0
    assert "winbox" in written  # auto-added by hostname
    assert written["winbox"][0] == "http://192.168.1.42:7070"
