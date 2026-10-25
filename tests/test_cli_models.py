from unittest.mock import patch
from typer.testing import CliRunner
from orchestra.cli import app

runner = CliRunner()

_FAKE_DISCOVERY = {
    "macbook": {"base_url": "http://localhost:11434", "online": True,
                "models": [{"name": "qwen2.5:7b", "size": 4_700_000_000}]},
    "windows-gpu": {"base_url": "http://192.168.1.42:11434", "online": False, "models": []},
}


def test_models_lists_backends_and_marks_offline():
    with patch("orchestra.providers.ollama_backends.discover_all", return_value=_FAKE_DISCOVERY):
        result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "macbook" in result.output
    assert "qwen2.5:7b" in result.output
    assert "windows-gpu" in result.output
    assert "offline" in result.output.lower()


def test_models_json_output():
    with patch("orchestra.providers.ollama_backends.discover_all", return_value=_FAKE_DISCOVERY):
        result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0
    assert "qwen2.5:7b" in result.output
    assert "windows-gpu" in result.output


def test_backends_shows_reachability():
    with patch("orchestra.providers.ollama_backends.discover_all", return_value=_FAKE_DISCOVERY):
        result = runner.invoke(app, ["backends"])
    assert result.exit_code == 0
    assert "macbook" in result.output
    assert "windows-gpu" in result.output
