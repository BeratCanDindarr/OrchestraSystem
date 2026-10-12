"""Tests for the `orchestra claude` launcher (no real gateway / claude process)."""
from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

from orchestra import claude_launcher as cl


def _fake_health_resp(status=200):
    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    r = _R(b"")
    r.status = status
    return r


def test_gateway_healthy_true_on_200():
    with patch("orchestra.claude_launcher.urllib.request.urlopen", return_value=_fake_health_resp(200)):
        assert cl.gateway_healthy("127.0.0.1", 7070) is True


def test_gateway_healthy_false_on_error():
    import urllib.error
    def _boom(req, timeout=0):
        raise urllib.error.URLError("refused")
    with patch("orchestra.claude_launcher.urllib.request.urlopen", _boom):
        assert cl.gateway_healthy("127.0.0.1", 7070) is False


def test_gateway_settings_defaults():
    with patch("orchestra.claude_launcher.config.section", return_value={}):
        assert cl.gateway_settings() == ("127.0.0.1", 7070, "oll-fast")


def test_gateway_settings_from_config():
    with patch("orchestra.claude_launcher.config.section",
               return_value={"host": "127.0.0.1", "port": 8100, "default_model": "oll-coder"}):
        assert cl.gateway_settings() == ("127.0.0.1", 8100, "oll-coder")


def test_build_claude_env_sets_anthropic_vars():
    with patch.dict("os.environ", {}, clear=True):
        env = cl.build_claude_env("127.0.0.1", 7070, "oll-fast")
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:7070"
    assert env["ANTHROPIC_MODEL"] == "oll-fast"
    assert env["ANTHROPIC_API_KEY"] == "orchestra-local"   # placeholder when unset


def test_run_claude_uses_running_gateway_without_spawning():
    with patch("orchestra.claude_launcher.gateway_settings", return_value=("127.0.0.1", 7070, "oll-fast")), \
         patch("orchestra.claude_launcher.gateway_healthy", return_value=True), \
         patch("orchestra.remote_nodes.verify_gateway", return_value=True), \
         patch("orchestra.claude_launcher._spawn_gateway") as spawn, \
         patch("orchestra.claude_launcher.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0)
        code = cl.run_claude(model=None, port=None, extra_args=["-p", "hi"])
    assert code == 0
    spawn.assert_not_called()                              # gateway already up
    args, kwargs = run.call_args
    assert args[0] == ["claude", "-p", "hi"]               # extra args passed through
    assert kwargs["env"]["ANTHROPIC_MODEL"] == "oll-fast"


def test_run_claude_spawns_and_tears_down_gateway():
    proc = MagicMock()
    with patch("orchestra.claude_launcher.gateway_settings", return_value=("127.0.0.1", 7070, "oll-fast")), \
         patch("orchestra.claude_launcher.gateway_healthy", return_value=False), \
         patch("orchestra.claude_launcher.wait_for_gateway", return_value=True), \
         patch("orchestra.claude_launcher._spawn_gateway", return_value=proc) as spawn, \
         patch("orchestra.claude_launcher.subprocess.run", return_value=MagicMock(returncode=3)):
        code = cl.run_claude()
    assert code == 3
    spawn.assert_called_once()
    proc.terminate.assert_called_once()                    # we started it → we stop it


def test_run_claude_missing_binary_raises():
    with patch("orchestra.claude_launcher.gateway_settings", return_value=("127.0.0.1", 7070, "oll-fast")), \
         patch("orchestra.claude_launcher.gateway_healthy", return_value=True), \
         patch("orchestra.remote_nodes.verify_gateway", return_value=True), \
         patch("orchestra.claude_launcher.subprocess.run", side_effect=FileNotFoundError()):
        import pytest
        with pytest.raises(RuntimeError, match="claude"):
            cl.run_claude()
