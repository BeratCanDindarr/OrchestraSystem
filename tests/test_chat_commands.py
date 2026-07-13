from unittest.mock import patch, MagicMock
from orchestra.chat import commands


def _session():
    s = MagicMock()
    s.history = [{"role": "user", "content": "x"}]
    s.session_id = "20260713-a-aaaa"
    s.alias = "oll-fast"
    return s


def test_is_command():
    assert commands.is_command("/model x")
    assert not commands.is_command("hello")


def test_exit_returns_exit():
    out = []
    assert commands.handle_command(_session(), "/exit", out.append) == "exit"


def test_model_switches():
    s = _session()
    assert commands.handle_command(s, "/model gmn-pro", lambda *_: None) == "continue"
    s.switch_model.assert_called_once_with("gmn-pro")


def test_clear_empties_history():
    s = _session()
    commands.handle_command(s, "/clear", lambda *_: None)
    assert s.history == []


def test_unknown_command_hints():
    out = []
    assert commands.handle_command(_session(), "/wat", out.append) == "continue"
    assert any("unknown" in o.lower() for o in out)
