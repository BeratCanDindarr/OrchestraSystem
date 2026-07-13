from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from orchestra.cli import app

runner = CliRunner()


def test_chat_list_prints_sessions():
    fake = [{"session_id": "20260713-a-aaaa", "alias": "oll-fast", "created_at": "2026-07-13", "first_user": "hi there"}]
    with patch("orchestra.chat.store.list_sessions", return_value=fake):
        result = runner.invoke(app, ["chat", "--list"])
    assert result.exit_code == 0
    assert "20260713-a-aaaa" in result.output
    assert "oll-fast" in result.output


def test_chat_starts_session_and_runs_repl():
    sess = MagicMock()
    with patch("orchestra.chat.session.ChatSession.start", return_value=sess) as start, \
         patch("orchestra.chat.loop.run_repl") as repl:
        result = runner.invoke(app, ["chat", "oll-fast"])
    assert result.exit_code == 0
    start.assert_called_once_with("oll-fast")
    repl.assert_called_once_with(sess)


def test_chat_resume_calls_resume():
    sess = MagicMock()
    with patch("orchestra.chat.session.ChatSession.resume", return_value=sess) as res, \
         patch("orchestra.chat.loop.run_repl"):
        result = runner.invoke(app, ["chat", "--resume", "20260713-a-aaaa"])
    assert result.exit_code == 0
    res.assert_called_once_with("20260713-a-aaaa")
