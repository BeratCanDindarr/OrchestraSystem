"""Tests for the injectable REPL loop."""
from unittest.mock import MagicMock
from orchestra.chat.loop import run_repl


def test_loop_sends_then_exits():
    session = MagicMock()
    session.send.return_value = "reply"
    inputs = iter(["hello", "/exit"])
    def _in(_prompt=""):
        return next(inputs)
    outputs = []
    run_repl(session, input_fn=_in, output_fn=lambda *a, **k: outputs.append(a[0] if a else ""))
    session.send.assert_called_once()
    assert session.send.call_args[0][0] == "hello"


def test_loop_skips_empty_and_stops_on_eof():
    session = MagicMock()
    seq = iter(["", "   "])
    def _in(_prompt=""):
        try:
            return next(seq)
        except StopIteration:
            raise EOFError
    run_repl(session, input_fn=_in, output_fn=lambda *a, **k: None)
    session.send.assert_not_called()


def test_loop_keyboardinterrupt_cancels_turn_not_loop():
    session = MagicMock()
    calls = {"n": 0}
    def _send(msg, on_chunk=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return "ok"
    session.send.side_effect = _send
    seq = iter(["first", "second", "/exit"])
    run_repl(session, input_fn=lambda _p="": next(seq), output_fn=lambda *a, **k: None)
    assert calls["n"] == 2  # first cancelled, second completed
