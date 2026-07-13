# tests/test_chat_session.py
from unittest.mock import patch
from orchestra.chat.session import ChatSession


class _StreamProvider:
    name = "mock"
    def model_label(self, e): return f"mock/{e}"
    def stream(self, prompt, effort, timeout=180, **k):
        assert "User: hi" in prompt
        yield "hel"; yield "lo"


class _RunOnlyProvider:
    name = "mockrun"
    def model_label(self, e): return f"mockrun/{e}"
    def stream(self, prompt, effort, timeout=180, **k):
        raise NotImplementedError
    def run(self, prompt, effort, timeout=180, **k):
        return "one-shot reply", 0


def _patch_root(tmp_path):
    return patch("orchestra.config.orchestra_root", return_value=tmp_path)


def test_send_streams_and_records(tmp_path):
    with _patch_root(tmp_path), patch("orchestra.chat.session.resolve", return_value=(_StreamProvider(), "fast")):
        s = ChatSession.start("mock-alias")
        chunks = []
        reply = s.send("hi", on_chunk=chunks.append)
    assert reply == "hello"
    assert chunks == ["hel", "lo"]
    assert s.history[-2:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_send_falls_back_to_run(tmp_path):
    with _patch_root(tmp_path), patch("orchestra.chat.session.resolve", return_value=(_RunOnlyProvider(), "fast")):
        s = ChatSession.start("mock-alias")
        reply = s.send("hi")
    assert reply == "one-shot reply"


def test_switch_model_keeps_history(tmp_path):
    with _patch_root(tmp_path), patch("orchestra.chat.session.resolve", return_value=(_StreamProvider(), "fast")):
        s = ChatSession.start("mock-alias")
        s.send("hi")
        with patch("orchestra.chat.session.resolve", return_value=(_RunOnlyProvider(), "slow")) as r:
            s.switch_model("other")
        assert r.called
        assert len(s.history) == 2  # preserved
        assert s.alias == "other"
