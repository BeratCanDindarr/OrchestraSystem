"""The Ollama provider tells a model how much to think and gives up at its deadline.

Without a think level a thinking model can reason for minutes before its first
answer token, and every thinking token is billed as output.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from orchestra.providers import ollama
from orchestra.providers.ollama import OllamaProvider, think_level


class _FakeStream:
    """A streaming /api/generate response: one JSON object per line."""

    def __init__(self, chunks: list[dict]):
        self._lines = [json.dumps(chunk).encode() + b"\n" for chunk in chunks]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._lines[0]


def _capture(chunks: list[dict], captured: dict):
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeStream(chunks)

    return fake_urlopen


def _ollama_config(**ollama_section):
    return patch.object(ollama.config, "providers_config", return_value={"ollama": ollama_section})


ANSWER = [{"response": "ok", "done": False}, {"response": "", "done": True}]


class TestThinkLevel:
    def test_nothing_configured_means_nothing_is_sent(self):
        with _ollama_config():
            assert think_level("glm-5.3:cloud") is None

    def test_the_default_applies_to_every_model(self):
        with _ollama_config(think="low"):
            assert think_level("glm-5.3:cloud") == "low"

    def test_a_model_of_its_own_wins_over_the_default(self):
        with _ollama_config(think="low", think_by_model={"deepseek-v4-pro:cloud": False}):
            assert think_level("deepseek-v4-pro:cloud") is False
            assert think_level("glm-5.3:cloud") == "low"

    def test_values_that_are_not_a_level_are_ignored(self):
        with _ollama_config(think=3, think_by_model={"a": ["low"], "b": "extreme"}):
            assert think_level("a") is None
            assert think_level("b") is None
            assert think_level("c") is None


class TestRequestsCarryTheThinkLevel:
    def test_native_run_sends_the_level_of_its_model(self):
        captured: dict = {}
        with _ollama_config(think_by_model={"glm-5.3:cloud": "low"}), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(ANSWER, captured)):
            output, returncode = OllamaProvider().native_run("hi", "glm-5.3:cloud")
        assert (output, returncode) == ("ok", 0)
        assert captured["body"]["think"] == "low"

    def test_native_run_can_switch_thinking_off(self):
        captured: dict = {}
        with _ollama_config(think_by_model={"deepseek-v4-pro:cloud": False}), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(ANSWER, captured)):
            OllamaProvider().native_run("hi", "deepseek-v4-pro:cloud")
        assert captured["body"]["think"] is False

    def test_native_run_sends_no_level_when_none_is_set(self):
        captured: dict = {}
        with _ollama_config(), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(ANSWER, captured)):
            OllamaProvider().native_run("hi", "glm-5.3:cloud")
        assert "think" not in captured["body"]

    def test_the_level_follows_the_resolved_model_of_a_short_name(self):
        captured: dict = {}
        resolved = ollama.MODELS["fast"]
        with _ollama_config(think_by_model={resolved: False}), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(ANSWER, captured)):
            OllamaProvider().native_run("hi", "fast")
        assert captured["body"]["model"] == resolved
        assert captured["body"]["think"] is False

    def test_stream_sends_the_level(self):
        captured: dict = {}
        with _ollama_config(think="low"), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(ANSWER, captured)):
            assert list(OllamaProvider().stream("hi", "glm-5.3:cloud")) == ["ok"]
        assert captured["body"]["think"] == "low"

    def test_native_messages_sends_the_level(self):
        captured: dict = {}
        reply = [{"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}]
        with _ollama_config(think_by_model={"glm-5.3:cloud": "low"}), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(reply, captured)):
            OllamaProvider().native_messages({"model": "glm-5.3:cloud", "messages": [{"role": "user", "content": "hi"}]})
        assert captured["body"]["think"] == "low"


class TestDeadline:
    """Thinking chunks keep the socket alive, so the read timeout alone never fires."""

    def test_native_run_gives_up_when_the_model_only_thinks(self):
        thinking_forever = [{"thinking": "hmm", "response": "", "done": False}] * 50
        clock = iter(range(0, 10_000, 10))  # every look at the clock is 10 s later
        with _ollama_config(), \
                patch("orchestra.providers.ollama.time.monotonic", lambda: next(clock)), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(thinking_forever, {})):
            output, returncode = OllamaProvider().native_run("hi", "glm-5.3:cloud", timeout=60)
        assert returncode == 124
        assert output.startswith("[ERROR]") and "60s" in output

    def test_an_answer_that_arrives_in_time_is_returned(self):
        chunks = [{"thinking": "hmm", "response": "", "done": False}] * 3 + ANSWER
        clock = iter(range(0, 10_000, 10))
        with _ollama_config(), \
                patch("orchestra.providers.ollama.time.monotonic", lambda: next(clock)), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(chunks, {})):
            assert OllamaProvider().native_run("hi", "glm-5.3:cloud", timeout=600) == ("ok", 0)

    def test_stream_stops_at_the_deadline(self):
        thinking_forever = [{"thinking": "hmm", "response": "", "done": False}] * 50
        clock = iter(range(0, 10_000, 10))
        with _ollama_config(), \
                patch("orchestra.providers.ollama.time.monotonic", lambda: next(clock)), \
                patch("orchestra.providers.ollama.urllib.request.urlopen", _capture(thinking_forever, {})):
            chunks = list(OllamaProvider().stream("hi", "glm-5.3:cloud", timeout=60))
        assert len(chunks) == 1 and chunks[0].startswith("[ERROR]")
