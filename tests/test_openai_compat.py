import asyncio
import json
import threading
from unittest.mock import patch, MagicMock

import pytest
from orchestra.openai_compat import (
    messages_to_prompt, resolve_target, to_response, to_sse_chunk, sse_done, list_models,
    run_target, stream_target,
)
import orchestra.openai_compat as oc


def test_messages_to_prompt_system_prefix_and_history():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "again?"},
    ]
    out = messages_to_prompt(msgs)
    assert out == (
        "System: be terse\n\n"
        "User: hi\n\n"
        "Assistant: hello\n\n"
        "User: again?\n\n"
        "Assistant:"
    )


def test_messages_to_prompt_no_system():
    out = messages_to_prompt([{"role": "user", "content": "hi"}])
    assert out == "User: hi\n\nAssistant:"


def test_resolve_target_alias():
    assert resolve_target("oll-fast") == ("alias", "oll-fast")
    assert resolve_target("cdx-deep@windows") == ("alias", "cdx-deep@windows")


def test_resolve_target_modes():
    assert resolve_target("orchestra-dual") == ("mode", "dual")
    assert resolve_target("orchestra-critical") == ("mode", "critical")
    assert resolve_target("orchestra-auto") == ("mode", "auto")


def test_resolve_target_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_target("orchestra-bogus")


def test_to_response_shape():
    r = to_response("oll-fast", "hello", prompt_tokens=3, completion_tokens=1, _id="chatcmpl-x", _created=111)
    assert r["object"] == "chat.completion"
    assert r["id"] == "chatcmpl-x"
    assert r["created"] == 111
    assert r["model"] == "oll-fast"
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"] == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}


def test_to_sse_chunk_delta():
    line = to_sse_chunk("oll-fast", "hel", _id="chatcmpl-x", _created=111)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    obj = json.loads(line[len("data: "):].strip())
    assert obj["object"] == "chat.completion.chunk"
    assert obj["choices"][0]["delta"] == {"content": "hel"}
    assert obj["choices"][0]["finish_reason"] is None


def test_to_sse_chunk_final():
    line = to_sse_chunk("oll-fast", "", finish_reason="stop", _id="chatcmpl-x", _created=111)
    obj = json.loads(line[len("data: "):].strip())
    assert obj["choices"][0]["finish_reason"] == "stop"
    assert obj["choices"][0]["delta"] == {}


def test_sse_done():
    assert sse_done() == "data: [DONE]\n\n"


def test_list_models_includes_aliases_and_modes():
    with patch("orchestra.openai_compat.available_alias_ids", return_value=["oll-fast"]):
        models = list_models()
    ids = [m["id"] for m in models]
    assert "oll-fast" in ids            # available single alias
    assert "gmn-pro" not in ids         # unavailable, skipped
    assert "orchestra-dual" in ids
    assert "orchestra-critical" in ids
    assert "orchestra-auto" in ids
    assert all(m["object"] == "model" for m in models)


# ---------------------------------------------------------------------------
# Task 4: run_target + stream_target
# ---------------------------------------------------------------------------

class _RunProvider:
    def stream(self, prompt, effort, timeout=180, **k):
        raise NotImplementedError   # like base/CLI/remote providers

    def run(self, prompt, effort, timeout=180, **k):
        assert "User: hi" in prompt
        return "one-shot", 0


def test_run_target_alias_uses_provider_run():
    with patch("orchestra.openai_compat.resolve", return_value=(_RunProvider(), "fast")):
        out = asyncio.run(oc.run_target("oll-fast", "User: hi\n\nAssistant:"))
    assert out == "one-shot"


def test_run_target_mode_dual_uses_summary():
    fake_run = MagicMock()
    fake_run.summary = "the synthesis"

    async def _fake_dual(prompt, **k):
        return fake_run

    with patch("orchestra.openai_compat.run_dual_async", _fake_dual):
        out = asyncio.run(oc.run_target("orchestra-dual", "User: go\n\nAssistant:"))
    assert out == "the synthesis"


def test_stream_target_nonstreaming_yields_once():
    async def _collect():
        chunks = []
        async for c in oc.stream_target("oll-fast", "User: hi\n\nAssistant:"):
            chunks.append(c)
        return chunks

    with patch("orchestra.openai_compat.resolve", return_value=(_RunProvider(), "fast")):
        chunks = asyncio.run(_collect())
    assert "".join(chunks) == "one-shot"


# ---------------------------------------------------------------------------
# C1: aclose() stops the background producer thread on client disconnect
# ---------------------------------------------------------------------------

def _fake_alias_provider(chunks, started, released):
    class _P:
        def stream(self, prompt, effort):
            for c in chunks:
                started.set()
                released.wait(timeout=2)   # block until the test releases us
                yield c
        def run(self, prompt, effort):
            return ("full", 0)
    return _P()


def test_stream_target_aclose_stops_producer(monkeypatch):
    started, released = threading.Event(), threading.Event()
    monkeypatch.setattr(oc, "resolve", lambda m: (_fake_alias_provider(["a", "b", "c"], started, released), "fast"))

    async def _run():
        agen = oc.stream_target("oll-fast", "hi")
        # release one chunk, consume it, then close
        released.set()
        first = await agen.__anext__()
        assert first == "a"
        await agen.aclose()  # must run the finally → stop.set(); no hang

    asyncio.run(_run())


def _unbounded_counting_provider(produced):
    """A provider that would stream forever — used to prove aclose() stops it early."""
    import itertools
    import time

    class _P:
        def stream(self, prompt, effort):
            for i in itertools.count():
                produced.append(i)
                time.sleep(0.001)
                yield i

        def run(self, prompt, effort):
            return ("full", 0)

    return _P()


def test_stream_target_aclose_stops_producer_early(monkeypatch):
    # Strong regression guard for the C1 leak fix: an UNBOUNDED producer must stop
    # producing within one iteration of aclose(). If the stop Event were broken it
    # would keep filling the queue indefinitely (~hundreds of items in 0.2s).
    produced: list = []
    monkeypatch.setattr(oc, "resolve", lambda m: (_unbounded_counting_provider(produced), "fast"))

    async def _run():
        agen = oc.stream_target("oll-fast", "hi")
        await agen.__anext__()       # consume one chunk
        await agen.aclose()          # sets stop → producer must break out of stream()
        n1 = len(produced)
        await asyncio.sleep(0.2)     # a broken producer would add ~hundreds here
        return n1, len(produced)

    n1, n2 = asyncio.run(_run())
    assert n2 - n1 <= 2              # bounded: producer observed stop and stopped


# ---------------------------------------------------------------------------
# Task 6 FIX: unknown model must raise ValueError, not silently fallback
# ---------------------------------------------------------------------------

def test_resolve_target_unknown_alias_raises():
    with pytest.raises(ValueError):
        resolve_target("nope-xyz")


def test_resolve_target_known_alias_ok():
    assert resolve_target("cdx-deep") == ("alias", "cdx-deep")


def test_resolve_target_backend_ref_ok():
    # a known base alias with a backend/node suffix is accepted verbatim
    assert resolve_target("cdx-deep@windows") == ("alias", "cdx-deep@windows")
