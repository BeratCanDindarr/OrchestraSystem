import asyncio
import threading
from unittest.mock import patch, MagicMock

import pytest
from orchestra.gateway_dispatch import resolve_target, run_target, stream_target
import orchestra.gateway_dispatch as gd


def test_resolve_target_alias():
    assert resolve_target("oll-fast") == ("alias", "oll-fast")
    assert resolve_target("cdx-deep@windows") == ("alias", "cdx-deep@windows")


def test_resolve_target_modes():
    assert resolve_target("orchestra-dual") == ("mode", "dual")
    assert resolve_target("orchestra-consensus") == ("mode", "consensus")
    assert resolve_target("orchestra-auto") == ("mode", "auto")


def test_resolve_target_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_target("orchestra-bogus")


def test_resolve_target_unknown_alias_raises():
    with pytest.raises(ValueError):
        resolve_target("nope-xyz")


def test_resolve_target_known_alias_ok():
    assert resolve_target("cdx-deep") == ("alias", "cdx-deep")


def test_resolve_target_backend_ref_ok():
    # a known base alias with a backend/node suffix is accepted verbatim
    assert resolve_target("cdx-deep@windows") == ("alias", "cdx-deep@windows")


class _RunProvider:
    def stream(self, prompt, effort, timeout=180, **k):
        raise NotImplementedError   # like base/CLI/remote providers

    def run(self, prompt, effort, timeout=180, **k):
        assert "User: hi" in prompt
        return "one-shot", 0


def _finished_run(text="one-shot", status="completed"):
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

    run = OrchestraRun(mode="ask", task="t", status=RunStatus(status))
    run.agents = [AgentRun(alias="oll-fast", provider="ollama", model="ollama/m",
                           status=AgentStatus.COMPLETED, stdout_log=text)]
    return run


def test_a_plain_reply_goes_through_the_run_pipeline():
    """Fallback, timeout, cost and budget come with the pipeline; a bare provider.run has none."""
    from unittest.mock import AsyncMock

    with patch("orchestra.gateway_dispatch.run_ask_async", new=AsyncMock(return_value=_finished_run())) as run_ask:
        out = asyncio.run(gd.run_target("oll-fast@windows", "User: hi\n\nAssistant:"))

    assert out == "one-shot"
    run_ask.assert_awaited_once_with("oll-fast@windows", "User: hi\n\nAssistant:", internal=True,
                                     show_live=False, emit_console=False)


def test_a_failed_run_is_an_error_not_an_answer():
    from unittest.mock import AsyncMock

    failed = _finished_run("[ERROR] Timeout after 180s", status="failed")
    with patch("orchestra.gateway_dispatch.run_ask_async", new=AsyncMock(return_value=failed)):
        with pytest.raises(RuntimeError):
            asyncio.run(gd.run_target("oll-fast", "User: hi\n\nAssistant:"))


def test_run_target_mode_dual_uses_summary():
    fake_run = MagicMock()
    fake_run.summary = "the synthesis"

    async def _fake_dual(prompt, **k):
        return fake_run

    with patch("orchestra.gateway_dispatch.run_dual_async", _fake_dual):
        out = asyncio.run(gd.run_target("orchestra-dual", "User: go\n\nAssistant:"))
    assert out == "the synthesis"


def test_stream_target_nonstreaming_yields_once():
    async def _collect():
        chunks = []
        async for c in gd.stream_target("oll-fast", "User: hi\n\nAssistant:"):
            chunks.append(c)
        return chunks

    with patch("orchestra.gateway_dispatch.resolve", return_value=(_RunProvider(), "fast")), \
         patch("orchestra.gateway_dispatch.run_ask", return_value=_finished_run()) as run_ask:
        chunks = asyncio.run(_collect())
    assert "".join(chunks) == "one-shot"
    run_ask.assert_called_once()          # a provider that cannot stream answers through the pipeline


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
    monkeypatch.setattr(gd, "resolve", lambda m: (_fake_alias_provider(["a", "b", "c"], started, released), "fast"))

    async def _run():
        agen = gd.stream_target("oll-fast", "hi")
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
    monkeypatch.setattr(gd, "resolve", lambda m: (_unbounded_counting_provider(produced), "fast"))

    async def _run():
        agen = gd.stream_target("oll-fast", "hi")
        await agen.__anext__()       # consume one chunk
        await agen.aclose()          # sets stop → producer must break out of stream()
        n1 = len(produced)
        await asyncio.sleep(0.2)     # a broken producer would add ~hundreds here
        return n1, len(produced)

    n1, n2 = asyncio.run(_run())
    assert n2 - n1 <= 2              # bounded: producer observed stop and stopped


# ---------------------------------------------------------------------------
# Native messages support (Task 4)
# ---------------------------------------------------------------------------

class _NativeProvider:
    supports_native_messages = True
    name = "native-stub"
    def native_messages(self, payload, timeout=180, **kwargs):
        return {"content": [{"type": "text", "text": f"ran {payload['model']}"}],
                "stop_reason": "end_turn", "usage": {}}


class _TextOnlyProvider:
    supports_native_messages = False
    name = "text-stub"


def test_native_capable_true_for_capable_alias():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_NativeProvider(), "fast")):
        assert gd.native_capable("oll-fast") is True


def test_native_capable_false_for_text_provider_and_modes():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_TextOnlyProvider(), "x")):
        assert gd.native_capable("cdx-deep") is False
    assert gd.native_capable("orchestra-dual") is False


def test_run_native_passes_effort_as_model():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_NativeProvider(), "fast")):
        result = asyncio.run(gd.run_native("oll-fast", {"messages": [{"role": "user", "content": "hi"}]}))
    assert result["content"][0]["text"] == "ran fast"


def test_run_native_rejects_non_capable():
    with patch("orchestra.gateway_dispatch.resolve", return_value=(_TextOnlyProvider(), "x")):
        with pytest.raises(NotImplementedError):
            asyncio.run(gd.run_native("cdx-deep", {"messages": []}))


# ---------------------------------------------------------------------------
# What the gateway serves directly is still on the books
# ---------------------------------------------------------------------------

def _gateway_run(task_part: str) -> dict:
    """The one run the gateway recorded for a task containing *task_part*."""
    import json

    from orchestra import config

    runs = [json.loads(p.read_text()) for p in config.artifact_root().glob("*/manifest.json")]
    (run,) = [run for run in runs
              if run.get("parent_run_id") == gd.GATEWAY_PARENT and task_part in run.get("task", "")]
    return run


def test_a_streamed_reply_leaves_a_run_with_its_cost(monkeypatch):
    from orchestra import config

    class _Streaming:
        name = "paid"

        def model_label(self, effort):
            return "paid/model"

        def stream(self, prompt, effort):
            yield from ("streamed ", "answer")

        def last_usage(self):
            return {"prompt_tokens": 2000, "completion_tokens": 1000}

    monkeypatch.setitem(config._cfg, "pricing", {"paid/model": 0.01})

    async def _collect():
        return [chunk async for chunk in gd.stream_target("cld-fast", "User: stream probe\n\nAssistant:")]

    with patch("orchestra.gateway_dispatch.resolve", return_value=(_Streaming(), "fast")):
        assert "".join(asyncio.run(_collect())) == "streamed answer"

    run = _gateway_run("stream probe")
    agent = run["agents"][0]
    assert run["status"] == "completed" and agent["alias"] == "cld-fast"
    assert agent["estimated_cost_usd"] == pytest.approx(0.03)
    assert run["total_cost_usd"] == pytest.approx(0.03)


def test_a_native_tool_call_turn_leaves_a_run_with_the_reported_tokens():
    class _Native:
        name = "ollama"
        supports_native_messages = True

        def model_label(self, effort):
            return "ollama/local-model"

        def native_messages(self, payload, **kwargs):
            return {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 900, "output_tokens": 40}}

    payload = {"messages": [{"role": "user", "content": "native probe"}], "tools": [{"name": "t"}]}

    with patch("orchestra.gateway_dispatch.resolve", return_value=(_Native(), "local-model")):
        asyncio.run(gd.run_native("oll-fast", payload))

    agent = _gateway_run("native probe")["agents"][0]
    assert (agent["prompt_tokens"], agent["estimated_completion_tokens"]) == (900, 40)
    assert agent["estimated_cost_usd"] == 0.0            # a local model
