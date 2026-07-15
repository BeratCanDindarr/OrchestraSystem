from __future__ import annotations

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from aiohttp.test_utils import TestClient, TestServer

from orchestra import gateway

_TOOLS = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]
_NATIVE_RESULT = {
    "content": [{"type": "text", "text": "running"},
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"cmd": "ls"}}],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 9, "output_tokens": 4},
}


def _body(stream=False):
    b = {"model": "oll-fast", "max_tokens": 64, "tools": _TOOLS,
         "messages": [{"role": "user", "content": "list files"}]}
    if stream:
        b["stream"] = True
    return b


@pytest.mark.asyncio
async def test_tools_capable_uses_native_path():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
             patch("orchestra.gateway_dispatch.run_native", new=AsyncMock(return_value=_NATIVE_RESULT)) as mock_native:
            resp = await client.post("/v1/messages", json=_body())
            assert resp.status == 200
            body = await resp.json()
        assert mock_native.call_args.args[0] == "oll-fast"
        assert mock_native.call_args.args[1]["tools"] == _TOOLS
        assert body["stop_reason"] == "tool_use"
        assert body["content"][1]["type"] == "tool_use"


@pytest.mark.asyncio
async def test_tools_non_capable_falls_back_to_text_path():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=False), \
             patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="plain")) as mock_text:
            resp = await client.post("/v1/messages", json=_body())
            assert resp.status == 200
            body = await resp.json()
        mock_text.assert_called_once()
        assert body["content"][0]["text"] == "plain"


@pytest.mark.asyncio
async def test_no_tools_never_checks_native():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable") as mock_cap, \
             patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")):
            b = _body(); del b["tools"]
            resp = await client.post("/v1/messages", json=b)
            assert resp.status == 200
        mock_cap.assert_not_called()


@pytest.mark.asyncio
async def test_native_stream_event_sequence_with_tool_use():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
             patch("orchestra.gateway_dispatch.run_native", new=AsyncMock(return_value=_NATIVE_RESULT)):
            resp = await client.post("/v1/messages", json=_body(stream=True))
            assert resp.status == 200
            text = await resp.text()
    events = [ln[len("event: "):] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_stop",   # text block 0
        "content_block_start", "content_block_delta", "content_block_stop",   # tool_use block 1
        "message_delta", "message_stop",
    ]
    assert '"input_json_delta"' in text and '"partial_json"' in text
    assert '"stop_reason": "tool_use"' in text


@pytest.mark.asyncio
async def test_native_run_failure_500_generic():
    async def _boom(model, payload):
        raise RuntimeError("secret /path token=abc")
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
             patch("orchestra.gateway_dispatch.run_native", _boom):
            resp = await client.post("/v1/messages", json=_body())
            assert resp.status == 500
            raw = await resp.text()
        assert "secret" not in raw and "token=abc" not in raw


@pytest.mark.asyncio
async def test_native_stream_emits_ping_during_slow_run():
    """A slow buffered run must emit keep-alive ping events before the content blocks,
    so the client's idle timeout doesn't fire during a long local-model generation."""
    async def _slow(model, payload):
        await asyncio.sleep(0.25)
        return _NATIVE_RESULT
    with patch.object(gateway, "_NATIVE_PING_INTERVAL", 0.05):
        async with TestClient(TestServer(gateway.make_app(token=""))) as client:
            with patch("orchestra.gateway_dispatch.native_capable", return_value=True), \
                 patch("orchestra.gateway_dispatch.run_native", _slow):
                resp = await client.post("/v1/messages", json=_body(stream=True))
                assert resp.status == 200
                text = await resp.text()
    events = [ln[len("event: "):] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events[0] == "message_start"
    assert "ping" in events                                   # keep-alive fired
    assert events[-1] == "message_stop"
    assert events.index("ping") < events.index("content_block_start")  # pings before content
