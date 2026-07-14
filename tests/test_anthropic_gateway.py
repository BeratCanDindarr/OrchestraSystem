from __future__ import annotations

import json
import pytest
from unittest.mock import patch, AsyncMock
from aiohttp.test_utils import TestClient, TestServer

from orchestra import gateway


async def _fake_run_target(model, prompt):
    return f"reply for {model}"


async def _fake_stream(model, prompt):
    for c in ["Hel", "lo"]:
        yield c


@pytest.mark.asyncio
async def test_messages_non_stream():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _fake_run_target):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
            body = await resp.json()
        assert body["type"] == "message" and body["role"] == "assistant"
        assert body["content"][0]["text"] == "reply for oll-fast"
        assert body["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_messages_lenient_unknown_model_uses_default():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="ok")) as mock_run:
            resp = await client.post("/v1/messages", json={
                "model": "claude-3-5-haiku", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
        # unknown model routed to the default alias, not 404
        assert mock_run.call_args.args[0] == "oll-fast"


@pytest.mark.asyncio
async def test_messages_stream_event_sequence():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.stream_target", _fake_stream):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16, "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
            text = await resp.text()
    events = [ln[len("event: "):] for ln in text.splitlines() if ln.startswith("event: ")]
    assert events[0] == "message_start"
    assert events[1] == "content_block_start"
    assert "content_block_delta" in events
    assert events[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    assert "text_delta" in text and "Hel" in text


@pytest.mark.asyncio
async def test_messages_requires_token_when_set():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        with patch("orchestra.gateway_dispatch.run_target", new=AsyncMock(return_value="x")) as mock_run:
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 401
            body = await resp.json()
        assert body["type"] == "error" and body["error"]["type"] == "authentication_error"
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_messages_x_api_key_authorizes():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _fake_run_target):
            resp = await client.post("/v1/messages",
                headers={"x-api-key": "secret"},
                json={"model": "oll-fast", "max_tokens": 16,
                      "messages": [{"role": "user", "content": "hi"}]})
            assert resp.status == 200


@pytest.mark.asyncio
async def test_messages_missing_messages_400():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/messages", json={"model": "oll-fast", "max_tokens": 16})
        assert resp.status == 400
        body = await resp.json()
        assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_messages_run_failure_500_generic():
    async def _explode(model, prompt):
        raise RuntimeError("secret /path token=abc")
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _explode):
            resp = await client.post("/v1/messages", json={
                "model": "oll-fast", "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 500
            raw = await resp.text()
        assert "secret" not in raw and "token=abc" not in raw
        assert '"type": "error"' in raw


@pytest.mark.asyncio
async def test_count_tokens():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/messages/count_tokens", json={
            "model": "oll-fast",
            "messages": [{"role": "user", "content": "hello world"}],
        })
        assert resp.status == 200
        body = await resp.json()
        assert isinstance(body["input_tokens"], int) and body["input_tokens"] >= 0
