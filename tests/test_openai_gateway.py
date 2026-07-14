# tests/test_openai_gateway.py
import json
import pytest
from unittest.mock import patch
from aiohttp.test_utils import TestServer, TestClient
from orchestra import gateway


@pytest.mark.asyncio
async def test_models_lists(monkeypatch):
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.openai_compat.list_models", return_value=[{"id": "oll-fast", "object": "model"}]):
            resp = await client.get("/v1/models")
        assert resp.status == 200
        body = await resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "oll-fast"


@pytest.mark.asyncio
async def test_chat_completions_non_stream():
    async def _fake_run_target(model, prompt):
        assert model == "oll-fast"
        return "hello world"
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _fake_run_target):
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status == 200
        body = await resp.json()
        assert body["choices"][0]["message"]["content"] == "hello world"


@pytest.mark.asyncio
async def test_chat_completions_stream_sse():
    async def _fake_stream(model, prompt):
        for c in ["hel", "lo"]:
            yield c
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.stream_target", _fake_stream):
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast", "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status == 200
            text = await resp.text()
    assert "data: " in text
    assert "[DONE]" in text
    assert "hel" in text and "lo" in text


@pytest.mark.asyncio
async def test_chat_completions_unknown_model_404():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/chat/completions", json={
            "model": "orchestra-bogus",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status == 404
        body = await resp.json()
        assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_chat_completions_missing_messages_400():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/v1/chat/completions", json={"model": "oll-fast"})
    assert resp.status == 400


@pytest.mark.asyncio
async def test_v1_requires_token_when_set():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        resp = await client.get("/v1/models")
    assert resp.status == 401


@pytest.mark.asyncio
async def test_v1_chat_requires_token_when_set():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        with patch("orchestra.gateway_dispatch.run_target") as mock_run:
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status == 401
        mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_non_stream_500_does_not_leak_exception():
    """I1: raw exception text (including PII) must not reach the client on 500."""
    async def _explode(model, prompt):
        raise Exception("secret /path/to/thing token=abc")

    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway_dispatch.run_target", _explode):
            resp = await client.post("/v1/chat/completions", json={
                "model": "oll-fast",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status == 500
        body = await resp.text()
    assert "secret" not in body
    assert "token=abc" not in body
