"""Tests for the agent gateway HTTP server (Task 4).

Uses aiohttp.test_utils.TestClient + TestServer directly (pytest-aiohttp is not installed).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from aiohttp.test_utils import TestClient, TestServer

from orchestra import gateway


class _FakeRun:
    def __init__(self):
        agent = MagicMock()
        agent.stdout_log = "remote output"
        agent.model = "gpt/5.4"
        agent.estimated_cost_usd = 0.01
        self.agents = [agent]
        self.status = MagicMock(value="completed")


@pytest.mark.asyncio
async def test_run_requires_token_when_configured():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        resp = await client.post("/run", json={"alias": "cdx-deep", "task": "x"})
        assert resp.status == 401


@pytest.mark.asyncio
async def test_run_executes_and_returns_stdout():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        with patch(
            "orchestra.gateway.run_ask_async",
            new=AsyncMock(return_value=_FakeRun()),
        ) as mock_ask:
            resp = await client.post(
                "/run",
                json={"alias": "cdx-deep", "task": "do it"},
                headers={"Authorization": "Bearer secret"},
            )
        assert resp.status == 200
        body = await resp.json()
        assert body["stdout"] == "remote output"
        assert body["status"] == "completed"
        mock_ask.assert_called_once_with("cdx-deep", "do it")


@pytest.mark.asyncio
async def test_health_needs_no_auth():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_aliases_lists_local():
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        with patch("orchestra.gateway.available_aliases", return_value=["✓ cdx-deep"]):
            resp = await client.get("/aliases")
        assert resp.status == 200
        body = await resp.json()
        assert "aliases" in body


@pytest.mark.asyncio
async def test_aliases_requires_token_when_configured():
    async with TestClient(TestServer(gateway.make_app(token="secret"))) as client:
        # No auth header
        resp = await client.get("/aliases")
        assert resp.status == 401
        # Wrong token
        resp = await client.get(
            "/aliases", headers={"Authorization": "Bearer wrong"}
        )
        assert resp.status == 401


def test_serve_refuses_public_bind_without_token():
    with pytest.raises(ValueError):
        gateway.serve(host="0.0.0.0", port=7070, token="")


@pytest.mark.asyncio
async def test_run_rejects_non_dict_json_body():
    """A valid-but-non-object JSON body (e.g. a list) must give a clean 400, not a 500."""
    async with TestClient(TestServer(gateway.make_app(token=""))) as client:
        resp = await client.post("/run", json=[1, 2, 3])
        assert resp.status == 400
