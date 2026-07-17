"""Tests for A2A Agent Card builder + gateway endpoint (Faz 5 Block A/B)."""
import pytest
from aiohttp.test_utils import TestClient, TestServer

from orchestra import config, gateway
from orchestra.engine.agent_card import build_agent_card


def test_build_agent_card_shape():
    card = build_agent_card(
        host="h", version="1.0.0", url="http://h/",
        aliases=["cdx-deep", "oll-fast"],
        mcp_tools=[{"name": "orchestra_run", "description": "d"}],
    )
    assert card["name"] == "orchestra@h"
    assert card["protocolVersion"] == "0.2.0"
    assert card["version"] == "1.0.0"
    assert len(card["skills"]) == 2
    assert card["skills"][0]["id"] == "cdx-deep"
    assert card["mcpTools"][0]["name"] == "orchestra_run"


def test_build_agent_card_without_mcp_tools():
    card = build_agent_card(host="h", version="1.0.0", url="http://h/", aliases=["a"])
    assert "mcpTools" not in card


@pytest.mark.asyncio
async def test_gateway_serves_agent_card():
    async with TestClient(TestServer(gateway.make_app())) as client:
        resp = await client.get("/.well-known/agent.json")
        assert resp.status == 200
        card = await resp.json()
    assert card["name"].startswith("orchestra@")
    assert isinstance(card["skills"], list)


@pytest.mark.asyncio
async def test_gateway_card_omits_mcp_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "mcp_enabled", lambda: False)
    async with TestClient(TestServer(gateway.make_app())) as client:
        resp = await client.get("/.well-known/agent.json")
        card = await resp.json()
    assert "mcpTools" not in card


@pytest.mark.asyncio
async def test_gateway_card_includes_mcp_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "mcp_enabled", lambda: True)
    monkeypatch.setattr(
        "orchestra.engine.mcp_registry.REGISTRY.list_tools",
        lambda: [{"name": "t", "description": "d", "inputSchema": {}}],
    )
    async with TestClient(TestServer(gateway.make_app())) as client:
        resp = await client.get("/.well-known/agent.json")
        card = await resp.json()
    assert card["mcpTools"][0]["name"] == "t"
