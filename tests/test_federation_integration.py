"""Integration tests for federation: summarize, bus, HTTP server/client."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import asyncio

import pytest
import pytest_asyncio

from aiohttp import web

from orchestra.engine.federation import FederationBus, FederationPeer, FederationSummary
from orchestra.engine.federation_crdt import FederatedState, GCounter, LWWRegister, ORSet
from orchestra.engine.federation_summarizer import summarize_run
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.federation_server.federation_http import create_app
from orchestra.federation_server.federation_client import push_to_peer, query_peer, sync_with_peer


@pytest_asyncio.fixture
async def client(aiohttp_client):
    app = create_app()
    return await aiohttp_client(app)


@pytest_asyncio.fixture
async def client_session(aiohttp_server):
    app = create_app()
    server = await aiohttp_server(app)
    return server


class TestSummarizer:
    def test_summarize_run_basic(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="how to handle null refs")
        run.agents.append(
            AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED, stdout_log="output a", confidence=0.9)
        )
        run.agents.append(
            AgentRun(alias="gmn-pro", provider="gemini", model="m", status=AgentStatus.COMPLETED, stdout_log="output b", confidence=0.8)
        )
        summary = summarize_run(run, peer_id="test-peer")
        assert summary is not None
        assert summary.peer_id == "test-peer"
        assert summary.metadata["agent_count"] == 2
        assert summary.metadata["mode"] == "dual"
        # Raw stdout should never appear
        assert "output a" not in summary.crdt_state.content.value or ""

    def test_summarize_run_failed_run_returns_none(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="task")
        run.agents.append(
            AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.FAILED)
        )
        assert summarize_run(run) is None

    def test_summarize_run_empty_task_returns_none(self) -> None:
        run = OrchestraRun(run_id="r1", mode="dual", task="")
        run.agents.append(
            AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.COMPLETED, stdout_log="x")
        )
        assert summarize_run(run) is None


class TestFederationBus:
    def test_enqueue_and_dequeue_outbox(self, tmp_path: Path) -> None:
        bus = FederationBus(root=tmp_path / "fed")
        s = FederationSummary(content_hash="abc", embedding_digest="def", peer_id="p1")
        bus.enqueue_outbox(s)
        items = bus.dequeue_outbox()
        assert len(items) == 1
        assert items[0].content_hash == s.content_hash

    def test_clear_outbox(self, tmp_path: Path) -> None:
        bus = FederationBus(root=tmp_path / "fed")
        s = FederationSummary(content_hash="abc", embedding_digest="def")
        bus.enqueue_outbox(s)
        bus.clear_outbox(["abc"])
        assert bus.dequeue_outbox() == []

    def test_write_and_read_inbox(self, tmp_path: Path) -> None:
        bus = FederationBus(root=tmp_path / "fed")
        s = FederationSummary(content_hash="xyz", embedding_digest="uvw", peer_id="p2")
        bus.write_inbox(s)
        items = bus.read_inbox()
        assert len(items) == 1
        assert items[0].peer_id == "p2"

    def test_stats(self, tmp_path: Path) -> None:
        bus = FederationBus(root=tmp_path / "fed")
        bus.enqueue_outbox(FederationSummary(content_hash="a", embedding_digest="b"))
        bus.write_inbox(FederationSummary(content_hash="c", embedding_digest="d"))
        stats = bus.stats()
        assert stats["outbox_count"] == 1
        assert stats["inbox_count"] == 1


class TestFederationHTTP:
    @pytest.mark.asyncio
    async def test_health(self, client) -> None:
        resp = await client.get("/federation/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_push_and_query(self, client) -> None:
        s = FederationSummary(
            content_hash="hash1",
            embedding_digest="emb1",
            crdt_state=FederatedState(content=LWWRegister("hello world")),
            peer_id="peer-a",
        )
        resp = await client.post("/federation/push", json=[s.to_dict()])
        assert resp.status == 200
        data = await resp.json()
        assert data["accepted"] == 1

        resp = await client.get("/federation/query?q=hello&limit=5")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["results"]) >= 1

    @pytest.mark.asyncio
    async def test_status(self, client) -> None:
        resp = await client.get("/federation/status")
        assert resp.status == 200
        data = await resp.json()
        assert "outbox_count" in data


class TestFederationClient:
    @pytest.mark.asyncio
    async def test_push_to_peer(self, client_session) -> None:
        peer = FederationPeer(peer_id="test", endpoint_url=str(client_session.make_url("")))
        s = FederationSummary(content_hash="h1", embedding_digest="e1", peer_id="local")
        result = await push_to_peer(peer, [s])
        assert "accepted" in result
        assert result["accepted"] == 1

    @pytest.mark.asyncio
    async def test_query_peer(self, client_session) -> None:
        peer = FederationPeer(peer_id="test", endpoint_url=str(client_session.make_url("")))
        # Pre-seed inbox
        s = FederationSummary(
            content_hash="h2",
            embedding_digest="e2",
            crdt_state=FederatedState(content=LWWRegister("python typing")),
            peer_id="remote",
        )
        bus = FederationBus()
        bus.write_inbox(s)

        results = await query_peer(peer, "python typing", limit=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_sync_with_peer(self, client_session, tmp_path: Path) -> None:
        peer = FederationPeer(peer_id="test", endpoint_url=str(client_session.make_url("")))
        bus = FederationBus(root=tmp_path / "fed")
        s = FederationSummary(content_hash="h3", embedding_digest="e3", peer_id="local")
        bus.enqueue_outbox(s)

        result = await sync_with_peer(peer)
        assert "pushed" in result
        assert result["outbox_remaining"] == 0
