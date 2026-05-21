"""Integration tests for federation HTTP layer (F3.2)."""
from __future__ import annotations

import pytest
import httpx
from pathlib import Path

from orchestra.federation_server.federation_http import create_app
from orchestra.federation_server.federation_client import push_to_peer
from orchestra.engine.federation import FederationBus, FederationPeer, FederationSummary
from orchestra.engine.federation_crdt import FederatedState, LWWRegister
from orchestra import config


class TestFederationHTTPServer:
    """In-memory server endpoint tests."""

    @pytest.fixture(autouse=True)
    def _no_auth(self):
        config.set("federation", "auth_token", "")
        yield

    @pytest.mark.asyncio
    async def test_health(self, aiohttp_server) -> None:
        app = create_app()
        server = await aiohttp_server(app)
        async with httpx.AsyncClient() as client:
            resp = await client.get(str(server.make_url("/federation/health")))
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "orchestra-federation"

    @pytest.mark.asyncio
    async def test_push_and_merged_state(self, aiohttp_server, tmp_path: Path) -> None:
        bus = FederationBus(root=tmp_path / "fed")
        app = create_app()
        app["federation_bus"] = bus
        server = await aiohttp_server(app)

        summary = FederationSummary(
            content_hash="hash1",
            embedding_digest="emb1",
            crdt_state=FederatedState(content=LWWRegister("hello federation", 1.0, "peer-a")),
            peer_id="peer-a",
        )

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                str(server.make_url("/federation/push")),
                json=[summary.to_dict()],
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1
        assert data["total"] == 1

        # Verify merged state is queryable
        async with httpx.AsyncClient() as client:
            resp = await client.get(str(server.make_url("/federation/query?q=hello&limit=5")))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        assert data["results"][0]["content_hash"] == "hash1"
        assert data["results"][0]["score"] > 0

    @pytest.mark.asyncio
    async def test_push_invalid_payload(self, aiohttp_server) -> None:
        app = create_app()
        server = await aiohttp_server(app)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                str(server.make_url("/federation/push")),
                content="this is not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 400
        assert "error" in resp.json()


class TestTwoInstanceSync:
    """Two-instance federation sync over HTTP."""

    @pytest.fixture(autouse=True)
    def _no_auth(self):
        config.set("federation", "auth_token", "")
        yield

    @pytest.mark.asyncio
    async def test_instance_a_syncs_to_b(self, aiohttp_server, tmp_path: Path) -> None:
        # Two isolated Orchestra instances with temporary roots (simulating :memory: DBs)
        root_a = tmp_path / "instance_a"
        root_b = tmp_path / "instance_b"
        root_a.mkdir()
        root_b.mkdir()

        bus_a = FederationBus(root=root_a / "federation")
        bus_b = FederationBus(root=root_b / "federation")

        app_a = create_app()
        app_a["federation_bus"] = bus_a
        app_b = create_app()
        app_b["federation_bus"] = bus_b

        server_a = await aiohttp_server(app_a)
        server_b = await aiohttp_server(app_b)

        url_b = str(server_b.make_url(""))

        # Instance A writes a memory entry to its outbox
        summary = FederationSummary(
            content_hash="sync-hash-42",
            embedding_digest="sync-emb-42",
            crdt_state=FederatedState(content=LWWRegister("distributed knowledge graph", 1.0, "peer-a")),
            peer_id="peer-a",
            metadata={"mode": "dual", "agent_count": 2},
        )
        bus_a.enqueue_outbox(summary)

        # Instance A syncs to Instance B via HTTP
        peer_b = FederationPeer(peer_id="peer-b", endpoint_url=url_b)
        outbox_a = bus_a.dequeue_outbox()
        result = await push_to_peer(peer_b, outbox_a)
        assert result.get("accepted") == 1

        # Instance B can search and find the anonymized summary
        async with httpx.AsyncClient() as client:
            resp = await client.get(str(server_b.make_url("/federation/query?q=distributed&limit=5")))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["content_hash"] == "sync-hash-42"
        assert data["results"][0]["peer_id"] == "peer-a"

    @pytest.mark.asyncio
    async def test_raw_content_never_leaves_local_instance(self, aiohttp_server, tmp_path: Path) -> None:
        root_a = tmp_path / "instance_a"
        root_b = tmp_path / "instance_b"
        root_a.mkdir()
        root_b.mkdir()

        bus_a = FederationBus(root=root_a / "federation")
        bus_b = FederationBus(root=root_b / "federation")

        app_a = create_app()
        app_a["federation_bus"] = bus_a
        app_b = create_app()
        app_b["federation_bus"] = bus_b

        server_a = await aiohttp_server(app_a)
        server_b = await aiohttp_server(app_b)

        # Build a summary that only contains anonymized fields
        summary = FederationSummary(
            content_hash="abc123hash",
            embedding_digest="emb456digest",
            crdt_state=FederatedState(content=LWWRegister("anonymized summary of null ref fix", 1.0, "peer-a")),
            peer_id="peer-a",
            metadata={"agent_count": 2, "mode": "dual"},
        )
        bus_a.enqueue_outbox(summary)

        # Sync to B
        peer_b = FederationPeer(peer_id="peer-b", endpoint_url=str(server_b.make_url("")))
        await push_to_peer(peer_b, bus_a.dequeue_outbox())

        # Inspect what B received
        inbox_b = bus_b.read_inbox()
        assert len(inbox_b) == 1
        received = inbox_b[0]

        # Verify no raw content leaked across the wire
        payload = received.to_dict()
        assert "raw_content" not in payload
        assert "stdout_log" not in payload
        assert "stderr_log" not in payload
        # content_hash is a digest, not the raw text
        assert received.content_hash == "abc123hash"
        # The CRDT content is the anonymized summary, not raw logs
        assert "anonymized" in (received.crdt_state.content.value or "")


class TestCRDTConflictResolution:
    """CRDT LWW conflict resolution across two federated instances."""

    @pytest.fixture(autouse=True)
    def _no_auth(self):
        config.set("federation", "auth_token", "")
        yield

    @pytest.mark.asyncio
    async def test_concurrent_update_lww(self, aiohttp_server, tmp_path: Path) -> None:
        root_a = tmp_path / "instance_a"
        root_b = tmp_path / "instance_b"
        root_a.mkdir()
        root_b.mkdir()

        bus_a = FederationBus(root=root_a / "federation")
        bus_b = FederationBus(root=root_b / "federation")

        app_a = create_app()
        app_a["federation_bus"] = bus_a
        app_b = create_app()
        app_b["federation_bus"] = bus_b

        server_a = await aiohttp_server(app_a)
        server_b = await aiohttp_server(app_b)

        # Both instances update the same key concurrently with different timestamps
        summary_a = FederationSummary(
            content_hash="shared-key",
            embedding_digest="emb-shared",
            crdt_state=FederatedState(content=LWWRegister("version from A", 1.0, "peer-a")),
            peer_id="peer-a",
        )
        summary_b = FederationSummary(
            content_hash="shared-key",
            embedding_digest="emb-shared",
            crdt_state=FederatedState(content=LWWRegister("version from B", 2.0, "peer-b")),
            peer_id="peer-b",
        )

        bus_a.enqueue_outbox(summary_a)
        bus_b.enqueue_outbox(summary_b)

        # Sync A -> B
        await push_to_peer(
            FederationPeer(peer_id="peer-a", endpoint_url=str(server_b.make_url(""))),
            bus_a.dequeue_outbox(),
        )

        # Sync B -> A
        await push_to_peer(
            FederationPeer(peer_id="peer-b", endpoint_url=str(server_a.make_url(""))),
            bus_b.dequeue_outbox(),
        )

        # Verify what each inbox contains
        inbox_a = bus_a.read_inbox()
        inbox_b = bus_b.read_inbox()
        assert len(inbox_a) == 1
        assert len(inbox_b) == 1

        # A received B's state (newer timestamp)
        # B received A's state (older timestamp)
        received_by_a = inbox_a[0].crdt_state
        received_by_b = inbox_b[0].crdt_state

        # Verify LWW resolution: when local and received states are merged,
        # the newer timestamp (2.0 from peer-b) wins in both cases.
        merged_a = summary_a.crdt_state.merge(received_by_a)
        merged_b = summary_b.crdt_state.merge(received_by_b)

        assert merged_a.content.value == "version from B"
        assert merged_b.content.value == "version from B"
        assert merged_a.content.timestamp == 2.0
        assert merged_b.content.timestamp == 2.0


class TestAuthToken:
    """Bearer token authentication on federation endpoints."""

    @pytest.fixture(autouse=True)
    def _auth_token(self):
        original = config.federation_config().get("auth_token", "")
        config.set("federation", "auth_token", "test-secret-token")
        yield
        config.set("federation", "auth_token", original)

    @pytest.mark.asyncio
    async def test_reject_without_token(self, aiohttp_server) -> None:
        app = create_app()
        server = await aiohttp_server(app)
        async with httpx.AsyncClient() as client:
            resp = await client.post(str(server.make_url("/federation/push")), json=[])
        assert resp.status_code == 401
        assert resp.json()["error"] == "Unauthorized"

    @pytest.mark.asyncio
    async def test_accept_with_correct_token(self, aiohttp_server) -> None:
        app = create_app()
        server = await aiohttp_server(app)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                str(server.make_url("/federation/push")),
                json=[],
                headers={"Authorization": "Bearer test-secret-token"},
            )
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 0
