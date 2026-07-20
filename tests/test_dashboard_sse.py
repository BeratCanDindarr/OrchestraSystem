"""Tests for the dashboard SSE live stream + bus bridge (Faz 6 Block C)."""
import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from orchestra.monitoring import dashboard_app


@pytest.mark.asyncio
async def test_bus_handler_bridges_to_queue():
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = dashboard_app._make_bus_handler(loop, queue)
    handler("r1", {"event": "agent_completed"})  # simulates a bus-thread call
    run_id, event = await asyncio.wait_for(queue.get(), timeout=2)
    assert run_id == "r1"
    assert event["event"] == "agent_completed"


@pytest.mark.asyncio
async def test_stream_pushes_bus_events():
    from orchestra.events import bus
    async with TestClient(TestServer(dashboard_app.make_app())) as client:
        resp = await client.get("/api/stream")
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        bus.emit("r1", {"event": "agent_completed"})
        chunk = await asyncio.wait_for(resp.content.readuntil(b"\n\n"), timeout=3)
        assert b"run_update" in chunk
        assert b"agent_completed" in chunk
        resp.close()


@pytest.mark.asyncio
async def test_stream_unsubscribes_on_disconnect():
    """Regression (review): closing the SSE connection must remove the bus subscriber
    (no leaked handlers accumulating across reconnects)."""
    from orchestra.events import bus
    baseline = len(bus._global)
    async with TestClient(TestServer(dashboard_app.make_app())) as client:
        resp = await client.get("/api/stream")
        bus.emit("r1", {"event": "agent_completed"})
        await asyncio.wait_for(resp.content.readuntil(b"\n\n"), timeout=3)
        assert len(bus._global) == baseline + 1   # subscribed while streaming
        resp.close()
    # TestClient context exit closes the server → the stream handler's finally runs
    await asyncio.sleep(0.05)
    assert len(bus._global) == baseline           # unsubscribed on disconnect


@pytest.mark.asyncio
async def test_character_chat_stream(monkeypatch):
    from orchestra.monitoring import characters
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: True)

    async def fake_stream(cid, msg):
        for chunk in ["Hel", "lo"]:
            yield chunk
    monkeypatch.setattr(characters, "chat_stream", fake_stream)

    async with TestClient(TestServer(dashboard_app.make_app())) as client:
        resp = await client.get("/api/character/cdx-deep/chat/stream", params={"message": "hi"})
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        body = b""
        # read until we see the done event
        for _ in range(5):
            body += await asyncio.wait_for(resp.content.readuntil(b"\n\n"), timeout=3)
            if b"event: done" in body:
                break
        resp.close()
    assert b"event: token" in body and b"Hel" in body and b"lo" in body
    assert b"event: done" in body


@pytest.mark.asyncio
async def test_character_chat_stream_gated(monkeypatch):
    monkeypatch.setattr(dashboard_app.config, "dashboard_dispatch_enabled", lambda: False)
    async with TestClient(TestServer(dashboard_app.make_app())) as client:
        resp = await client.get("/api/character/cdx-deep/chat/stream", params={"message": "hi"})
        body = await asyncio.wait_for(resp.content.readuntil(b"\n\n"), timeout=3)
        resp.close()
    assert b"disabled" in body


@pytest.mark.asyncio
async def test_dispatch_live_stream(monkeypatch):
    from orchestra.monitoring import dispatch_inbox
    box = dispatch_inbox.DispatchInbox()
    did = box.add("a", "t")
    box.complete(did, "Hello world", ok=True)   # finished → stream sends output + done
    monkeypatch.setattr(dispatch_inbox, "get_inbox", lambda: box)
    async with TestClient(TestServer(dashboard_app.make_app())) as client:
        resp = await client.get(f"/api/dispatch/{did}/stream")
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        body = b""
        for _ in range(5):
            body += await asyncio.wait_for(resp.content.readuntil(b"\n\n"), timeout=3)
            if b"event: done" in body:
                break
        resp.close()
    assert b"event: token" in body and b"Hello world" in body and b"event: done" in body
