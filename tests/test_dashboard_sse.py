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
