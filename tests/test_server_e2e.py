"""End-to-end tests for Orchestra WebSocket daemon."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import socket
import threading
from typing import Any, Dict

import pytest
import websockets

from orchestra import server as srv


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@dataclasses.dataclass
class ServerInfo:
    uri: str
    _server_obj: Dict[str, Any]
    _thread: threading.Thread

    def stop(self) -> None:
        loop = self._server_obj["loop"]
        stop_event = self._server_obj["stop_event"]
        loop.call_soon_threadsafe(stop_event.set)
        self._thread.join(timeout=5)


def _start_server(host: str, port: int) -> tuple[Dict[str, Any], threading.Thread]:
    loop = asyncio.new_event_loop()
    server_obj: Dict[str, Any] = {"loop": loop}

    async def _runner() -> None:
        srv.SERVER_LOOP = loop
        srv.CONNECTED_CLIENTS.clear()
        stop_event = asyncio.Event()
        server_obj["stop_event"] = stop_event
        server = await websockets.serve(srv.server_loop, host, port)
        server_obj["server"] = server
        try:
            await stop_event.wait()
        finally:
            server.close()
            await server.wait_closed()
            srv.CONNECTED_CLIENTS.clear()
            srv.SERVER_LOOP = None

    def _thread_target() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_runner())

    thread = threading.Thread(target=_thread_target, daemon=True)
    thread.start()

    # Wait briefly for the server to be ready
    import time
    for _ in range(100):
        if "server" in server_obj:
            break
        time.sleep(0.01)

    return server_obj, thread


@pytest.fixture
def ws_server():
    host = "127.0.0.1"
    port = _free_port()
    server_obj, thread = _start_server(host, port)
    info = ServerInfo(uri=f"ws://{host}:{port}", _server_obj=server_obj, _thread=thread)
    yield info
    if thread.is_alive():
        info.stop()


@pytest.mark.asyncio
async def test_daemon_lifecycle(ws_server: ServerInfo) -> None:
    """Start server, connect, verify handshake, subscribe, and stop."""
    async with websockets.connect(ws_server.uri) as ws:
        # Handshake: server pushes history, slots, and toolbox on connect
        msgs = []
        for _ in range(3):
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            msgs.append(json.loads(raw))

        types = {m["type"] for m in msgs}
        assert "history" in types
        assert "slots" in types
        assert "toolbox" in types

        # Subscribe to a run
        await ws.send(json.dumps({"type": "subscribe", "payload": {"run_id": "test-run"}}))
        raw = await asyncio.wait_for(ws.recv(), timeout=2)
        ack = json.loads(raw)
        assert ack["type"] == "ack"
        assert "test-run" in ack["message"]


@pytest.mark.asyncio
async def test_run_status_streaming(ws_server: ServerInfo) -> None:
    """Broadcast a run update and verify the client receives it."""
    async with websockets.connect(ws_server.uri) as ws:
        # Consume handshake messages
        for _ in range(3):
            await asyncio.wait_for(ws.recv(), timeout=2)

        mock_run = {
            "run_id": "run-123",
            "status": "running",
            "mode": "ask",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        srv._schedule_broadcast({"type": "history", "runs": [mock_run]})

        raw = await asyncio.wait_for(ws.recv(), timeout=2)
        data = json.loads(raw)
        assert data["type"] == "history"
        assert len(data["runs"]) == 1
        assert data["runs"][0]["run_id"] == "run-123"
        assert data["runs"][0]["status"] == "running"


@pytest.mark.asyncio
async def test_malformed_json(ws_server: ServerInfo) -> None:
    """Server responds with an error to malformed JSON and remains responsive."""
    async with websockets.connect(ws_server.uri) as ws:
        for _ in range(3):
            await asyncio.wait_for(ws.recv(), timeout=2)

        await ws.send("this is not json")
        raw = await asyncio.wait_for(ws.recv(), timeout=2)
        data = json.loads(raw)
        assert data["type"] == "error"
        assert "json" in data["message"].lower()

        # Verify the connection is still usable
        await ws.send(json.dumps({"type": "ping"}))
        raw = await asyncio.wait_for(ws.recv(), timeout=2)
        assert json.loads(raw)["type"] == "pong"


@pytest.mark.asyncio
async def test_server_stop_closes_clients(ws_server: ServerInfo) -> None:
    """Stopping the server sends close frames to connected clients."""
    ws = await websockets.connect(ws_server.uri)
    try:
        for _ in range(3):
            await asyncio.wait_for(ws.recv(), timeout=2)

        ws_server.stop()

        # After server stops, recv should raise because the connection is closed
        with pytest.raises(websockets.ConnectionClosed):
            await ws.recv()
    finally:
        await ws.close()
