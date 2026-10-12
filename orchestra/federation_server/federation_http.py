from __future__ import annotations
"""aiohttp companion server for federation endpoints.

Mounts alongside the existing WebSocket server in the same asyncio loop.
"""

import json
import logging
from typing import Any

from aiohttp import web

from orchestra import config
from orchestra.engine.federation import FederationBus, FederationSummary
from orchestra.engine.federation_crdt import FederatedState
from orchestra.utils import browser_guard

logger = logging.getLogger(__name__)


def _auth_token() -> str:
    return config.federation_config().get("auth_token", "")


def _check_auth(request: web.Request) -> bool:
    expected = _auth_token()
    if not expected:
        return True  # No auth configured
    header = request.headers.get("Authorization", "")
    parts = header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == expected:
        return True
    return False


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "orchestra-federation"})


async def push_handler(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    summaries = payload if isinstance(payload, list) else [payload]
    bus = request.app.get("federation_bus") or FederationBus()
    accepted = 0
    for item in summaries:
        try:
            summary = FederationSummary.from_dict(item)
            bus.write_inbox(summary)
            accepted += 1
        except Exception as exc:
            logger.warning("Failed to ingest federated summary: %s", exc)

    return web.json_response({"accepted": accepted, "total": len(summaries)})


async def query_handler(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)

    q = request.query.get("q", "").strip()
    limit = int(request.query.get("limit", 5))
    if not q:
        return web.json_response({"error": "Missing query parameter 'q'"}, status=400)

    # Search local inbox summaries by simple keyword overlap (no embeddings across instances)
    bus = request.app.get("federation_bus") or FederationBus()
    inbox = bus.read_inbox()
    results: list[dict[str, Any]] = []
    q_tokens = set(q.lower().split())
    for summary in inbox:
        content = summary.crdt_state.content.value or ""
        c_tokens = set(content.lower().split())
        score = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
        if score > 0.0:
            results.append({
                "content_hash": summary.content_hash,
                "embedding_digest": summary.embedding_digest,
                "metadata": summary.metadata,
                "peer_id": summary.peer_id,
                "score": round(score, 4),
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return web.json_response({"query": q, "results": results[:limit]})


async def peers_handler(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)
    peers = config.federation_config().get("peer_urls", [])
    return web.json_response({"peers": peers})


async def status_handler(request: web.Request) -> web.Response:
    if not _check_auth(request):
        return web.json_response({"error": "Unauthorized"}, status=401)
    bus = request.app.get("federation_bus") or FederationBus()
    return web.json_response(bus.stats())


def attach_sync_daemon(app: web.Application) -> web.Application:
    """Start/stop the background federation sync daemon with the app lifecycle.

    The daemon only runs when ``config.federation_sync_enabled()`` — otherwise the
    startup hook is a no-op. The task is cancelled on cleanup to avoid leaks.
    """
    import asyncio

    async def _on_startup(app: web.Application) -> None:
        if config.federation_sync_enabled():
            from orchestra.engine.federation_daemon import run_sync_loop
            app["_sync_task"] = asyncio.create_task(run_sync_loop())

    async def _on_cleanup(app: web.Application) -> None:
        task = app.get("_sync_task")
        if task is not None:
            task.cancel()
            try:
                await task  # let cancellation propagate + the loop's finally run
            except asyncio.CancelledError:
                pass

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def create_app() -> web.Application:
    app = web.Application(middlewares=[browser_guard.middleware()])
    app.router.add_get("/federation/health", health_handler)
    app.router.add_post("/federation/push", push_handler)
    app.router.add_get("/federation/query", query_handler)
    app.router.add_get("/federation/peers", peers_handler)
    app.router.add_get("/federation/status", status_handler)
    attach_sync_daemon(app)
    return app


def run_app(host: str = "127.0.0.1", port: int = 8766) -> None:
    """Run the federation HTTP server (blocking)."""
    app = create_app()
    web.run_app(app, host=host, port=port)
