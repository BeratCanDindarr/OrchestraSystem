"""Agent gateway: expose this node's run_ask over HTTP so a remote Orchestra can use it.

Small aiohttp server (see federation_http.py for the pattern). Wraps run_ask; adds nothing.
"""
from __future__ import annotations

import hmac
import json
import logging
import socket

from aiohttp import web

from orchestra.engine.runners.common import run_ask_async
from orchestra.providers.fallback import available_aliases

logger = logging.getLogger(__name__)


def _authorized(request: web.Request, token: str) -> bool:
    if not token:
        return True
    header = request.headers.get("Authorization", "")
    parts = header.split()
    return (
        len(parts) == 2
        and parts[0].lower() == "bearer"
        and hmac.compare_digest(parts[1], token)
    )


async def _health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok", "service": "orchestra-gateway", "host": socket.gethostname(),
    })


def make_app(token: str = "") -> web.Application:
    app = web.Application()

    async def _run(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": "Unauthorized"}, status=401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return web.json_response({"error": "Invalid JSON"}, status=400)
        if not isinstance(payload, dict):
            return web.json_response({"error": "body must be a JSON object"}, status=400)
        alias, task = payload.get("alias"), payload.get("task")
        if not alias or not task:
            return web.json_response({"error": "alias and task are required"}, status=400)
        try:
            run = await run_ask_async(alias, task)
        except Exception as exc:  # surface, don't swallow
            logger.warning("gateway run_ask failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=500)
        agent = run.agents[0] if run.agents else None
        return web.json_response({
            "stdout": agent.stdout_log if agent else "",
            "model": getattr(agent, "model", "") if agent else "",
            "cost_usd": getattr(agent, "estimated_cost_usd", 0.0) if agent else 0.0,
            "status": getattr(run.status, "value", str(run.status)),
        })

    async def _aliases(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": "Unauthorized"}, status=401)
        return web.json_response({"aliases": available_aliases()})

    async def _v1_models(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_request_error", "code": "401"}}, status=401)
        from orchestra import openai_compat
        return web.json_response({"object": "list", "data": openai_compat.list_models()})

    async def _v1_chat(request: web.Request):
        from orchestra import openai_compat
        if not _authorized(request, token):
            return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_request_error", "code": "401"}}, status=401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error", "code": "400"}}, status=400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return web.json_response({"error": {"message": "messages is required", "type": "invalid_request_error", "code": "400"}}, status=400)
        model = payload.get("model", "orchestra-auto")
        try:
            openai_compat.resolve_target(model)  # validates orchestra-<mode>
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc), "type": "invalid_request_error", "code": "404"}}, status=404)
        prompt = openai_compat.messages_to_prompt(payload["messages"])
        if payload.get("stream"):
            return await _v1_chat_stream(request, model, prompt)
        try:
            content = await openai_compat.run_target(model, prompt)
        except Exception as exc:
            logger.warning("openai chat run failed: %s", exc)
            return web.json_response({"error": {"message": str(exc), "type": "server_error", "code": "500"}}, status=500)
        return web.json_response(openai_compat.to_response(model, content))

    async def _v1_chat_stream(request: web.Request, model: str, prompt: str):
        from orchestra import openai_compat
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        try:
            async for chunk in openai_compat.stream_target(model, prompt):
                await resp.write(openai_compat.to_sse_chunk(model, chunk).encode())
            await resp.write(openai_compat.to_sse_chunk(model, "", finish_reason="stop").encode())
        except Exception as exc:
            await resp.write(openai_compat.to_sse_chunk(model, f"[ERROR] {exc}", finish_reason="stop").encode())
        await resp.write(openai_compat.sse_done().encode())
        await resp.write_eof()
        return resp

    app.router.add_get("/v1/models", _v1_models)
    app.router.add_post("/v1/chat/completions", _v1_chat)
    app.router.add_post("/run", _run)
    app.router.add_get("/aliases", _aliases)
    app.router.add_get("/health", _health)
    return app


def serve(host: str, port: int, token: str) -> None:
    if host not in ("127.0.0.1", "localhost") and not token:
        raise ValueError(
            "Refusing to bind a public host without a token — the gateway can run agents "
            "(and thus code) on this machine. Set a token: orchestra serve --token <secret>."
        )
    logger.info("orchestra gateway on %s:%s (token=%s)", host, port, "set" if token else "NONE")
    web.run_app(make_app(token), host=host, port=port)
