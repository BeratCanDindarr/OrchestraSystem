"""Agent gateway: expose this node's run_ask over HTTP so a remote Orchestra can use it.

Small aiohttp server (see federation_http.py for the pattern). Wraps run_ask; adds nothing.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import socket
from typing import AbstractSet

from aiohttp import web, ClientError

from orchestra.engine.runners.common import run_ask_async
from orchestra.providers.fallback import available_aliases
from orchestra.utils import browser_guard

logger = logging.getLogger(__name__)

# Seconds between keep-alive pings emitted while a buffered native run is in flight.
_NATIVE_PING_INTERVAL = 3.0


def _authorized(request: web.Request, token: str) -> bool:
    if not token:
        return True
    token_bytes = token.encode("utf-8")
    api_key = request.headers.get("x-api-key")
    if api_key is not None and hmac.compare_digest(api_key.encode("utf-8"), token_bytes):
        return True
    header = request.headers.get("Authorization", "")
    parts = header.split()
    return (
        len(parts) == 2
        and parts[0].lower() == "bearer"
        # Compare bytes, not str: hmac.compare_digest raises TypeError on non-ASCII
        # str, so a non-ASCII bearer (untrusted client input) or a non-ASCII configured
        # token would otherwise surface as a 500 instead of a clean 401.
        and hmac.compare_digest(parts[1].encode("utf-8"), token_bytes)
    )


async def _health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok", "service": "orchestra-gateway", "host": socket.gethostname(),
    })


def make_app(token: str = "", allowed_hosts: AbstractSet[str] | None = None) -> web.Application:
    app = web.Application(middlewares=[browser_guard.middleware(
        allowed_hosts, is_authenticated=lambda request: bool(token) and _authorized(request, token),
    )])

    async def _identify(request: web.Request) -> web.Response:
        # Unauthenticated, like /health: lets a caller check that this gateway knows the
        # token without either side sending it (see remote_nodes.identity_proof).
        from orchestra import remote_nodes
        nonce = request.query.get("nonce", "")
        if not remote_nodes.valid_nonce(nonce):
            return web.json_response({"error": "nonce must be 16-128 url-safe characters"}, status=400)
        sockname = request.transport.get_extra_info("sockname") if request.transport else None
        address = sockname[0] if sockname else ""
        return web.json_response({
            "service": "orchestra-gateway",
            "token_required": bool(token),
            "proof": remote_nodes.identity_proof(token, nonce, address),
        })

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
        except Exception as exc:  # surface, don't swallow — but keep detail server-side
            logger.warning("gateway run_ask failed: %s", exc)
            return web.json_response({"error": "internal error running the agent"}, status=500)
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

    async def _agent_card(request: web.Request) -> web.Response:
        # Read-only A2A discovery metadata — unauthenticated, like /health.
        from orchestra import __version__, config
        from orchestra.engine.agent_card import build_agent_card
        mcp_tools = None
        if config.mcp_enabled():
            from orchestra.engine.mcp_registry import REGISTRY
            mcp_tools = REGISTRY.list_tools()
        return web.json_response(build_agent_card(
            host=socket.gethostname(),
            version=__version__,
            url=str(request.url.with_path("/")),
            aliases=available_aliases(),
            mcp_tools=mcp_tools,
        ))

    async def _v1_models(request: web.Request) -> web.Response:
        if not _authorized(request, token):
            return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_request_error", "code": "401"}}, status=401)
        from orchestra import openai_compat
        return web.json_response({"object": "list", "data": openai_compat.list_models()})

    async def _v1_chat(request: web.Request):
        from orchestra import openai_compat, gateway_dispatch, thinking_modes
        if not _authorized(request, token):
            return web.json_response({"error": {"message": "Unauthorized", "type": "invalid_request_error", "code": "401"}}, status=401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return web.json_response({"error": {"message": "Invalid JSON", "type": "invalid_request_error", "code": "400"}}, status=400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return web.json_response({"error": {"message": "messages is required", "type": "invalid_request_error", "code": "400"}}, status=400)
        model, mode = thinking_modes.split_mode(payload.get("model", "orchestra-auto"))
        try:
            gateway_dispatch.resolve_target(model)  # validates orchestra-<mode>
        except ValueError as exc:
            return web.json_response({"error": {"message": str(exc), "type": "invalid_request_error", "code": "404"}}, status=404)
        messages = payload["messages"]
        if mode:
            mp = thinking_modes.mode_prompt(mode, model)
            if mp:
                messages = [{"role": "system", "content": mp}, *messages]
        prompt = openai_compat.messages_to_prompt(messages)
        if payload.get("stream"):
            return await _v1_chat_stream(request, model, prompt)
        try:
            content = await gateway_dispatch.run_target(model, prompt)
        except Exception as exc:
            logger.warning("openai chat run failed: %s", exc)
            return web.json_response({"error": {"message": "internal error running the model", "type": "server_error", "code": "500"}}, status=500)
        return web.json_response(openai_compat.to_response(model, content))

    async def _v1_chat_stream(request: web.Request, model: str, prompt: str):
        from orchestra import openai_compat, gateway_dispatch
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        agen = gateway_dispatch.stream_target(model, prompt)
        try:
            async for chunk in agen:
                await resp.write(openai_compat.to_sse_chunk(model, chunk).encode())
        except (ConnectionResetError, ClientError):
            return resp  # client disconnected; nothing more to send
        except Exception as exc:
            logger.warning("openai chat stream failed: %s", exc)
            with contextlib.suppress(ConnectionResetError, ClientError):
                await resp.write(openai_compat.to_sse_chunk(model, "[stream error]", finish_reason="stop").encode())
        finally:
            await agen.aclose()  # deterministically runs stream_target's finally → stop.set()
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(openai_compat.to_sse_chunk(model, "", finish_reason="stop").encode())
            await resp.write(openai_compat.sse_done().encode())
            await resp.write_eof()
        return resp

    def _anthropic_error(message: str, err_type: str, status: int) -> web.Response:
        return web.json_response(
            {"type": "error", "error": {"type": err_type, "message": message}}, status=status
        )

    async def _v1_messages(request: web.Request):
        from orchestra import gateway_dispatch, anthropic_compat, thinking_modes
        if not _authorized(request, token):
            return _anthropic_error("authentication failed", "authentication_error", 401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _anthropic_error("Invalid JSON", "invalid_request_error", 400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return _anthropic_error("messages is required", "invalid_request_error", 400)
        base, mode = thinking_modes.split_mode(payload.get("model", "oll-fast"))
        model = anthropic_compat.resolve_anthropic_model(base)
        if mode:
            mp = thinking_modes.mode_prompt(mode, base)
            if mp:
                payload = {**payload, "system": thinking_modes.prepend_system(payload.get("system"), mp)}
        if payload.get("tools") and gateway_dispatch.native_capable(model):
            return await _v1_messages_native(request, model, payload)
        prompt = anthropic_compat.request_to_prompt(payload.get("system"), payload["messages"])
        if payload.get("stream"):
            return await _v1_messages_stream(request, model, prompt)
        try:
            text = await gateway_dispatch.run_target(model, prompt)
        except Exception as exc:
            logger.warning("anthropic messages run failed: %s", exc)
            return _anthropic_error("internal error running the model", "api_error", 500)
        return web.json_response(anthropic_compat.to_message_response(
            model, text,
            input_tokens=anthropic_compat.approx_tokens(prompt),
            output_tokens=anthropic_compat.approx_tokens(text),
        ))

    async def _v1_messages_stream(request: web.Request, model: str, prompt: str):
        from orchestra import gateway_dispatch, anthropic_compat
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        msg_id = anthropic_compat.new_message_id()
        agen = gateway_dispatch.stream_target(model, prompt)
        output = 0
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(anthropic_compat.sse_message_start(model, msg_id, anthropic_compat.approx_tokens(prompt)).encode())
            await resp.write(anthropic_compat.sse_content_block_start().encode())
        try:
            async for chunk in agen:
                output += anthropic_compat.approx_tokens(chunk)
                await resp.write(anthropic_compat.sse_text_delta(chunk).encode())
        except (ConnectionResetError, ClientError):
            return resp  # client disconnected
        except Exception as exc:
            logger.warning("anthropic messages stream failed: %s", exc)
            with contextlib.suppress(ConnectionResetError, ClientError):
                await resp.write(anthropic_compat.sse_text_delta("[stream error]").encode())
        finally:
            await agen.aclose()  # deterministically stops the producer thread
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(anthropic_compat.sse_content_block_stop().encode())
            await resp.write(anthropic_compat.sse_message_delta(output).encode())
            await resp.write(anthropic_compat.sse_message_stop().encode())
            await resp.write_eof()
        return resp

    async def _v1_messages_native(request: web.Request, model: str, payload: dict):
        from orchestra import gateway_dispatch, anthropic_compat
        native_payload = {k: payload[k] for k in
                          ("system", "messages", "tools", "tool_choice", "max_tokens")
                          if k in payload}
        if payload.get("stream"):
            return await _v1_messages_stream_native(request, model, native_payload)
        try:
            result = await gateway_dispatch.run_native(model, native_payload)
        except Exception as exc:
            logger.warning("anthropic native run failed: %s", exc)
            return _anthropic_error("internal error running the model", "api_error", 500)
        return web.json_response(anthropic_compat.to_native_message_response(model, result))

    async def _v1_messages_stream_native(request: web.Request, model: str, native_payload: dict):
        from orchestra import gateway_dispatch, anthropic_compat
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        # Emit message_start BEFORE the (buffered, possibly slow) provider call so the
        # client gets a first byte immediately — a slow local model would otherwise
        # produce seconds of silence and trip the client's first-byte timeout.
        with contextlib.suppress(ConnectionResetError, ClientError):
            await resp.write(anthropic_compat.sse_message_start(
                model, anthropic_compat.new_message_id(), 0).encode())
        # Run the buffered provider call concurrently and emit keep-alive pings every
        # few seconds so a long local-model generation doesn't idle out the stream.
        task = asyncio.ensure_future(gateway_dispatch.run_native(model, native_payload))
        while not task.done():
            await asyncio.wait({task}, timeout=_NATIVE_PING_INTERVAL)
            if task.done():
                break
            try:
                await resp.write(anthropic_compat.sse_ping().encode())
            except (ConnectionResetError, ClientError):
                task.cancel()
                return resp  # client gone; abandon the run
        try:
            result = await task
        except asyncio.CancelledError:
            return resp
        except Exception as exc:
            logger.warning("anthropic native stream failed: %s", exc)
            result = {"content": [{"type": "text", "text": "[stream error]"}],
                      "stop_reason": "end_turn", "usage": {}}
        usage = result.get("usage") or {}
        with contextlib.suppress(ConnectionResetError, ClientError):
            for i, block in enumerate(result.get("content") or []):
                if block.get("type") == "tool_use":
                    start = {"type": "tool_use", "id": block.get("id", ""),
                             "name": block.get("name", ""), "input": {}}
                    await resp.write(anthropic_compat.sse_content_block_start_typed(i, start).encode())
                    await resp.write(anthropic_compat.sse_input_json_delta(
                        i, json.dumps(block.get("input", {}), ensure_ascii=False)).encode())
                else:
                    await resp.write(anthropic_compat.sse_content_block_start_typed(
                        i, {"type": "text", "text": ""}).encode())
                    await resp.write(anthropic_compat.sse_text_delta_at(i, block.get("text", "")).encode())
                await resp.write(anthropic_compat.sse_content_block_stop_at(i).encode())
            await resp.write(anthropic_compat.sse_message_delta_reason(
                result.get("stop_reason", "end_turn"), int(usage.get("output_tokens") or 0)).encode())
            await resp.write(anthropic_compat.sse_message_stop().encode())
            await resp.write_eof()
        return resp

    async def _v1_messages_count_tokens(request: web.Request):
        from orchestra import anthropic_compat
        if not _authorized(request, token):
            return _anthropic_error("authentication failed", "authentication_error", 401)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _anthropic_error("Invalid JSON", "invalid_request_error", 400)
        if not isinstance(payload, dict) or not payload.get("messages"):
            return _anthropic_error("messages is required", "invalid_request_error", 400)
        prompt = anthropic_compat.request_to_prompt(payload.get("system"), payload["messages"])
        return web.json_response({"input_tokens": anthropic_compat.approx_tokens(prompt)})

    app.router.add_get("/v1/models", _v1_models)
    app.router.add_post("/v1/chat/completions", _v1_chat)
    app.router.add_post("/v1/messages", _v1_messages)
    app.router.add_post("/v1/messages/count_tokens", _v1_messages_count_tokens)
    app.router.add_post("/run", _run)
    app.router.add_get("/aliases", _aliases)
    app.router.add_get("/.well-known/agent.json", _agent_card)
    app.router.add_get("/health", _health)
    app.router.add_get("/identify", _identify)
    return app


def serve(host: str, port: int, token: str) -> None:
    if not browser_guard.is_loopback(host) and not token:
        raise ValueError(
            "Refusing to bind a public host without a token — the gateway can run agents "
            "(and thus code) on this machine. Set a token: orchestra serve --token <secret>."
        )
    logger.info("orchestra gateway on %s:%s (token=%s)", host, port, "set" if token else "NONE")
    web.run_app(make_app(token, browser_guard.allowed_hosts_for(host)), host=host, port=port)
