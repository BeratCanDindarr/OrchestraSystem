"""Keep web pages away from the local servers.

The gateway, the dashboard and the WebSocket daemon run agents, and a browser will send a
request to 127.0.0.1 on behalf of any site the user visits. Two checks stop that:

- a request that carries an ``Origin`` must come from the server's own origin;
- a server bound to loopback only answers to a loopback ``Host`` (a page whose domain was
  re-pointed at 127.0.0.1 has that domain in ``Host`` and in ``Origin``, so they agree).

Clients that are not browsers send no ``Origin`` and are left to the token check.
"""
from __future__ import annotations

from typing import AbstractSet, Optional
from urllib.parse import urlsplit

from aiohttp import web

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def allowed_hosts_for(bind_host: str) -> Optional[frozenset[str]]:
    """The ``Host`` names a server bound to *bind_host* answers to (None: any)."""
    return LOOPBACK_HOSTS if is_loopback(bind_host) else None


def _hostname(netloc: str) -> str:
    try:
        return urlsplit(f"//{netloc}").hostname or ""
    except ValueError:
        return ""


def request_allowed(
    host: str, origin: Optional[str], allowed_hosts: Optional[AbstractSet[str]] = None
) -> bool:
    """True if a request with this ``Host`` and ``Origin`` may reach a handler."""
    if allowed_hosts is not None and _hostname(host) not in allowed_hosts:
        return False
    if origin is None:
        return True
    try:
        return urlsplit(origin).netloc.lower() == host.lower() != ""
    except ValueError:
        return False


def middleware(allowed_hosts: Optional[AbstractSet[str]] = None):
    """aiohttp middleware that answers 403 to requests `request_allowed` rejects."""

    @web.middleware
    async def _guard(request: web.Request, handler):
        host = request.headers.get("Host", "")
        if not request_allowed(host, request.headers.get("Origin"), allowed_hosts):
            return web.json_response({"error": "Forbidden: cross-origin request"}, status=403)
        return await handler(request)

    return _guard
