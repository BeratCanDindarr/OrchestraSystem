"""Keep web pages away from the local servers.

The gateway, the dashboard and the WebSocket daemon run agents, and a browser will send a
request to 127.0.0.1 on behalf of any site the user visits. Three checks stop that:

- a request that carries an ``Origin`` must come from the server's own origin;
- a request the browser marks as started by another site (``Sec-Fetch-Site``) is refused:
  an ``<img>``, a ``<script>`` or a link sends no ``Origin`` at all;
- a server bound to loopback only answers to a loopback ``Host`` (a page whose domain was
  re-pointed at 127.0.0.1 has that domain in ``Host`` and in ``Origin``, so they agree).

Clients that are not browsers send none of these headers and are left to the token check.
A client that presents the server's token is past all of this: no page has the token.
"""
from __future__ import annotations

from typing import AbstractSet, Callable, Optional
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


_FOREIGN_FETCH_SITES = frozenset({"cross-site", "same-site"})


def request_allowed(
    host: str,
    origin: Optional[str],
    allowed_hosts: Optional[AbstractSet[str]] = None,
    fetch_site: Optional[str] = None,
) -> bool:
    """True if a request with these ``Host``, ``Origin`` and ``Sec-Fetch-Site`` may reach a handler."""
    if allowed_hosts is not None and _hostname(host) not in allowed_hosts:
        return False
    if fetch_site is not None and fetch_site.lower() in _FOREIGN_FETCH_SITES:
        return False
    if origin is None:
        return True
    try:
        return urlsplit(origin).netloc.lower() == host.lower() != ""
    except ValueError:
        return False


def middleware(
    allowed_hosts: Optional[AbstractSet[str]] = None,
    is_authenticated: Optional[Callable[[web.Request], bool]] = None,
):
    """aiohttp middleware that answers 403 to requests `request_allowed` rejects.

    *is_authenticated* says whether a request carries the server's token; such a request
    is let through whatever its origin.
    """

    def _allowed(request: web.Request) -> bool:
        if is_authenticated is not None and is_authenticated(request):
            return True
        origins = request.headers.getall("Origin", [])
        if len(origins) > 1:
            return False
        return request_allowed(
            request.headers.get("Host", ""),
            origins[0] if origins else None,
            allowed_hosts,
            request.headers.get("Sec-Fetch-Site"),
        )

    @web.middleware
    async def _guard(request: web.Request, handler):
        if not _allowed(request):
            return web.json_response({"error": "Forbidden: cross-origin request"}, status=403)
        return await handler(request)

    return _guard
