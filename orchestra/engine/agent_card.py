from __future__ import annotations
"""A2A (Agent-to-Agent) Agent Card: a standard, discoverable JSON document that
describes this Orchestra node — its agents (as A2A "skills") and, optionally, its
MCP tools — so other A2A-compatible frameworks can find and use it.

Pure builder (unit-testable) + a thin urllib fetch helper for the client side.
Served by the gateway at ``GET /.well-known/agent.json``.
"""
import json
import urllib.request
from typing import Any, Optional

_PROTOCOL_VERSION = "0.2.0"


def build_agent_card(
    *,
    host: str,
    version: str,
    url: str,
    aliases: list[str],
    mcp_tools: Optional[list[dict]] = None,
) -> dict:
    """Build an A2A Agent Card for this node.

    Args:
        host: Node hostname (identity).
        version: Orchestra version string.
        url: Base URL where this node is reachable.
        aliases: Agent aliases exposed as A2A skills.
        mcp_tools: Optional MCP tool descriptors (from the MCP registry); when
            provided, advertised under the ``mcpTools`` extension key.
    """
    card: dict[str, Any] = {
        "name": f"orchestra@{host}",
        "description": "Orchestra multi-agent orchestration node",
        "url": url,
        "version": version,
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": a, "name": a, "description": f"Orchestra agent alias '{a}'", "tags": ["agent"]}
            for a in aliases
        ],
    }
    if mcp_tools:
        card["mcpTools"] = [
            {"name": t["name"], "description": t.get("description", "")} for t in mcp_tools
        ]
    return card


def fetch_agent_card(base_url: str, token: str = "", *, timeout: float = 5.0) -> dict:
    """Fetch and parse a remote node's Agent Card.

    Args:
        base_url: Remote node base URL (e.g. "http://host:7070").
        token: Optional bearer token.
        timeout: Request timeout in seconds.
    """
    url = base_url.rstrip("/") + "/.well-known/agent.json"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted LAN peer)
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else {}
