from __future__ import annotations
"""External MCP client: connect to another MCP server and use its tools.

Speaks newline-delimited JSON-RPC over stdio — byte-compatible with Orchestra's own
``mcp_server`` (``json.dumps(msg)+"\\n"`` / ``readline()``). The client is built on an
injectable transport so its request/response correlation is unit-testable without
spawning a real subprocess.
"""
import json
import logging
import subprocess
from typing import Any, Optional, cast

from orchestra import __version__, config

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"


class McpClientError(Exception):
    """Raised on an MCP error response or a closed/failed connection."""


class StdioTransport:
    """Newline-delimited JSON-RPC over an external MCP server subprocess's stdio."""

    def __init__(self, command: list[str]):
        self._command = command
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def send(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def recv(self) -> Optional[dict]:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            return None
        message = json.loads(line)
        return message if isinstance(message, dict) else None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
        except Exception:
            logger.warning("Failed to terminate MCP subprocess", exc_info=True)


class McpClient:
    """JSON-RPC client for an external MCP server over an injectable transport."""

    def __init__(self, transport: Any):
        self._transport = transport
        self._next_id = 0

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        request: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        self._transport.send(request)
        while True:
            message = self._transport.recv()
            if message is None:
                raise McpClientError("MCP connection closed before a response arrived")
            if message.get("id") != request_id:
                continue  # skip notifications / responses to other requests
            if "error" in message:
                raise McpClientError(str(message["error"]))
            return cast(dict, message.get("result", {}))

    def initialize(self) -> dict:
        return self._request("initialize", {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "orchestra", "version": __version__},
        })

    def list_tools(self) -> list[dict]:
        return cast("list[dict]", self._request("tools/list").get("tools", []))

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        return self._request("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @classmethod
    def from_command(cls, command: list[str]) -> "McpClient":
        transport = StdioTransport(command)
        transport.start()
        return cls(transport)


def connect_from_config(name: str) -> McpClient:
    """Connect to an external MCP server defined under ``[mcp.client_servers]``."""
    servers = config.mcp_client_servers()
    if name not in servers:
        raise McpClientError(f"No MCP client server configured under name '{name}'")
    return McpClient.from_command(list(servers[name]))
