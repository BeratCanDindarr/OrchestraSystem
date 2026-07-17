from __future__ import annotations
"""External MCP client: connect to another MCP server and use its tools.

Speaks newline-delimited JSON-RPC over stdio — byte-compatible with Orchestra's own
``mcp_server`` (``json.dumps(msg)+"\\n"`` / ``readline()``). The client is built on an
injectable transport so its request/response correlation is unit-testable without
spawning a real subprocess.

Resilience: ``StdioTransport`` reads on a background thread and every ``recv`` is
bounded by a timeout, so a hung/buggy external server can never deadlock the caller;
``close()`` reaps the subprocess (terminate → wait → kill).
"""
import json
import logging
import queue
import subprocess
import threading
from typing import Any, Optional, cast

from orchestra import __version__, config

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"
_DEFAULT_TIMEOUT = 30.0
_EOF = object()  # sentinel pushed by the reader thread on stream close


class McpClientError(Exception):
    """Raised on an MCP error response, a timeout, or a closed/failed connection."""


class StdioTransport:
    """Newline-delimited JSON-RPC over an external MCP server subprocess's stdio.

    A background reader thread drains stdout into a queue so ``recv`` can enforce a
    timeout — a plain blocking ``readline`` cannot be interrupted and would deadlock
    on a server that accepts a request but never answers.
    """

    def __init__(self, command: list[str]):
        self._command = command
        self._proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    break
                self._queue.put(line)
        except Exception:
            logger.warning("MCP reader thread failed", exc_info=True)
        finally:
            self._queue.put(_EOF)

    def send(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def recv(self, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise McpClientError(f"no response from MCP server within {timeout}s")
        if item is _EOF:
            return None
        try:
            message = json.loads(item)
        except json.JSONDecodeError:
            logger.warning("MCP: skipping malformed line")
            return {}  # non-dict-with-id → caller's correlation loop reads the next message
        return message if isinstance(message, dict) else None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        except Exception:
            logger.warning("Failed to reap MCP subprocess", exc_info=True)


class McpClient:
    """JSON-RPC client for an external MCP server over an injectable transport."""

    def __init__(self, transport: Any, *, timeout: float = _DEFAULT_TIMEOUT):
        self._transport = transport
        self._timeout = timeout
        self._next_id = 0

    def _request(self, method: str, params: Optional[dict] = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        request: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            request["params"] = params
        self._transport.send(request)
        while True:
            message = self._transport.recv(self._timeout)
            if message is None:
                raise McpClientError("MCP connection closed before a response arrived")
            if message.get("id") != request_id:
                continue  # skip notifications / responses to other requests / malformed
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
    def from_command(cls, command: list[str], *, timeout: float = _DEFAULT_TIMEOUT) -> "McpClient":
        transport = StdioTransport(command)
        try:
            transport.start()
        except Exception as exc:
            transport.close()
            raise McpClientError(f"failed to start MCP server {command!r}: {exc}") from exc
        return cls(transport, timeout=timeout)


def connect_from_config(name: str) -> McpClient:
    """Connect to an external MCP server defined under ``[mcp.client_servers]``."""
    servers = config.mcp_client_servers()
    if name not in servers:
        raise McpClientError(f"No MCP client server configured under name '{name}'")
    return McpClient.from_command(list(servers[name]))
