"""MCP e2e round-trip test.

Spawns orchestra.mcp_server as a subprocess, pipes NDJSON requests, and asserts
the stdio transport + real tool registrations behave correctly.

This test pins:
- initialize response shape (protocolVersion, capabilities.tools, no resources)
- tools/list count (>= 11) and orchestra_submit_run required schema
- unknown-method -32601 error (regression: previously hung forever)
- tools/call orchestra_list_runs succeeds (isError false, content list)

Mark: runs in the default `-m "not slow"` selection (no slow mark applied).
"""
from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_S = 15  # per-read timeout in seconds


def _send(proc: subprocess.Popen, obj: dict) -> None:
    line = json.dumps(obj) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()


def _recv(proc: subprocess.Popen, timeout: float = TIMEOUT_S) -> Optional[dict]:
    """Read one line from proc stdout with a timeout.

    Returns the parsed dict, or raises TimeoutError if nothing arrives in time.
    Raises RuntimeError if the process has exited unexpectedly.
    """
    rlist, _, _ = select.select([proc.stdout], [], [], timeout)
    if not rlist:
        raise TimeoutError(
            f"MCP server did not respond within {timeout}s — possible hang regression"
        )
    raw = proc.stdout.readline()
    if not raw:
        # EOF — process likely crashed
        raise RuntimeError(
            f"MCP server stdout closed unexpectedly (returncode={proc.poll()})"
        )
    return json.loads(raw.decode())


# ---------------------------------------------------------------------------
# Fixture helpers (manual setup/teardown — no pytest fixtures needed)
# ---------------------------------------------------------------------------

def _make_tmp_project() -> Path:
    """Create a temp directory with a minimal .orchestra/config.toml."""
    tmp = Path(tempfile.mkdtemp(prefix="orch_e2e_"))
    dot_orch = tmp / ".orchestra"
    dot_orch.mkdir()
    # Minimal config — empty file is acceptable (config.py returns {} on missing keys).
    # We write a valid TOML stub to be explicit.
    (dot_orch / "config.toml").write_text(
        textwrap.dedent("""\
            # minimal e2e test config
            [orchestra]
            timeout = 60
        """)
    )
    return tmp


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_mcp_e2e_round_trip() -> None:
    """Full stdio round-trip through the MCP server subprocess."""
    tmp_project = _make_tmp_project()
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "orchestra.mcp_server"],
            cwd=str(tmp_project),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # suppress log noise in CI
        )

        # ------------------------------------------------------------------
        # 1. initialize
        # ------------------------------------------------------------------
        _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        resp = _recv(proc)
        assert resp.get("id") == 1, f"Bad id: {resp}"
        result = resp.get("result", {})
        assert "protocolVersion" in result, f"Missing protocolVersion: {result}"
        caps = result.get("capabilities", {})
        assert "tools" in caps, f"Missing capabilities.tools: {caps}"
        assert "resources" not in result, (
            f"Unexpected 'resources' key in initialize result: {result}"
        )

        # ------------------------------------------------------------------
        # 2. tools/list
        # ------------------------------------------------------------------
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        resp = _recv(proc)
        assert resp.get("id") == 2, f"Bad id: {resp}"
        tools = resp["result"]["tools"]
        assert len(tools) >= 11, f"Expected >= 11 tools, got {len(tools)}: {[t['name'] for t in tools]}"

        submit_tool = next(
            (t for t in tools if t["name"] == "orchestra_submit_run"), None
        )
        assert submit_tool is not None, "orchestra_submit_run not found in tools/list"
        schema = submit_tool.get("inputSchema", {})
        required = schema.get("required", [])
        assert "mode" in required, f"'mode' not in required: {required}"
        assert "task" in required, f"'task' not in required: {required}"

        # ------------------------------------------------------------------
        # 3. unknown method → -32601 (regression: used to hang)
        # ------------------------------------------------------------------
        _send(
            proc,
            {"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
        )
        resp = _recv(proc)
        assert resp.get("id") == 3, f"Bad id: {resp}"
        error = resp.get("error", {})
        assert error.get("code") == -32601, (
            f"Expected -32601 for unknown method, got: {resp}"
        )

        # ------------------------------------------------------------------
        # 4. tools/call orchestra_list_runs → success
        # ------------------------------------------------------------------
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "orchestra_list_runs", "arguments": {}},
            },
        )
        resp = _recv(proc, timeout=TIMEOUT_S)
        assert resp.get("id") == 4, f"Bad id: {resp}"
        tool_result = resp.get("result", {})
        assert tool_result.get("isError") is not True, (
            f"orchestra_list_runs returned isError=true: {tool_result}"
        )
        content = tool_result.get("content")
        assert isinstance(content, list), (
            f"Expected content list, got: {type(content)}: {tool_result}"
        )

    finally:
        if proc is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmp_project, ignore_errors=True)
