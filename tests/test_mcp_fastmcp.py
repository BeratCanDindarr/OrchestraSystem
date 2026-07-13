"""Tests for the SDK-based MCP bridge server (orchestra/mcp_fastmcp.py).

Run from a scratch directory to avoid artifact pollution:
    cd /tmp && /abs/.venv/bin/pytest /abs/tests/test_mcp_fastmcp.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from mcp.shared.memory import create_connected_server_and_client_session as _client_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build():
    """Import and build the server fresh (avoids cross-test state)."""
    from orchestra.mcp_fastmcp import build_server
    return build_server()


# ---------------------------------------------------------------------------
# 1. list_tools — at least 13 tools with required fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_returns_all_registered_tools():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.list_tools()
    names = [t.name for t in result.tools]
    # Bridge must expose every REGISTRY tool (12 today); guard against silent drop.
    assert len(names) >= 12, f"Expected >= 12 tools, got {len(names)}"
    assert all(t.inputSchema for t in result.tools), "every tool must carry an inputSchema"


@pytest.mark.asyncio
async def test_list_tools_includes_core_tools():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.list_tools()
    names = {t.name for t in result.tools}
    required = {
        "orchestra_submit_run",
        "orchestra_get_run",
        "orchestra_continue_run",
        "orchestra_cancel_run",
        "orchestra_list_runs",
    }
    missing = required - names
    assert not missing, f"Missing core tools: {missing}"


@pytest.mark.asyncio
async def test_list_tools_each_has_input_schema():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.list_tools()
    for tool in result.tools:
        assert tool.inputSchema is not None, f"Tool {tool.name!r} has no inputSchema"


# ---------------------------------------------------------------------------
# 2. call_tool success path — orchestra_list_runs {}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_tool_list_runs_success():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.call_tool("orchestra_list_runs", {})
    assert not result.isError, f"Expected success but got error: {result.content}"
    assert result.content, "Expected non-empty content"


# ---------------------------------------------------------------------------
# 3. call_tool error path — nonexistent run_id → isError True, not a crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_tool_get_run_not_found_returns_tool_error():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.call_tool(
            "orchestra_get_run", {"run_id": "nonexistent-run-abc123"}
        )
    assert result.isError, "Expected isError=True for nonexistent run"
    text = result.content[0].text if result.content else ""
    assert text, "Expected a non-empty error message"


# ---------------------------------------------------------------------------
# 4. Policy gate — deny → tool error mentioning denial
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_policy_gate_deny_returns_tool_error(monkeypatch):
    import orchestra.mcp_fastmcp as mod

    class _FakeDecision:
        action = "deny"
        reason = "test-policy-block"

    class _FakePolicy:
        def evaluate_tool(self, name):
            return _FakeDecision()

    monkeypatch.setattr(mod, "POLICY", _FakePolicy())
    srv = _build()

    async with _client_session(srv) as client:
        result = await client.call_tool("orchestra_list_runs", {})

    assert result.isError, "Expected isError=True when policy denies"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert "denied" in text.lower() or "policy" in text.lower() or "test-policy-block" in text.lower(), (
        f"Expected denial message in content, got: {text!r}"
    )


# ---------------------------------------------------------------------------
# 5. Missing-SDK guard — RuntimeError with helpful message
# ---------------------------------------------------------------------------

def test_missing_sdk_guard_message():
    """The module-level guard must raise RuntimeError with install hint."""
    import orchestra.mcp_fastmcp as mod
    import importlib
    import sys
    import types as builtin_types

    # Temporarily shadow the mcp package with a broken module so we can
    # test the guard path without actually uninstalling the SDK.
    original_mcp = sys.modules.get("mcp")
    fake_mcp = builtin_types.ModuleType("mcp")
    # Remove sub-modules so the lazy import fails to find mcp.server.lowlevel
    mcp_server = builtin_types.ModuleType("mcp.server")
    mcp_server_lowlevel = builtin_types.ModuleType("mcp.server.lowlevel")

    def _raise(*a, **kw):
        raise ModuleNotFoundError("No module named 'mcp'")

    # Patch the guard helper directly to verify the message shape
    # (testing the actual import-time behavior is OS-dependent and fragile).
    # Instead, call the internal guard function if exposed, or check the
    # module docstring / raise condition.
    #
    # The real check: ensure build_server() raises RuntimeError (not ImportError)
    # when mcp cannot be imported.  We test this by temporarily replacing
    # the mcp import inside build_server with a version that raises.
    saved = sys.modules.copy()
    # Inject broken mcp so that `import mcp.server.lowlevel` fails
    sys.modules["mcp"] = fake_mcp  # type: ignore[assignment]
    sys.modules.pop("mcp.server", None)
    sys.modules.pop("mcp.server.lowlevel", None)
    sys.modules.pop("mcp.types", None)

    try:
        # Force reload of our module so it re-executes the lazy guard
        if "orchestra.mcp_fastmcp" in sys.modules:
            del sys.modules["orchestra.mcp_fastmcp"]
        import orchestra.mcp_fastmcp as fresh_mod
        with pytest.raises(RuntimeError, match="mcp.*extra"):
            fresh_mod.build_server()
    finally:
        # Restore everything
        sys.modules.update(saved)
        if "orchestra.mcp_fastmcp" in sys.modules:
            del sys.modules["orchestra.mcp_fastmcp"]
