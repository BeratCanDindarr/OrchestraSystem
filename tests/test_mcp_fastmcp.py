"""Tests for the SDK-based MCP bridge server (orchestra/mcp_fastmcp.py).

Run from a scratch directory to avoid artifact pollution:
    cd /tmp && /abs/.venv/bin/pytest /abs/tests/test_mcp_fastmcp.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

import pytest
import mcp.types as types

from mcp.shared.memory import create_connected_server_and_client_session as _client_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build():
    """Import and build the server fresh (avoids cross-test state)."""
    from orchestra.mcp_fastmcp import build_server
    return build_server()


# ---------------------------------------------------------------------------
# 1. list_tools — all registered tools (12 today) with required fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_returns_all_registered_tools():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.list_tools()
    names = [t.name for t in result.tools]
    # Bridge must expose every REGISTRY tool (10) plus orchestra_request_approval.
    assert len(names) == 11, f"Expected 11 tools, got {len(names)}"
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


# ---------------------------------------------------------------------------
# 6. orchestra_request_approval — list_tools includes it with required run_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools_includes_request_approval():
    srv = _build()
    async with _client_session(srv) as client:
        result = await client.list_tools()
    names = {t.name: t for t in result.tools}
    assert "orchestra_request_approval" in names, (
        f"Expected orchestra_request_approval in tool list, got: {sorted(names)}"
    )
    tool = names["orchestra_request_approval"]
    schema = tool.inputSchema
    assert schema is not None, "orchestra_request_approval must have an inputSchema"
    required = schema.get("required", [])
    assert "run_id" in required, f"run_id must be required, got required={required}"


# ---------------------------------------------------------------------------
# 6b. _build_elicitation_message scrubs PII (parity with REGISTRY._tool_success)
# ---------------------------------------------------------------------------

def test_elicitation_message_scrubs_pii():
    from orchestra.mcp_fastmcp import _build_elicitation_message

    run = {
        "run_id": "run-pii",
        "mode": "critical",
        "summary": "Contact the user at jane.doe@example.com about the change.",
        "agents": [{"alias": "cdx-deep"}],
    }
    msg = _build_elicitation_message(run)
    assert "jane.doe@example.com" not in msg, "email must be scrubbed from the dialog"
    # non-PII content survives
    assert "critical" in msg and "cdx-deep" in msg


# 7. orchestra_request_approval — waiting_approval + accept/approve → continue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_accept_approve_calls_continue(monkeypatch):
    """waiting_approval + callback returns accept/approve → continue_run called once."""
    import orchestra.mcp_fastmcp as mod

    fake_run = {
        "run_id": "run-123",
        "mode": "critical",
        "status": "waiting_approval",
        "agents": [{"alias": "cdx-fast"}, {"alias": "gmn-pro"}],
        "summary": "Round 1 done",
    }

    get_run_calls = []
    continue_calls = []
    cancel_calls = []

    def fake_get_run(run_id):
        get_run_calls.append(run_id)
        return {"run": fake_run}

    def fake_continue_run(run_id):
        continue_calls.append(run_id)
        return {"run": {**fake_run, "status": "completed"}}

    def fake_cancel_run(run_id):
        cancel_calls.append(run_id)
        return {"run": {**fake_run, "status": "cancelled"}}

    monkeypatch.setattr(mod, "_svc_get_run", fake_get_run)
    monkeypatch.setattr(mod, "_svc_continue_run", fake_continue_run)
    monkeypatch.setattr(mod, "_svc_cancel_run", fake_cancel_run)

    async def elicitation_callback(context, params):
        return types.ElicitResult(action="accept", content={"decision": "approve"})

    srv = _build()
    async with _client_session(srv, elicitation_callback=elicitation_callback) as client:
        result = await client.call_tool("orchestra_request_approval", {"run_id": "run-123"})

    assert not result.isError, f"Expected success, got: {[c.text for c in result.content]}"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert "resum" in text.lower() or "approv" in text.lower() or "continu" in text.lower(), (
        f"Expected approval/resume message, got: {text!r}"
    )
    assert continue_calls == ["run-123"], f"continue_run should be called once, got: {continue_calls}"
    assert cancel_calls == [], f"cancel_run should not be called, got: {cancel_calls}"


# ---------------------------------------------------------------------------
# 8. orchestra_request_approval — waiting_approval + accept/reject → cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_accept_reject_calls_cancel(monkeypatch):
    """waiting_approval + accept/reject → cancel_run called; continue NOT called."""
    import orchestra.mcp_fastmcp as mod

    fake_run = {
        "run_id": "run-456",
        "mode": "critical",
        "status": "waiting_approval",
        "agents": [],
        "summary": "",
    }

    continue_calls = []
    cancel_calls = []

    monkeypatch.setattr(mod, "_svc_get_run", lambda rid: {"run": fake_run})
    monkeypatch.setattr(mod, "_svc_continue_run", lambda rid: continue_calls.append(rid) or {})
    monkeypatch.setattr(mod, "_svc_cancel_run", lambda rid: cancel_calls.append(rid) or {"run": {**fake_run, "status": "cancelled"}})

    async def elicitation_callback(context, params):
        return types.ElicitResult(action="accept", content={"decision": "reject"})

    srv = _build()
    async with _client_session(srv, elicitation_callback=elicitation_callback) as client:
        result = await client.call_tool("orchestra_request_approval", {"run_id": "run-456"})

    assert not result.isError, f"Unexpected error: {[c.text for c in result.content]}"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert "cancel" in text.lower() or "reject" in text.lower(), (
        f"Expected cancel/reject mention, got: {text!r}"
    )
    assert cancel_calls == ["run-456"], f"cancel_run should be called, got: {cancel_calls}"
    assert continue_calls == [], f"continue_run should NOT be called, got: {continue_calls}"


# ---------------------------------------------------------------------------
# 9. orchestra_request_approval — waiting_approval + decline → cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_decline_calls_cancel(monkeypatch):
    """waiting_approval + callback returns decline → cancel_run called; continue NOT called."""
    import orchestra.mcp_fastmcp as mod

    fake_run = {
        "run_id": "run-789",
        "mode": "dual",
        "status": "waiting_approval",
        "agents": [],
        "summary": "",
    }

    continue_calls = []
    cancel_calls = []

    monkeypatch.setattr(mod, "_svc_get_run", lambda rid: {"run": fake_run})
    monkeypatch.setattr(mod, "_svc_continue_run", lambda rid: continue_calls.append(rid) or {})
    monkeypatch.setattr(mod, "_svc_cancel_run", lambda rid: cancel_calls.append(rid) or {"run": {**fake_run, "status": "cancelled"}})

    async def elicitation_callback(context, params):
        return types.ElicitResult(action="decline")

    srv = _build()
    async with _client_session(srv, elicitation_callback=elicitation_callback) as client:
        result = await client.call_tool("orchestra_request_approval", {"run_id": "run-789"})

    assert not result.isError, f"Unexpected error: {[c.text for c in result.content]}"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert "cancel" in text.lower() or "declin" in text.lower(), (
        f"Expected cancel/decline mention, got: {text!r}"
    )
    assert cancel_calls == ["run-789"]
    assert continue_calls == []


# ---------------------------------------------------------------------------
# 10. orchestra_request_approval — waiting_approval + cancel action → cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_cancel_action_calls_cancel(monkeypatch):
    """waiting_approval + callback returns cancel → cancel_run called; continue NOT called."""
    import orchestra.mcp_fastmcp as mod

    fake_run = {
        "run_id": "run-abc",
        "mode": "dual",
        "status": "waiting_approval",
        "agents": [],
        "summary": "",
    }

    continue_calls = []
    cancel_calls = []

    monkeypatch.setattr(mod, "_svc_get_run", lambda rid: {"run": fake_run})
    monkeypatch.setattr(mod, "_svc_continue_run", lambda rid: continue_calls.append(rid) or {})
    monkeypatch.setattr(mod, "_svc_cancel_run", lambda rid: cancel_calls.append(rid) or {"run": {**fake_run, "status": "cancelled"}})

    async def elicitation_callback(context, params):
        return types.ElicitResult(action="cancel")

    srv = _build()
    async with _client_session(srv, elicitation_callback=elicitation_callback) as client:
        result = await client.call_tool("orchestra_request_approval", {"run_id": "run-abc"})

    assert not result.isError, f"Unexpected error: {[c.text for c in result.content]}"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert "cancel" in text.lower(), f"Expected cancel mention, got: {text!r}"
    assert cancel_calls == ["run-abc"]
    assert continue_calls == []


# ---------------------------------------------------------------------------
# 11. orchestra_request_approval — status not waiting_approval → no elicitation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_non_waiting_status_no_elicit(monkeypatch):
    """status not waiting_approval → NO elicit call; result explains not-awaiting."""
    import orchestra.mcp_fastmcp as mod

    fake_run = {
        "run_id": "run-xyz",
        "mode": "dual",
        "status": "running",
        "agents": [],
        "summary": "",
    }

    continue_calls = []
    cancel_calls = []

    monkeypatch.setattr(mod, "_svc_get_run", lambda rid: {"run": fake_run})
    monkeypatch.setattr(mod, "_svc_continue_run", lambda rid: continue_calls.append(rid) or {})
    monkeypatch.setattr(mod, "_svc_cancel_run", lambda rid: cancel_calls.append(rid) or {})

    async def elicitation_callback(context, params):
        pytest.fail("elicitation_callback should NOT be invoked when status != waiting_approval")

    srv = _build()
    async with _client_session(srv, elicitation_callback=elicitation_callback) as client:
        result = await client.call_tool("orchestra_request_approval", {"run_id": "run-xyz"})

    assert not result.isError, f"Unexpected error: {[c.text for c in result.content]}"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    # Should mention that the run is not awaiting approval, include actual status
    assert "running" in text.lower() or "not" in text.lower() or "awaiting" in text.lower(), (
        f"Expected 'not awaiting' explanation with status, got: {text!r}"
    )
    assert continue_calls == []
    assert cancel_calls == []


# ---------------------------------------------------------------------------
# 12. orchestra_request_approval — get_run raises → isError result, not crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_approval_get_run_error_returns_tool_error(monkeypatch):
    """get_run raising (nonexistent) inside the tool → isError result, not a crash."""
    import orchestra.mcp_fastmcp as mod

    monkeypatch.setattr(mod, "_svc_get_run", lambda rid: (_ for _ in ()).throw(ValueError(f"Run not found: {rid}")))

    async def elicitation_callback(context, params):
        pytest.fail("elicitation_callback should NOT be invoked when get_run fails")

    srv = _build()
    async with _client_session(srv, elicitation_callback=elicitation_callback) as client:
        result = await client.call_tool("orchestra_request_approval", {"run_id": "nonexistent"})

    assert result.isError, f"Expected isError=True when get_run raises, got: {result}"
    text = " ".join(c.text for c in result.content if hasattr(c, "text"))
    assert text, "Expected a non-empty error message"
