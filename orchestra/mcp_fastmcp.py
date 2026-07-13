"""SDK-based MCP bridge server for Orchestra.

Wraps the existing ``REGISTRY`` (orchestra.engine.mcp_registry) and
``POLICY`` (orchestra.engine.policy_engine) behind the official ``mcp``
SDK low-level Server, preserving the same policy-gate and PII-scrub
guarantees as the hand-rolled server (orchestra.mcp_server).

The hand-rolled server is NOT replaced — it remains the no-SDK fallback.
This module adds server-initiated elicitation: ``orchestra_request_approval``
triggers a Claude Code popup that lets the user approve or reject a paused
critical run without leaving the editor.

SDK error-signalling finding
----------------------------
The ``@srv.call_tool()`` handler can signal a tool error (``isError=True``)
in two ways:

1. Return ``types.CallToolResult(content=[...], isError=True)`` directly —
   the low-level Server wraps it in ``ServerResult`` unchanged.
2. Raise an exception — the low-level handler catches it and calls
   ``_make_error_result(str(e))``, which also sets ``isError=True``.

We use approach (1) for explicit policy denials and tool-not-found, and
rely on approach (2) as an implicit fallback.  The ``REGISTRY.invoke``
already returns ``_tool_error(...)`` dicts (``isError=True``) for runtime
failures, so we inspect that flag and return the appropriate
``CallToolResult``.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level service function refs — monkeypatchable in tests
# ---------------------------------------------------------------------------

def _svc_get_run(run_id: str) -> dict:
    from orchestra import service  # noqa: PLC0415
    return service.get_run(run_id)


def _svc_continue_run(run_id: str) -> dict:
    from orchestra import service  # noqa: PLC0415
    return service.continue_run(run_id)


def _svc_cancel_run(run_id: str) -> dict:
    from orchestra import service  # noqa: PLC0415
    return service.cancel_run(run_id)


# ---------------------------------------------------------------------------
# Lazy SDK availability check
# ---------------------------------------------------------------------------

def _require_sdk():
    """Import and return (Server, types) or raise RuntimeError with install hint."""
    try:
        from mcp.server.lowlevel import Server  # noqa: PLC0415
        import mcp.types as types  # noqa: PLC0415
        return Server, types
    except (ImportError, ModuleNotFoundError):
        raise RuntimeError(
            "orchestra MCP SDK server requires the 'mcp' extra: "
            "pip install -e '.[mcp]'"
        )


# ---------------------------------------------------------------------------
# Approval tool schema
# ---------------------------------------------------------------------------

_APPROVAL_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {
            "type": "string",
            "description": "The Orchestra run ID to request approval for.",
        }
    },
    "required": ["run_id"],
}

_APPROVAL_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approve", "reject"],
            "description": "Choose 'approve' to continue round 2 or 'reject' to cancel the run.",
        }
    },
    "required": ["decision"],
}


# ---------------------------------------------------------------------------
# Elicitation helpers
# ---------------------------------------------------------------------------

def _build_elicitation_message(run: dict) -> str:
    """Build a concise human-readable summary of the run for the elicitation dialog."""
    run_id = run.get("run_id", "?")
    mode = run.get("mode", "?")
    summary = run.get("summary") or ""
    agents = run.get("agents", [])
    alias_list = ", ".join(a.get("alias", "?") for a in agents) if agents else "none"

    lines = [
        f"Orchestra run {run_id!r} is paused and awaiting approval.",
        f"  Mode: {mode}",
        f"  Agents: {alias_list}",
    ]
    if summary:
        lines.append(f"  Round-1 summary: {summary}")
    lines.append("Approve to continue round 2, or reject to cancel the run.")
    return "\n".join(lines)


def _make_text_result(types, text: str, *, is_error: bool = False):
    """Return a CallToolResult with a single TextContent block."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        isError=is_error,
    )


async def _handle_approval_elicitation(session, types, run: dict) -> tuple[str, bool]:
    """Call session.elicit_form and interpret the result.

    Returns (message, is_error) — caller builds the final CallToolResult.
    """
    message = _build_elicitation_message(run)
    result = await session.elicit(message, _APPROVAL_DECISION_SCHEMA)
    action = result.action
    content = result.content or {}

    if action == "accept":
        decision = content.get("decision", "reject")
        if decision == "approve":
            return "approve", False
        else:
            return "reject", False
    elif action == "decline":
        return "decline", False
    else:
        # action == "cancel"
        return "cancel", False


async def _dispatch_approval_decision(
    decision: str,
    run_id: str,
    types,
):
    """Invoke the appropriate service function and return a CallToolResult."""
    import orchestra.mcp_fastmcp as _self  # noqa: PLC0415

    if decision == "approve":
        try:
            resp = _self._svc_continue_run(run_id)
        except Exception as exc:
            return _make_text_result(types, f"Failed to resume run {run_id!r}: {exc}", is_error=True)
        status = resp.get("run", {}).get("status", "?")
        return _make_text_result(
            types,
            f"Run {run_id!r} approved and resumed. Current status: {status}.",
        )
    else:
        # reject / decline / cancel — all cancel the run
        try:
            _self._svc_cancel_run(run_id)
        except Exception as exc:
            return _make_text_result(types, f"Failed to cancel run {run_id!r}: {exc}", is_error=True)
        label = {"reject": "rejected", "decline": "declined", "cancel": "cancelled"}.get(decision, "cancelled")
        return _make_text_result(
            types,
            f"Run {run_id!r} {label} and cancelled.",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_server():
    """Build and return a configured low-level MCP Server.

    Imports ``orchestra.mcp_server`` as a side-effect so that all tools are
    registered into REGISTRY before building the handler list.

    Returns:
        mcp.server.lowlevel.Server
    """
    Server, types = _require_sdk()

    # Side-effect import: registers all ~13+ tools into REGISTRY.
    import orchestra.mcp_server  # noqa: F401, PLC0415

    from orchestra.engine.mcp_registry import REGISTRY  # noqa: PLC0415

    # Module-level POLICY so tests can monkeypatch it.
    import orchestra.mcp_fastmcp as _self  # noqa: PLC0415
    policy = _self.POLICY

    srv = Server("orchestra-mcp")

    @srv.list_tools()
    async def _list_tools() -> list:
        descriptors = REGISTRY.list_tools()
        tool_list = [
            types.Tool(
                name=d["name"],
                description=d.get("description", ""),
                inputSchema=d["inputSchema"],
            )
            for d in descriptors
        ]
        # Append the SDK-only approval tool (not in REGISTRY).
        tool_list.append(
            types.Tool(
                name="orchestra_request_approval",
                description=(
                    "Trigger a Claude Code elicitation dialog to approve or reject "
                    "a paused Orchestra run that is waiting_approval. "
                    "The tool presents run details and a decision form inline."
                ),
                inputSchema=_APPROVAL_TOOL_SCHEMA,
            )
        )
        return tool_list

    @srv.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        args = arguments or {}

        # ---- SDK-only approval tool — handled before REGISTRY ----
        if name == "orchestra_request_approval":
            return await _handle_request_approval(srv, types, args)

        # Policy gate — mirrors _handle_tool_call in mcp_server.py.
        try:
            decision = policy.evaluate_tool(name)
        except Exception as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Policy evaluation failed: {exc}")],
                isError=True,
            )

        if decision.action == "deny":
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=f"Policy denied tool {name!r}: {decision.reason}",
                )],
                isError=True,
            )

        # Invoke through the registry (PII scrub is inside REGISTRY._tool_success).
        result = REGISTRY.invoke(name, args)
        if result is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        is_error = bool(result.get("isError", False))
        raw_content = result.get("content", [])
        content_blocks = [
            types.TextContent(type="text", text=item.get("text", ""))
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if not content_blocks:
            content_blocks = [types.TextContent(type="text", text="")]

        return types.CallToolResult(content=content_blocks, isError=is_error)

    return srv


async def _handle_request_approval(srv, types, args: dict):
    """Handle the orchestra_request_approval tool call.

    Steps:
    1. Load run status; if not waiting_approval, return early without eliciting.
    2. Elicit a decision from the user via the MCP session.
    3. Dispatch approve → continue_run, reject/decline/cancel → cancel_run.
    """
    import orchestra.mcp_fastmcp as _self  # noqa: PLC0415

    run_id = args.get("run_id", "")
    if not run_id:
        return _make_text_result(types, "run_id is required.", is_error=True)

    # Step 1: load run status.
    try:
        resp = _self._svc_get_run(run_id)
    except Exception as exc:
        return _make_text_result(types, f"Failed to load run {run_id!r}: {exc}", is_error=True)

    run = resp.get("run", {})
    status = run.get("status", "unknown")

    if status != "waiting_approval":
        return _make_text_result(
            types,
            f"Run {run_id!r} is not awaiting approval (current status: {status!r}). "
            "Nothing to do.",
        )

    # Step 2: elicit decision from user.
    try:
        session = srv.request_context.session
        decision, _ = await _handle_approval_elicitation(session, types, run)
    except Exception as exc:
        return _make_text_result(types, f"Elicitation failed for run {run_id!r}: {exc}", is_error=True)

    # Step 3: dispatch.
    return await _dispatch_approval_decision(decision, run_id, types)


# Module-level POLICY so monkeypatching in tests works correctly.
# Initialised lazily to avoid import-time side-effects when mcp is absent.
def _make_default_policy():
    from orchestra.engine.policy_engine import PolicyEngine  # noqa: PLC0415
    return PolicyEngine.from_default()


try:
    POLICY = _make_default_policy()
except Exception as _exc:  # pragma: no cover
    logger.warning("Failed to build default PolicyEngine; tool calls will error: %s", _exc)
    POLICY = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run() -> None:
    from mcp.server.stdio import stdio_server  # noqa: PLC0415

    srv = build_server()
    async with stdio_server() as (read, write):
        await srv.run(read, write, srv.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
