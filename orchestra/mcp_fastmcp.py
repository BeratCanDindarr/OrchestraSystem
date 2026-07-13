"""SDK-based MCP bridge server for Orchestra.

Wraps the existing ``REGISTRY`` (orchestra.engine.mcp_registry) and
``POLICY`` (orchestra.engine.policy_engine) behind the official ``mcp``
SDK low-level Server, preserving the same policy-gate and PII-scrub
guarantees as the hand-rolled server (orchestra.mcp_server).

The hand-rolled server is NOT replaced — it remains the no-SDK fallback.
This module is the foundation for server-initiated elicitation (next task).

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
    from orchestra.engine.policy_engine import PolicyEngine  # noqa: PLC0415

    # Module-level POLICY so tests can monkeypatch it.
    import orchestra.mcp_fastmcp as _self  # noqa: PLC0415
    policy = _self.POLICY

    srv = Server("orchestra-mcp")

    @srv.list_tools()
    async def _list_tools() -> list:
        descriptors = REGISTRY.list_tools()
        return [
            types.Tool(
                name=d["name"],
                description=d.get("description", ""),
                inputSchema=d["inputSchema"],
            )
            for d in descriptors
        ]

    @srv.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> types.CallToolResult:
        args = arguments or {}

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


# Module-level POLICY so monkeypatching in tests works correctly.
# Initialised lazily to avoid import-time side-effects when mcp is absent.
def _make_default_policy():
    from orchestra.engine.policy_engine import PolicyEngine  # noqa: PLC0415
    return PolicyEngine.from_default()


try:
    POLICY = _make_default_policy()
except Exception:  # pragma: no cover
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
