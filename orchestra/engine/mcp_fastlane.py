"""In-process Fast Lane dispatcher for MCP tools.

Bypasses JSON-RPC serialization + stdio round-trip when the caller is
already inside the same Python process.
"""
from __future__ import annotations

import logging
from typing import Any

from orchestra.engine.mcp_registry import REGISTRY, _tool_error
from orchestra.engine.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)


class FastLaneDispatcher:
    """Dispatch MCP tools directly without JSON-RPC serde."""

    def __init__(self, registry: McpRegistry | None = None) -> None:
        self._registry = registry or REGISTRY

    def is_available(self, name: str) -> bool:
        """Return True if the tool can be invoked via Fast Lane."""
        entry = self._registry.get(name)
        if entry is None:
            return False
        _, fastlane = entry
        if not fastlane:
            return False
        decision = PolicyEngine.from_default().evaluate_tool(name)
        if decision.action == "deny":
            return False
        return True

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict:
        """Invoke a tool directly and return an MCP-compatible result dict.

        Falls back to ``_tool_error`` if the tool is unknown or denied.
        """
        if not self.is_available(name):
            return _tool_error(f"Fast Lane not available for '{name}'")
        # Per-tool autonomy policy check ( privileged actions )
        from orchestra.engine.autonomy_policy import AutonomyPolicy
        allowed, reason = AutonomyPolicy.from_config().check_before_tool_call(name, arguments)
        if not allowed:
            return _tool_error(f"Tool '{name}' blocked by autonomy policy: {reason}")
        result = self._registry.invoke(name, arguments)
        if result is None:
            return _tool_error(f"Tool '{name}' returned None from registry")
        return result


# Global singleton
FASTLANE = FastLaneDispatcher()
