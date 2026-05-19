"""Unified MCP tool registry.

Every MCP tool is registered as a name → callable mapping so that both
JSON-RPC handlers and in-process Fast Lane can dispatch through the same
surface without duplication.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

from orchestra.engine.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)

# Tools that return run-shaped payloads and need _tool_success_run wrapping
_RUN_TOOLS = {
    "orchestra_submit_run",
    "orchestra_run",
    "orchestra_continue",
    "orchestra_wait_for_run",
    "orchestra_execute_job",
    "orchestra_get_run",
}


def _scrub_dict(payload: dict) -> dict:
    from orchestra.engine.pii_scrubber import PiiScrubber
    return PiiScrubber.scrub_dict(payload)


def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _tool_success(payload: dict) -> dict:
    scrubbed = _scrub_dict(payload)
    return {
        "content": [{"type": "text", "text": _json_dumps(scrubbed)}],
        "structuredContent": scrubbed,
        "isError": False,
    }


def _structured_run_result(payload: dict) -> dict:
    run = payload.get("run", {})
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "mode": run.get("mode"),
        "task": run.get("task"),
        "elapsed": run.get("elapsed"),
        "summary": run.get("summary", "")[:500],
        "agent_count": len(run.get("agents", [])),
    }


def _tool_success_run(payload: dict) -> dict:
    structured = dict(payload)
    structured["result"] = _structured_run_result(payload)
    scrubbed = _scrub_dict(structured)
    return {
        "content": [{"type": "text", "text": _json_dumps(scrubbed)}],
        "structuredContent": scrubbed,
        "isError": False,
    }


def _tool_error(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _wrap_result(name: str, result: Any) -> dict:
    if isinstance(result, dict):
        if name in _RUN_TOOLS:
            return _tool_success_run(result)
        return _tool_success(result)
    return _tool_success({"result": result})


class McpRegistry:
    """Registry of MCP tools with metadata."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable, bool]] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        *,
        fastlane: bool = True,
    ) -> None:
        """Register a tool.

        *fn* receives the ``arguments`` dict and must return a dict
        (or any JSON-serialisable value).

        *fastlane=False* forces JSON-RPC path even for in-process calls
        (useful for sensitive tools like shell_exec).
        """
        if name in self._tools:
            logger.warning("MCP tool '%s' re-registered (overwriting previous entry)", name)
        self._tools[name] = (fn, fastlane)
        logger.debug("Registered MCP tool: %s (fastlane=%s)", name, fastlane)

    def get(self, name: str) -> tuple[Callable, bool] | None:
        entry = self._tools.get(name)
        return entry

    def invoke(self, name: str, arguments: dict) -> dict | None:
        """Invoke a registered tool directly.

        Returns ``None`` if the tool is not registered.
        """
        entry = self._tools.get(name)
        if entry is None:
            return None
        fn, _ = entry
        try:
            result = fn(arguments)
            return _wrap_result(name, result)
        except Exception as exc:
            logger.exception("MCP tool '%s' failed", name)
            return _tool_error(str(exc))

    def is_fastlane(self, name: str) -> bool:
        entry = self._tools.get(name)
        if entry is None:
            return False
        _, fastlane = entry
        return fastlane

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())


def _auto_wrap(fn: Callable) -> Callable:
    """Wrap a service function so it accepts an ``arguments`` dict."""
    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    def wrapper(arguments: dict) -> Any:
        kwargs = {}
        for p in params:
            if p in arguments:
                kwargs[p] = arguments[p]
        return fn(**kwargs)

    return wrapper


# Global singleton registry
REGISTRY = McpRegistry()
