from __future__ import annotations
"""In-process Fast Lane dispatcher for MCP tools.

Bypasses JSON-RPC serialization + stdio round-trip when the caller is
already inside the same Python process.
"""

import functools
import logging
import subprocess
import time
from typing import Any

from orchestra.engine.mcp_registry import REGISTRY, McpRegistry, _tool_error
from orchestra.engine.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)


def _fastlane_timer(fn):
    """Decorator that logs milliseconds saved per Fast Lane call.

    Compares the Fast Lane execution time against an estimated JSON-RPC
    round-trip cost (serialization + stdio + parsing) and logs the delta.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Estimated JSON-RPC overhead: ~5 ms for serde + stdio round-trip
            jsonrpc_overhead_ms = 5.0
            saved_ms = max(0.0, jsonrpc_overhead_ms - elapsed_ms)
            logger.debug(
                "Fast Lane call '%s' took %.2f ms (est. saved %.2f ms vs JSON-RPC)",
                fn.__name__,
                elapsed_ms,
                saved_ms,
            )

    return wrapper


class _ProviderProcessCache:
    """Cache for provider subprocess handles.

    Reuses :class:`subprocess.Popen` instances across calls within a run
    when the provider supports persistent connections.
    """

    def __init__(self) -> None:
        self._handles: dict[str, subprocess.Popen] = {}

    def get(self, alias: str) -> subprocess.Popen | None:
        proc = self._handles.get(alias)
        if proc is not None and proc.poll() is not None:
            # Process has exited — evict stale handle
            del self._handles[alias]
            return None
        return proc

    def set(self, alias: str, proc: subprocess.Popen) -> None:
        self._handles[alias] = proc

    def clear(self) -> None:
        for proc in list(self._handles.values()):
            try:
                proc.terminate()
            except Exception:
                logger.exception("Ignored exception")

        self._handles.clear()

    def __len__(self) -> int:
        return len(self._handles)


class FastLaneDispatcher:
    """Dispatch MCP tools directly without JSON-RPC serde.

    Also supports direct provider dispatch via :meth:`dispatch_provider`,
    caching subprocess handles for reuse across calls.
    """

    def __init__(self, registry: McpRegistry | None = None) -> None:
        self._registry = registry or REGISTRY
        self._provider_cache = _ProviderProcessCache()

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

    @_fastlane_timer
    def dispatch_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Invoke a registered MCP tool directly and return an MCP-compatible result dict.

        Falls back to ``_tool_error`` if the tool is unknown or denied.
        """
        if not self.is_available(name):
            return _tool_error(f"Fast Lane not available for '{name}'")
        # Per-tool autonomy policy check (privileged actions)
        from orchestra.engine.autonomy_policy import AutonomyPolicy

        allowed, reason = AutonomyPolicy.from_config().check_before_tool_call(name, arguments)
        if not allowed:
            return _tool_error(f"Tool '{name}' blocked by autonomy policy: {reason}")
        result = self._registry.invoke(name, arguments)
        if result is None:
            return _tool_error(f"Tool '{name}' returned None from registry")
        return result

    @_fastlane_timer
    def dispatch_provider(self, alias: str, task: str) -> dict:
        """Dispatch *task* to *alias* directly, skipping JSON-RPC envelope creation.

        Attempts to reuse a cached provider process handle; falls back to
        normal provider resolution on cache miss or error.
        """
        try:
            from orchestra.providers.fallback import resolve

            provider, effort = resolve(alias)
        except Exception as exc:
            logger.warning("Fast Lane provider resolve failed for '%s': %s", alias, exc)
            return _tool_error(f"Provider resolve failed for '{alias}': {exc}")

        # Try cached handle first
        cached = self._provider_cache.get(alias)
        if cached is not None and cached.poll() is None:
            try:
                cached.stdin.write(task.encode("utf-8") + b"\n")
                cached.stdin.flush()
                stdout = cached.stdout.read() if cached.stdout else ""
                return {
                    "content": [{"type": "text", "text": stdout}],
                    "isError": False,
                }
            except Exception as exc:
                logger.warning("Fast Lane cached provider write failed for '%s': %s", alias, exc)
                self._provider_cache.clear()

        # Cache miss or error — fall back to spawning a new process
        try:
            cmd = provider.build_command(task, effort)
            popen_kwargs: dict[str, Any] = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
            )
            proc = subprocess.Popen(cmd, **popen_kwargs)
            stdout, stderr = proc.communicate(input=task, timeout=300)
            if proc.returncode != 0:
                return _tool_error(
                    f"Provider '{alias}' exited with code {proc.returncode}: {stderr}"
                )
            return {
                "content": [{"type": "text", "text": stdout}],
                "isError": False,
            }
        except Exception as exc:
            logger.warning("Fast Lane provider dispatch failed for '%s': %s", alias, exc)
            return _tool_error(f"Provider dispatch failed for '{alias}': {exc}")

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict:
        """Backward-compatible alias for :meth:`dispatch_tool`."""
        return self.dispatch_tool(name, arguments)

    def clear_cache(self) -> None:
        """Evict all cached provider process handles."""
        self._provider_cache.clear()


# Global singleton
FASTLANE = FastLaneDispatcher()
