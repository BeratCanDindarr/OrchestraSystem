"""Tests for the unified MCP tool registry."""
from __future__ import annotations

import pytest

import orchestra.mcp_server  # side-effect: populates REGISTRY  # noqa: F401
from orchestra.engine.mcp_registry import McpRegistry, REGISTRY, _auto_wrap


class TestMcpRegistry:
    def test_register_and_invoke(self) -> None:
        reg = McpRegistry()
        reg.register("test_add", lambda args: {"sum": args["a"] + args["b"]})
        result = reg.invoke("test_add", {"a": 1, "b": 2})
        assert result is not None
        assert result["isError"] is False

    def test_unknown_tool_returns_none(self) -> None:
        reg = McpRegistry()
        assert reg.invoke("unknown", {}) is None

    def test_list_tools(self) -> None:
        reg = McpRegistry()
        reg.register("z_tool", lambda args: {})
        reg.register("a_tool", lambda args: {})
        assert reg.list_tools() == ["a_tool", "z_tool"]

    def test_fastlane_flag(self) -> None:
        reg = McpRegistry()
        reg.register("fast", lambda args: {}, fastlane=True)
        reg.register("slow", lambda args: {}, fastlane=False)
        assert reg.is_fastlane("fast") is True
        assert reg.is_fastlane("slow") is False

    def test_error_handling(self) -> None:
        reg = McpRegistry()
        reg.register("fail", lambda args: (_ for _ in ()).throw(ValueError("boom")))
        result = reg.invoke("fail", {})
        assert result is not None
        assert result["isError"] is True
        assert "boom" in result["content"][0]["text"]

    def test_run_tools_use_run_wrapper(self) -> None:
        reg = McpRegistry()
        reg.register("orchestra_run", lambda args: {"run": {"run_id": "r1", "status": "completed"}})
        result = reg.invoke("orchestra_run", {})
        assert result is not None
        assert "result" in result["structuredContent"]
        assert result["structuredContent"]["result"]["run_id"] == "r1"

    def test_auto_wrap_kwargs(self) -> None:
        def fn(*, x: int, y: int = 0) -> dict:
            return {"value": x + y}

        wrapped = _auto_wrap(fn)
        assert wrapped({"x": 5}) == {"value": 5}
        assert wrapped({"x": 3, "y": 4}) == {"value": 7}

    def test_global_registry_has_tools(self) -> None:
        # After mcp_server.py imports, the global registry should be populated
        assert "orchestra_run" in REGISTRY.list_tools()
        assert "orchestra_memory_search" in REGISTRY.list_tools()

    def test_duplicate_registration_warning(self, caplog):
        """Re-registering the same tool name should emit a warning."""
        reg = McpRegistry()
        reg.register("dup", lambda args: {"v": 1})
        with caplog.at_level("WARNING", logger="orchestra.engine.mcp_registry"):
            reg.register("dup", lambda args: {"v": 2})
        assert any("re-registered" in rec.message for rec in caplog.records)
