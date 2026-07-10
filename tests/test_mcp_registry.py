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
        # list_tools() returns MCP tool descriptors (dicts), sorted by name.
        tools = reg.list_tools()
        assert [t["name"] for t in tools] == ["a_tool", "z_tool"]

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
        # After mcp_server.py imports, the global registry should be populated.
        # list_tools() returns MCP tool descriptors; check by name.
        names = {t["name"] for t in REGISTRY.list_tools()}
        assert "orchestra_run" in names
        assert "orchestra_memory_search" in names

    def test_duplicate_registration_warning(self, caplog):
        """Re-registering the same tool name should emit a warning."""
        reg = McpRegistry()
        reg.register("dup", lambda args: {"v": 1})
        with caplog.at_level("WARNING", logger="orchestra.engine.mcp_registry"):
            reg.register("dup", lambda args: {"v": 2})
        assert any("re-registered" in rec.message for rec in caplog.records)

    def test_derive_schema_pep563_stringified_annotations(self) -> None:
        """_derive_schema must correctly resolve stringified annotations.

        This test file itself has ``from __future__ import annotations``, so
        any function defined here will have its annotations stored as strings
        at runtime (PEP 563).  The registry must still emit "boolean" /
        "integer" rather than "string" for those parameters.
        """
        # Function defined in this module — annotations are strings at runtime.
        def fn_with_typed_params(name: str, count: int, enabled: bool = True) -> dict:
            return {}

        reg = McpRegistry()
        reg.register("typed_fn", fn_with_typed_params, description="test PEP 563")
        schema = reg.list_tools()[0]["inputSchema"]
        props = schema["properties"]

        assert props["name"]["type"] == "string", f"Expected 'string', got {props['name']['type']}"
        assert props["count"]["type"] == "integer", f"Expected 'integer', got {props['count']['type']}"
        assert props["enabled"]["type"] == "boolean", f"Expected 'boolean', got {props['enabled']['type']}"

    def test_real_service_functions_have_correct_schema_types(self) -> None:
        """Verify boolean/integer annotations on real service functions resolve correctly.

        orchestra/service.py uses ``from __future__ import annotations`` (PEP 563),
        so param.annotation would be a string at runtime without get_type_hints().
        """
        import orchestra.mcp_server  # ensure registrations run  # noqa: F401

        tools_by_name = {t["name"]: t for t in REGISTRY.list_tools()}

        # config_get_value(path: str, include_env: bool = True)
        cfg_get = tools_by_name.get("orchestra_config_get")
        if cfg_get is not None:
            props = cfg_get["inputSchema"]["properties"]
            assert props.get("include_env", {}).get("type") == "boolean", (
                f"config_get include_env should be 'boolean', got {props.get('include_env')}"
            )

        # config_set_value(path: str, value, persist: bool = True)
        cfg_set = tools_by_name.get("orchestra_config_set")
        if cfg_set is not None:
            props = cfg_set["inputSchema"]["properties"]
            assert props.get("persist", {}).get("type") == "boolean", (
                f"config_set persist should be 'boolean', got {props.get('persist')}"
            )

        # memory_search_entries(query: str, limit: int = 5)
        mem_search = tools_by_name.get("orchestra_memory_search")
        if mem_search is not None:
            props = mem_search["inputSchema"]["properties"]
            assert props.get("limit", {}).get("type") == "integer", (
                f"memory_search limit should be 'integer', got {props.get('limit')}"
            )
