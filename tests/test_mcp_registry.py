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

    def test_a_run_tool_returns_the_answer_without_repeating_the_task(self) -> None:
        from orchestra.engine.mcp_registry import _wrap_result

        task = "A long task. " * 200
        payload = {
            "job": {"job_id": "j1", "run_id": "r1", "status": "completed", "task": task, "worker_pid": 7},
            "run": {"run_id": "r1", "mode": "dual", "status": "completed", "task": task, "summary": "The answer. " * 100,
                    "agents": [{"alias": "cld-deep", "model": "claude/opus", "status": "completed",
                                "stdout_digest": "abc", "pid": None, "soft_failed": False}],
                    "total_cost_usd": 0.01, "schema_version": 2},
            "events": [{"event": "run_started"}, {"event": "approval_requested", "reason": "task contains 'deploy'"}],
            "timed_out": False,
        }

        wrapped = _wrap_result("orchestra_wait_for_run", payload)
        text = wrapped["content"][0]["text"]
        structured = wrapped["structuredContent"]

        assert len(text) < len(task)
        assert structured["run"]["task"].endswith("…") and len(structured["run"]["task"]) < 220
        assert "task" not in structured["job"]
        assert structured["run"]["summary"] == "The answer. " * 100
        assert structured["run"]["agents"] == [
            {"alias": "cld-deep", "model": "claude/opus", "status": "completed", "soft_failed": False}]
        assert structured["approval_reason"] == "task contains 'deploy'"
        assert structured["timed_out"] is False and "events" not in structured
        assert structured["result"]["run_id"] == "r1"

    def test_get_run_still_returns_everything(self) -> None:
        from orchestra.engine.mcp_registry import _wrap_result

        payload = {"run": {"run_id": "r1", "task": "t", "schema_version": 2}, "events": [{"event": "run_started"}]}
        structured = _wrap_result("orchestra_get_run", payload)["structuredContent"]

        assert structured["events"] == [{"event": "run_started"}] and structured["run"]["schema_version"] == 2

    def test_the_tools_the_readme_names_are_registered(self) -> None:
        names = {t["name"] for t in REGISTRY.list_tools()}
        for expected in ("orchestra_sessions_list", "orchestra_sessions_search", "orchestra_cue_list",
                         "orchestra_cue_trigger", "orchestra_playbook_status", "orchestra_playbook_run"):
            assert expected in names

    def test_playbook_status_reads_a_playbook_through_the_registry(self, tmp_path) -> None:
        playbook = tmp_path / "steps.md"
        playbook.write_text("# Steps\n\n- [ ] first step\n- [x] second step\n", encoding="utf-8")

        result = REGISTRY.invoke("orchestra_playbook_status", {"path": str(playbook)})

        assert result is not None and not result.get("isError")
        tasks = result["structuredContent"]["playbook"]["tasks"]
        assert [task["description"] for task in tasks] == ["first step", "second step"]

    def test_cue_trigger_takes_its_context_as_an_object(self) -> None:
        tools_by_name = {t["name"]: t for t in REGISTRY.list_tools()}
        props = tools_by_name["orchestra_cue_trigger"]["inputSchema"]["properties"]
        assert props["event_type"]["type"] == "string"
        assert props["context"]["type"] == "object"

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

    # ------------------------------------------------------------------
    # C-1 regression: single-param service functions must NOT be treated
    # as legacy (arguments: dict) callbacks just because they have one param.
    # ------------------------------------------------------------------

    def test_single_str_param_fn_receives_string_not_dict(self) -> None:
        """Registry must NOT treat def f(run_id: str) as a legacy callback.

        Regression for C-1: the old heuristic matched any function with exactly
        one positional parameter, including service functions like
        get_run(run_id: str).  REGISTRY.invoke must pass the STRING value, not
        the whole arguments dict.
        """
        received: list = []

        def f(run_id: str) -> dict:
            received.append(run_id)
            return {"run_id": run_id}

        reg = McpRegistry()
        reg.register("test_get_run", f)
        result = reg.invoke("test_get_run", {"run_id": "abc-123"})
        assert result is not None
        assert result["isError"] is False
        # The function must have received the string "abc-123", not a dict
        assert received == ["abc-123"], (
            f"Expected fn to receive 'abc-123' (str), got: {received}"
        )

    def test_legacy_dict_callback_still_works(self) -> None:
        """A function whose sole param is named 'arguments' is still a legacy callback."""

        def legacy(arguments: dict) -> dict:
            return {"echo": arguments.get("x")}

        reg = McpRegistry()
        reg.register("test_legacy", legacy)
        result = reg.invoke("test_legacy", {"x": "hello"})
        assert result is not None
        assert result["isError"] is False
        # Legacy callback receives the full dict; it extracts 'x' itself
        assert result["structuredContent"]["echo"] == "hello"

    def test_legacy_dict_annotated_callback_still_works(self) -> None:
        """A function annotated as (args: dict) is treated as a legacy callback."""

        def legacy2(args: dict) -> dict:
            return {"echo": args.get("y")}

        reg = McpRegistry()
        reg.register("test_legacy2", legacy2)
        result = reg.invoke("test_legacy2", {"y": "world"})
        assert result is not None
        assert result["isError"] is False
        assert result["structuredContent"]["echo"] == "world"

    def test_orchestra_get_run_type_error_regression(self) -> None:
        """REGISTRY.invoke('orchestra_get_run', {'run_id': 'nonexistent'}) must not
        raise a type error about PosixPath / dict.

        The error (if any) must be a not-found style error (KeyError, ValueError,
        or a clean tool-level isError response), NOT a TypeError about unsupported
        operand types.
        """
        import orchestra.mcp_server  # ensure registration runs  # noqa: F401

        result = REGISTRY.invoke("orchestra_get_run", {"run_id": "nonexistent-run-id"})
        assert result is not None
        # If isError, the message must not mention type errors about dict/PosixPath
        if result.get("isError"):
            err_text = result["content"][0]["text"]
            assert "unsupported operand type" not in err_text, (
                f"C-1 regression: got a type error instead of a not-found error: {err_text}"
            )
            assert "PosixPath" not in err_text, (
                f"C-1 regression: PosixPath/dict type error still present: {err_text}"
            )

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

    # ------------------------------------------------------------------
    # Task 27 (Phase 3a): orchestra_continue_run registry invoke
    # ------------------------------------------------------------------

    def test_continue_run_invoke_nonexistent_returns_clean_error(self) -> None:
        """REGISTRY.invoke('orchestra_continue_run', {'run_id': 'nonexistent'}) must
        return a clean tool error, not a TypeError or AttributeError.

        Mirrors the get_run regression pattern (test_orchestra_get_run_type_error_regression).
        This also confirms the single-param wrap fix applies to continue_run(run_id: str).
        """
        import orchestra.mcp_server  # ensure registration runs  # noqa: F401

        result = REGISTRY.invoke("orchestra_continue_run", {"run_id": "nonexistent-xyz-789"})
        assert result is not None, "Expected a result dict, got None (tool not registered?)"
        assert result.get("isError") is True, (
            f"Expected a not-found tool error for a nonexistent run, got: {result}"
        )
        err_text = result["content"][0]["text"]
        # The string run_id must have reached the service: a clean not-found error,
        # never a dict/type error from a mis-wrapped single-param function.
        assert "not found" in err_text.lower(), (
            f"Expected a 'not found' error, got: {err_text}"
        )
        assert "unsupported operand type" not in err_text, (
            f"Got a type error instead of a not-found error: {err_text}"
        )

    def test_continue_run_in_run_tools_set(self) -> None:
        """orchestra_continue_run must be in _RUN_TOOLS so its result gets the run wrapper."""
        from orchestra.engine.mcp_registry import _RUN_TOOLS
        assert "orchestra_continue_run" in _RUN_TOOLS, (
            f"orchestra_continue_run missing from _RUN_TOOLS; current set: {_RUN_TOOLS}"
        )
