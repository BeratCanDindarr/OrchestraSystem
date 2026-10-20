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
        # The compact run carries run_id and status once; no `result` block repeats them.
        assert "result" not in result["structuredContent"]
        assert result["structuredContent"]["run"]["run_id"] == "r1"

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
            "events": [{"event": "run_started"}, {"event": "approval_requested", "reason": "task contains 'deploy'"},
                       {"event": "agent_fallback", "alias": "kimi-deep", "requested": "kimi", "model": "claude/opus"}],
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
        assert structured["fallbacks"] == [{"alias": "kimi-deep", "answered_by": "claude/opus"}]
        assert structured["timed_out"] is False and "events" not in structured
        assert "result" not in structured

    def test_get_run_still_returns_everything(self) -> None:
        from orchestra.engine.mcp_registry import _wrap_result

        payload = {"run": {"run_id": "r1", "task": "t", "schema_version": 2}, "events": [{"event": "run_started"}]}
        structured = _wrap_result("orchestra_get_run", payload)["structuredContent"]

        assert structured["events"] == [{"event": "run_started"}] and structured["run"]["schema_version"] == 2

    def test_only_the_tools_the_dispatch_skill_calls_are_offered(self) -> None:
        # Every tool's schema lands in the caller's context; config writes stay out of an agent's reach.
        names = {t["name"] for t in REGISTRY.list_tools()}
        assert names == {"orchestra_run", "orchestra_submit_run", "orchestra_wait_for_run", "orchestra_get_run",
                         "orchestra_list_runs", "orchestra_get_logs", "orchestra_review_changes",
                         "orchestra_cancel_run", "orchestra_continue_run", "orchestra_memory_search"}


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

        from unittest.mock import patch

        # Approving over MCP is off by default; this test is about the wiring behind it.
        with patch("orchestra.config.mcp_allow_approve", return_value=True):
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


# ── what the agent on the other end of MCP may not do by default ────────────

class TestMcpBoundary:
    """The caller of these tools is an agent. An approval gate, the config and its tokens
    are the owner's."""

    @staticmethod
    def _invoke(name: str, arguments: dict) -> dict:
        import orchestra.mcp_server as mcp_server  # noqa: F401  (registers the tools)
        from orchestra.engine.mcp_registry import REGISTRY

        return REGISTRY.invoke(name, arguments)

    def test_an_agent_cannot_approve_a_gated_run(self, monkeypatch):
        from unittest.mock import patch

        from orchestra import config

        monkeypatch.setitem(config._cfg, "mcp", {})
        with patch("orchestra.service.resume_run") as resume:
            result = self._invoke("orchestra_continue_run", {"run_id": "some-run"})

        assert result["isError"] is True
        assert "orchestra_request_approval" in result["content"][0]["text"]
        resume.assert_not_called()


class TestReadingToolsStaySmall:
    def test_list_runs_cuts_each_task_and_drops_empty_fields(self) -> None:
        from orchestra.engine.mcp_registry import LIST_TASK_CHARS, compact_run_list

        rows = compact_run_list({"runs": [{"run_id": "r1", "mode": "dual", "status": "completed", "task": "x" * 5000,
                                           "approval_state": "not_required", "interrupt_state": None,
                                           "updated_at": "2026-10-20T10:00:00Z"}]})["runs"]

        assert rows[0]["run_id"] == "r1" and len(rows[0]["task"]) <= LIST_TASK_CHARS + 2
        assert "approval_state" not in rows[0] and "updated_at" not in rows[0]

    def test_logs_default_to_the_tail_of_failed_agents(self) -> None:
        from orchestra.engine.mcp_registry import LOG_TAIL_CHARS, failed_log_tails

        manifest = {"agents": [{"alias": "ok", "status": "completed"}, {"alias": "bad", "status": "failed"}]}
        payload = {"run_id": "r1", "logs": {"ok": "fine", "bad": "y" * 9000 + "THE END"}, "missing": []}

        logs = failed_log_tails(payload, manifest, None)["logs"]

        assert list(logs) == ["bad"] and logs["bad"].endswith("THE END") and len(logs["bad"]) == LOG_TAIL_CHARS

    def test_logs_of_a_clean_run_say_how_to_get_more(self) -> None:
        from orchestra.engine.mcp_registry import failed_log_tails

        result = failed_log_tails({"run_id": "r1", "logs": {"ok": "fine"}}, {"agents": [{"alias": "ok", "status": "completed"}]}, None)

        assert result["logs"] == {} and "full=true" in result["note"]

    def test_a_review_payload_sends_its_findings_once(self) -> None:
        from orchestra.engine.mcp_registry import _compact_run_payload

        compact = _compact_run_payload({"run": {"run_id": "r1", "status": "completed", "summary": "rendered findings"},
                                        "findings": [{"file": "a.py"}]})

        assert "summary" not in compact["run"] and compact["findings"] == [{"file": "a.py"}]


def test_a_synchronous_run_reports_the_alias_another_model_answered_for(monkeypatch):
    from orchestra import service
    from orchestra.engine.mcp_registry import _compact_run_payload
    from orchestra.models import OrchestraRun

    run = OrchestraRun(mode="ask", run_id="r9", task="t", created_at="2026-10-20T10:00:00Z")
    monkeypatch.setattr(service, "run_ask", lambda *a, **k: run)
    monkeypatch.setattr(service.artifacts, "read_events",
                        lambda run_id: [{"event": "agent_fallback", "alias": "cdx-fast", "model": "antigravity/flash"}])

    payload = service.run_task(mode="ask", task="t", alias="cdx-fast")

    assert _compact_run_payload(payload)["fallbacks"] == [{"alias": "cdx-fast", "answered_by": "antigravity/flash"}]
