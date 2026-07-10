from __future__ import annotations
import logging
import json
import sys
from typing import Any

logger = logging.getLogger(__name__)

from orchestra import __version__, config
from orchestra.protocol import parse_envelope
from orchestra.engine.mcp_registry import (
    REGISTRY,
    _auto_wrap,
    _tool_success,
    _tool_success_run,
    _structured_run_result,
    _tool_error,
    _json_dumps,
)

from orchestra.service import (
    run_task,
    continue_run,
    submit_run,
    execute_job,
    get_job,
    wait_for_run,
    list_jobs,
    list_checkpoints,
    cancel_run,
    list_runs,
    get_run,
    get_logs,
    list_aliases,
    get_stats,
    get_reputation,
    get_version,
    run_backfill,
    start_daemon,
    daemon_status,
    stop_daemon,
    config_get_value,
    config_set_value,
    config_validate_value,
    config_init_value,
    memory_add_entry,
    memory_search_entries,
    memory_index_project_entries,
    inspect_route,
    get_trace,
    get_timeline,
    compare_runs,
    replay_run_output,
    batch_run_tasks,
    get_eval_batch,
    get_eval_stats,
    undo_run_changes,
    submit_tool_proposal,
    list_toolmaker_proposals,
    get_toolmaker_proposal,
    review_toolmaker_proposal,
    record_toolmaker_test,
    promote_toolmaker_proposal,
    install_toolmaker_proposal,
    list_installed_tools,
    uninstall_installed_tool,
    submit_speculative_plan,
    get_speculation,
    list_speculations,
    prepare_speculation,
    get_run_plan,
    create_fleet_run,
    run_fleet_agent,
    finalize_fleet_run,
    discover_sessions,
    search_sessions,
    resume_session,
    list_cues,
    trigger_cue,
    run_playbook,
    get_playbook_status,
)
from orchestra.providers.fallback import available_aliases
from orchestra.engine.policy_engine import PolicyEngine

POLICY = PolicyEngine.from_default()

# ---------------------------------------------------------------------------
# Sentinel for "method not found" — avoids isinstance checks in main()
# ---------------------------------------------------------------------------

_METHOD_NOT_FOUND = object()


def _send_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _recv_message() -> dict | None:
    line = sys.stdin.readline()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _handle_tool_call(name: str, arguments: dict) -> dict:
    try:
        decision = POLICY.evaluate_tool(name)
    except Exception as e:
        return _tool_error(f"Policy evaluation failed: {e}")

    if decision.action == "deny":
        return _tool_error(f"Policy denied: {decision.reason}")

    try:
        result = REGISTRY.invoke(name, arguments)
        if result is None:
            return _tool_error(f"Unknown tool: {name}")
        return result
    except Exception as e:
        return _tool_error(f"Internal error: {e}")


def _handle_request(request: dict) -> Any:
    """Handle one JSON-RPC request.

    Returns:
    - A dict to be sent as the ``result`` field when a handler matched.
    - ``_METHOD_NOT_FOUND`` sentinel when no handler matched (only relevant
      for requests that carry an ``id``).
    - ``None`` for notifications (no ``id``) with no matching handler — the
      correct JSON-RPC response is silence.
    """
    if not isinstance(request, dict):
        return None

    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "orchestra-mcp",
                "version": __version__,
            },
        }

    if method == "tools/list":
        return {"tools": REGISTRY.list_tools()}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        if not name:
            return _tool_error("Missing tool name")
        return _handle_tool_call(name, args)

    return _METHOD_NOT_FOUND


def main() -> None:
    while True:
        msg = _recv_message()
        if msg is None:
            break
        res = _handle_request(msg)
        msg_id = msg.get("id")

        if res is _METHOD_NOT_FOUND:
            # Only send error response for requests with an id (not notifications)
            if msg_id is not None:
                method = msg.get("method", "")
                _send_message({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                })
        elif res is not None and msg_id is not None:
            _send_message({"jsonrpc": "2.0", "id": msg_id, "result": res})


# ---------------------------------------------------------------------------
# Explicit schemas for the 7 core tools
# ---------------------------------------------------------------------------

_SCHEMA_ORCHESTRA_RUN = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "The task description to run."},
        "mode": {
            "type": "string",
            "description": "Execution mode: 'ask', 'dual', 'critical', or 'auto'.",
        },
        "alias": {"type": "string", "description": "Agent alias override (used with mode='ask')."},
        "previous_alias": {
            "type": "string",
            "description": "Last active agent alias for context continuity.",
        },
        "pause_after_round1": {
            "type": "boolean",
            "description": "Pause for human approval after the first round (critical mode).",
        },
        "intent_hint": {
            "type": "string",
            "description": "Optional hint to help the router pick the right mode/agent.",
        },
    },
    "required": ["task", "mode"],
}

_SCHEMA_SUBMIT_RUN = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "The task description to submit."},
        "mode": {
            "type": "string",
            "description": "Execution mode: 'ask', 'dual', 'critical', or 'auto'.",
        },
        "alias": {
            "type": "string",
            "description": "Agent alias (required when mode='ask').",
        },
        "pause_after_round1": {
            "type": "boolean",
            "description": "Pause for approval after round 1 (critical mode).",
        },
    },
    "required": ["task", "mode"],
}

_SCHEMA_GET_RUN = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Run identifier to fetch."},
    },
    "required": ["run_id"],
}

_SCHEMA_LIST_RUNS = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "description": "Maximum number of runs to return (default 20)."},
        "mode": {"type": "string", "description": "Filter by mode ('ask', 'dual', etc.)."},
        "status": {"type": "string", "description": "Filter by status ('completed', 'failed', etc.)."},
    },
}

_SCHEMA_GET_LOGS = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Run identifier whose logs to retrieve."},
        "agent": {"type": "string", "description": "Filter to a specific agent alias."},
        "normalized": {"type": "boolean", "description": "Return normalized log lines."},
    },
    "required": ["run_id"],
}

_SCHEMA_WAIT_FOR_RUN = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Run identifier to wait for."},
        "job_id": {"type": "string", "description": "Job identifier to wait for (alternative to run_id)."},
        "timeout_seconds": {"type": "integer", "description": "Maximum seconds to poll (default 30)."},
        "poll_interval_seconds": {"type": "number", "description": "Poll interval in seconds (default 1.0)."},
    },
}

_SCHEMA_CANCEL_RUN = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Run identifier to cancel."},
    },
    "required": ["run_id"],
}

# ---------------------------------------------------------------------------
# Register tools
# ---------------------------------------------------------------------------

REGISTRY.register(
    "orchestra_run",
    run_task,
    description=(
        "Execute an Orchestra task synchronously. Blocks until the run completes. "
        "Use mode='ask' for a single agent, 'dual' for two-agent review, "
        "'critical' for multi-round consensus, or 'auto' for automatic routing."
    ),
    input_schema=_SCHEMA_ORCHESTRA_RUN,
)
REGISTRY.register(
    "orchestra_submit_run",
    submit_run,
    description=(
        "Submit a task as a background job and return immediately with a job_id. "
        "Poll with orchestra_get_run or orchestra_wait_for_run."
    ),
    input_schema=_SCHEMA_SUBMIT_RUN,
)
REGISTRY.register(
    "orchestra_get_run",
    get_run,
    description="Fetch the manifest and event log for a specific run by run_id.",
    input_schema=_SCHEMA_GET_RUN,
)
REGISTRY.register(
    "orchestra_list_runs",
    list_runs,
    description="List recent runs with optional mode/status filters.",
    input_schema=_SCHEMA_LIST_RUNS,
)
REGISTRY.register(
    "orchestra_get_logs",
    get_logs,
    description="Retrieve stdout logs for a run, optionally filtered to one agent.",
    input_schema=_SCHEMA_GET_LOGS,
)
REGISTRY.register(
    "orchestra_wait_for_run",
    wait_for_run,
    description=(
        "Poll a run or job until it reaches a terminal state or the timeout expires. "
        "Provide either run_id or job_id."
    ),
    input_schema=_SCHEMA_WAIT_FOR_RUN,
)
REGISTRY.register(
    "orchestra_cancel_run",
    cancel_run,
    description="Cancel an in-progress run, killing agent processes and marking it cancelled.",
    input_schema=_SCHEMA_CANCEL_RUN,
)

# Additional tools — use auto-derived schemas
REGISTRY.register(
    "orchestra_config_get",
    config_get_value,
    description="Get a configuration value at the given dot-path.",
)
REGISTRY.register(
    "orchestra_config_set",
    config_set_value,
    description="Set a configuration value at the given dot-path.",
)
REGISTRY.register(
    "orchestra_memory_add",
    memory_add_entry,
    description="Add a new entry to the Orchestra memory store.",
)
REGISTRY.register(
    "orchestra_memory_search",
    memory_search_entries,
    description="Search the Orchestra memory store using a natural-language query.",
)

if __name__ == "__main__":
    main()
