from __future__ import annotations
import logging
import json
import re
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
    review_changes,
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


# Sentinel returned by _recv_message when a line could not be decoded.
# main() checks for this and continues the loop rather than breaking.
_DECODE_ERROR = object()


def _recv_message() -> dict | object | None:
    """Read and parse one JSON-RPC line from stdin.

    Returns:
    - A parsed dict on success.
    - None on EOF (empty readline result) — signals main() to stop the loop.
    - ``_DECODE_ERROR`` sentinel on JSONDecodeError — signals main() to skip
      the line and continue (I-2: one malformed line must not kill the server).
    - A non-dict value (e.g. list, int) if the JSON is valid but not an object
      — main() must guard against calling .get() on it (I-1).
    """
    line = sys.stdin.readline()
    if not line:
        # EOF — clean shutdown
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        # I-2: log the error but do NOT return None (which would terminate loop)
        sys.stderr.write(f"MCP: malformed JSON input ignored: {exc}\n")
        sys.stderr.flush()
        return _DECODE_ERROR


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
    - ``_METHOD_NOT_FOUND`` sentinel when no handler matched for any input —
      ``main()`` then decides: send a -32601 error response when the message
      carries an ``id`` (request), or stay silent for notifications (no
      ``id``).
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
        # None → EOF: clean shutdown
        if msg is None:
            break
        # _DECODE_ERROR → malformed JSON: skip and continue (I-2)
        if msg is _DECODE_ERROR:
            continue
        # I-1: valid JSON but not a dict (e.g. [1,2,3]) — ignore silently
        if not isinstance(msg, dict):
            sys.stderr.write(
                f"MCP: ignoring non-object JSON message (type={type(msg).__name__})\n"
            )
            sys.stderr.flush()
            continue
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
            "description": (
                "Execution mode: 'ask' (one agent), 'dual' (two agents answer, a verifier checks, a synthesis "
                "merges), 'consensus' (the two agents are assigned opposite stances, for and against the "
                "proposal in the task, and the synthesis decides: use it for a decision or a second opinion "
                "you do not want to be agreeable), or 'auto' (the router picks)."
            ),
        },
        "alias": {"type": "string", "description": "Agent alias override (used with mode='ask')."},
        "previous_alias": {
            "type": "string",
            "description": "Last active agent alias for context continuity.",
        },
        "intent_hint": {
            "type": "string",
            "description": "Optional hint to help the router pick the right mode/agent.",
        },
        "fresh": {
            "type": "boolean",
            "description": "Ask the model again instead of returning a stored answer to the same task (ask mode).",
        },
        "follow_up_to": {
            "type": "string",
            "description": (
                "run_id of an earlier run this task continues. The agents are shown that discussion "
                "before the new task, so the task can be a short follow-up question. In ask mode the "
                "alias may be left out: the agent that answered the earlier run is used."
            ),
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
            "description": (
                "Execution mode: 'ask' (one agent), 'dual' (two agents answer, a verifier checks, a synthesis "
                "merges), 'consensus' (the two agents are assigned opposite stances, for and against the "
                "proposal in the task, and the synthesis decides: use it for a decision or a second opinion "
                "you do not want to be agreeable), or 'auto' (the router picks)."
            ),
        },
        "alias": {
            "type": "string",
            "description": "Agent alias (required when mode='ask').",
        },
        "fresh": {
            "type": "boolean",
            "description": "Ask the model again instead of returning a stored answer to the same task (ask mode).",
        },
        "follow_up_to": {
            "type": "string",
            "description": (
                "run_id of an earlier run this task continues. The agents are shown that discussion "
                "before the new task, so the task can be a short follow-up question. In ask mode the "
                "alias may be left out: the agent that answered the earlier run is used."
            ),
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

_SCHEMA_CONTINUE_RUN = {
    "type": "object",
    "properties": {
        "run_id": {
            "type": "string",
            "description": "Run identifier of the paused run to approve and resume.",
        },
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
        "Use mode='ask' for a single agent, 'dual' for two answers checked and merged, "
        "'consensus' for a decision argued from both sides, or 'auto' for automatic routing."
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
# ── What the agent on the other end may not do ────────────────────────────────
# The caller of these tools is an agent. An approval gate, the settings that decide what
# agents may do, and the tokens in the config belong to the owner.

_OWNER_ONLY_SETTINGS = ("providers.", "remote_nodes.", "gateway.", "dashboard.", "mcp.", "federation.",
                        "approval_policies.", "autonomy_policy.", "limits.", "execution.", "token_budget.")
_SECRET_KEY = re.compile(r"token|secret|password|passwd|api_?key|credential", re.IGNORECASE)
_REDACTED = "[REDACTED]"


def _without_secrets(value, key: str = ""):
    if isinstance(value, dict):
        return {name: _without_secrets(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_without_secrets(item, key) for item in value]
    return _REDACTED if value and _SECRET_KEY.search(key) else value


def _guarded_continue_run(run_id: str) -> dict:
    if not config.mcp_allow_approve():
        raise PermissionError(
            "Approving a run over MCP is off: the approval gate is for the owner. Use "
            "orchestra_request_approval (it asks them in the editor), or `orchestra resume "
            f"{run_id}` in a terminal. To let the calling agent approve: [mcp] allow_approve = true."
        )
    return continue_run(run_id)


def _guarded_config_get(path: str, include_env: bool = True) -> dict:
    result = config_get_value(path, include_env=include_env)
    return {**result, "value": _without_secrets(result.get("value"), path.rsplit(".", 1)[-1])}


def _guarded_config_set(path: str, value, persist: bool = True) -> dict:
    if not config.mcp_allow_config_write():
        raise PermissionError(
            "Changing settings over MCP is off. Edit .orchestra/config.toml, or run "
            f"`orchestra config set {path} <value>`. To allow it: [mcp] allow_config_write = true."
        )
    if (path + ".").startswith(_OWNER_ONLY_SETTINGS):
        raise PermissionError(f"{path} decides what agents may do or spend: change it in the config file.")
    return config_set_value(path, value, persist=persist)


REGISTRY.register(
    "orchestra_continue_run",
    _guarded_continue_run,
    description=(
        "Approve a run paused at an approval gate and resume it. "
        "BLOCKING: returns only after the resumed rounds finish — may take minutes. "
        "To reject instead, call orchestra_cancel_run."
    ),
    input_schema=_SCHEMA_CONTINUE_RUN,
)

# Additional tools — use auto-derived schemas
REGISTRY.register(
    "orchestra_config_get",
    _guarded_config_get,
    description="Get a configuration value at the given dot-path. Tokens and other secrets are redacted.",
)
REGISTRY.register(
    "orchestra_config_set",
    _guarded_config_set,
    description=("Set a configuration value at the given dot-path. Off unless the owner set "
                 "[mcp] allow_config_write = true; permission, budget and token settings cannot be set here."),
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

REGISTRY.register(
    "orchestra_review_changes",
    review_changes,
    description=(
        "Code review of the project's changes by two agents of different model families. Returns "
        "findings with severity, file, line, the quoted line, the problem and a fix. Every finding is "
        "checked: one whose quote is not in the diff is discarded. BLOCKING, usually one to three "
        "minutes. With no arguments it reviews everything not yet committed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "base": {"type": "string", "description": "What to compare with: 'HEAD' (default, uncommitted changes) or a branch/commit such as 'main'."},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Limit the review to these files or folders."},
            "focus": {"type": "string", "description": "What the reviewers should look at in particular."},
            "diff": {"type": "string", "description": "A unified diff to review instead of the project's own changes."},
        },
    },
)

# Sessions, cues and playbooks
REGISTRY.register(
    "orchestra_sessions_list",
    discover_sessions,
    description="List the agent CLI sessions (Claude, Codex, Gemini) found on this machine.",
)
REGISTRY.register(
    "orchestra_sessions_search",
    search_sessions,
    description="Search the discovered agent CLI sessions for a text query.",
)
REGISTRY.register(
    "orchestra_cue_list",
    list_cues,
    description="List the cue rules (event -> action) configured for this project.",
)
REGISTRY.register(
    "orchestra_cue_trigger",
    trigger_cue,
    description=(
        "Fire an event at the cue rules; every rule that matches runs its action, which may "
        "start an agent run. Use orchestra_cue_list first to see what would run."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "description": "The event name the rules match on."},
            "context": {
                "type": "object",
                "description": "Values the rules match on and substitute into their task text.",
            },
        },
        "required": ["event_type"],
    },
)
REGISTRY.register(
    "orchestra_playbook_status",
    get_playbook_status,
    description="Read a Markdown playbook (a checklist of tasks) and return each task with its status.",
)
REGISTRY.register(
    "orchestra_playbook_run",
    run_playbook,
    description=(
        "Run the pending tasks of a Markdown playbook one after another with the given agent "
        "alias, writing each task's status back into the playbook file. BLOCKING: returns when "
        "the last task has finished, which may take minutes."
    ),
)

if __name__ == "__main__":
    main()
