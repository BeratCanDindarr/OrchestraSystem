from __future__ import annotations
import logging
import json
import sys
from typing import Any

logger = logging.getLogger(__name__)

from orchestra import __version__, config
from orchestra.engine.mcp_registry import (
    REGISTRY,
    _tool_error,
    _compact_run_payload,
    compact_run_list,
    failed_log_tails,
)

from orchestra.service import (
    run_task,
    continue_run,
    submit_run,
    wait_for_run,
    cancel_run,
    list_runs,
    get_run,
    get_logs,
    memory_search_entries,
    review_changes,
)
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
        message: dict | object = json.loads(line)
        return message
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
        "alias": {"type": "string", "description": "Agent alias (mode='ask'). Optional in ask mode: left out, Orchestra picks by the task (a web-capable agent when it needs the web, else the first agent of the task's tier)."},
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
            "description": "Agent alias (mode='ask'). Optional in ask mode: left out, Orchestra picks by the task (a web-capable agent when it needs the web, else the first agent of the task's tier).",
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
        "detail": {"type": "boolean", "description": "Whole manifest and every event (large). Default: the compact run."},
    },
    "required": ["run_id"],
}

_SCHEMA_LIST_RUNS = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer", "description": "Maximum number of runs to return (default 10)."},
        "mode": {"type": "string", "description": "Filter by mode ('ask', 'dual', etc.)."},
        "status": {"type": "string", "description": "Filter by status ('completed', 'failed', etc.)."},
        "include_steps": {"type": "boolean", "description": "Also list internal steps (verifier, synthesiser)."},
    },
}

_SCHEMA_GET_LOGS = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Run identifier whose logs to retrieve."},
        "agent": {"type": "string", "description": "Filter to a specific agent alias."},
        "full": {"type": "boolean", "description": "Every agent's whole log (large). Default: the last 2000 characters of failed agents, or of `agent`."},
    },
    "required": ["run_id"],
}

_SCHEMA_WAIT_FOR_RUN = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Run identifier to wait for."},
        "job_id": {"type": "string", "description": "Job identifier to wait for (alternative to run_id)."},
        "timeout_seconds": {"type": "integer", "description": "Maximum seconds to wait (default 240; a dual run takes about 3-4 minutes)."},
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
def _get_run_tool(run_id: str, detail: bool = False) -> dict:
    payload = get_run(run_id)
    return payload if detail else _compact_run_payload(payload)


def _list_runs_tool(limit: int = 10, mode: str | None = None, status: str | None = None,
                    include_steps: bool = False) -> dict:
    return compact_run_list(list_runs(limit=limit, mode=mode, status=status, include_steps=include_steps))


def _get_logs_tool(run_id: str, agent: str | None = None, full: bool = False) -> dict:
    payload = get_logs(run_id, agent=agent)
    return payload if full else failed_log_tails(payload, get_run(run_id)["run"], agent)


def _wait_for_run_tool(run_id: str | None = None, job_id: str | None = None, timeout_seconds: int = 240,
                       poll_interval_seconds: float = 1.0) -> dict:
    return wait_for_run(run_id=run_id, job_id=job_id, timeout_seconds=timeout_seconds,
                        poll_interval_seconds=poll_interval_seconds)


REGISTRY.register(
    "orchestra_get_run",
    _get_run_tool,
    description="A run's state and answer by run_id; detail=true for the whole manifest and events.",
    input_schema=_SCHEMA_GET_RUN,
)
REGISTRY.register(
    "orchestra_list_runs",
    _list_runs_tool,
    description="List recent runs with optional mode/status filters.",
    input_schema=_SCHEMA_LIST_RUNS,
)
REGISTRY.register(
    "orchestra_get_logs",
    _get_logs_tool,
    description="The end of a run's failed agents' logs (or one agent's); full=true for every log in full.",
    input_schema=_SCHEMA_GET_LOGS,
)
REGISTRY.register(
    "orchestra_wait_for_run",
    _wait_for_run_tool,
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
# The caller of these tools is an agent: approving a paused run belongs to the owner.


def _guarded_continue_run(run_id: str) -> dict:
    if not config.mcp_allow_approve():
        raise PermissionError(
            "Approving a run over MCP is off: the approval gate is for the owner. Use "
            "orchestra_request_approval (it asks them in the editor), or `orchestra resume "
            f"{run_id}` in a terminal. To let the calling agent approve: [mcp] allow_approve = true."
        )
    return continue_run(run_id)


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
            "focus": {"type": "string", "description": "What the reviewers should look at in particular; 'unity' for the Unity checklist (added by itself when the change touches C# files)."},
            "diff": {"type": "string", "description": "A unified diff to review instead of the project's own changes."},
        },
    },
)

if __name__ == "__main__":
    main()
