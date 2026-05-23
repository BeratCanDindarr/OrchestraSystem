from __future__ import annotations
import logging
import json
import re
import sys
from typing import Any, Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

from orchestra import __version__, config
from orchestra.protocol import parse_envelope
from orchestra.engine.mcp_registry import REGISTRY, _auto_wrap, _tool_success, _tool_success_run, _tool_error, _json_dumps

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

def _send_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def _recv_message() -> dict | None:
    line = sys.stdin.readline()
    if not line: return None
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

def _handle_request(request: dict) -> dict | None:
    if not isinstance(request, dict): return None
    
    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(params, dict): params = {}

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                }
            },
            "serverInfo": {
                "name": "orchestra-mcp",
                "version": __version__
            }
        }
    
    if method == "tools/list":
        # Simplified list for brevity in code read, real implementation would populate properly
        return {"tools": [
            {"name": "orchestra_run", "description": "Run an Orchestra task.", "inputSchema": {"type": "object", "properties": {"task": {"type": "string"}, "mode": {"type": "string"}}}}
        ]}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict): args = {}
        if not name: return _tool_error("Missing tool name")
        return _handle_tool_call(name, args)

    return None

# Register tools with correct service mappings
REGISTRY.register("orchestra_run", _auto_wrap(run_task))
REGISTRY.register("orchestra_submit_run", _auto_wrap(submit_run))
REGISTRY.register("orchestra_get_run", _auto_wrap(get_run))
REGISTRY.register("orchestra_list_runs", _auto_wrap(list_runs))
REGISTRY.register("orchestra_get_logs", _auto_wrap(get_logs))
REGISTRY.register("orchestra_config_get", _auto_wrap(config_get_value))
REGISTRY.register("orchestra_config_set", _auto_wrap(config_set_value))
REGISTRY.register("orchestra_memory_add", _auto_wrap(memory_add_entry))
REGISTRY.register("orchestra_memory_search", _auto_wrap(memory_search_entries))

def main():
    while True:
        msg = _recv_message()
        if msg is None: break
        res = _handle_request(msg)
        if res is not None and "id" in msg:
            _send_message({"jsonrpc": "2.0", "id": msg["id"], "result": res})

if __name__ == "__main__":
    main()
