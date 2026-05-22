from __future__ import annotations
import logging
import json
import re
import sys
from typing import Any, Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

from orchestra import __version__, config
from orchestra.engine.pii_scrubber import PiiScrubber
from orchestra.engine.policy_engine import PolicyEngine
from orchestra.engine.synthesizer import parse_dissent
from orchestra.protocol import parse_envelope
from orchestra.engine.mcp_registry import REGISTRY, _auto_wrap, _tool_success, _tool_success_run, _tool_error, _json_dumps

from orchestra.service import (
    resume_run,
    run_planned,
    run_critical,
    run_dual,
    run_ask,
    discover_sessions,
    search_sessions,
    resume_session,
    list_cues,
    trigger_cue,
    run_playbook,
    get_playbook_status,
    batch_run_tasks,
    cancel_run,
    compare_runs,
    config_get_value,
    config_init_value,
    config_set_value,
    config_validate_value,
    continue_run,
    create_fleet_run,
    execute_job,
    get_eval_batch,
    get_eval_stats,
    finalize_fleet_run,
    get_logs,
    get_job,
    get_timeline,
    get_trace,
    get_reputation,
    get_run_plan,
    get_version,
    get_toolmaker_proposal,
    get_run,
    inspect_route,
    list_checkpoints,
    memory_add_entry,
    memory_index_project_entries,
    memory_search_entries,
    replay_run_output,
    run_backfill,
    start_daemon,
    daemon_status,
    stop_daemon,
    get_stats,
    get_speculation,
    
    install_toolmaker_proposal,
    list_aliases,
    list_installed_tools,
    list_jobs,
    list_speculations,
    list_toolmaker_proposals,
    list_runs,
    prepare_speculation,
    promote_toolmaker_proposal,
    record_toolmaker_test,
    review_toolmaker_proposal,
    run_fleet_agent,
    run_task,
    submit_speculative_plan,
    submit_run,
    submit_tool_proposal,
    undo_run_changes,
    uninstall_installed_tool,
    wait_for_run,
    watch_run_snapshot,
)

JSONRPC_VERSION = "2.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

# Register all tools
REGISTRY.register("orchestra_submit_run", _auto_wrap(submit_run))
REGISTRY.register("orchestra_run", _auto_wrap(run_task))
REGISTRY.register("orchestra_get_job", _auto_wrap(get_job))
REGISTRY.register("orchestra_list_jobs", _auto_wrap(list_jobs))
REGISTRY.register("orchestra_get_version", _auto_wrap(get_version))
REGISTRY.register("orchestra_list_aliases", _auto_wrap(list_aliases))
REGISTRY.register("orchestra_get_logs", _auto_wrap(get_logs))
REGISTRY.register("orchestra_get_run", _auto_wrap(get_run))
REGISTRY.register("orchestra_list_runs", _auto_wrap(list_runs))
REGISTRY.register("orchestra_get_timeline", _auto_wrap(get_timeline))
REGISTRY.register("orchestra_get_trace", _auto_wrap(get_trace))
REGISTRY.register("orchestra_get_reputation", _auto_wrap(get_reputation))
REGISTRY.register("orchestra_inspect_route", _auto_wrap(inspect_route))
REGISTRY.register("orchestra_list_checkpoints", _auto_wrap(list_checkpoints))
REGISTRY.register("orchestra_replay_run", _auto_wrap(replay_run_output))
REGISTRY.register("orchestra_run_backfill", _auto_wrap(run_backfill))
REGISTRY.register("orchestra_get_stats", _auto_wrap(get_stats))

REGISTRY.register("orchestra_get_stats", _auto_wrap(get_stats))
REGISTRY.register("orchestra_cancel", _auto_wrap(cancel_run))
REGISTRY.register("orchestra_config_get", _auto_wrap(config_get_value))
REGISTRY.register("orchestra_config_set", _auto_wrap(config_set_value))
REGISTRY.register("orchestra_playbook_run", _auto_wrap(run_playbook))
REGISTRY.register("orchestra_playbook_status", _auto_wrap(get_playbook_status))
REGISTRY.register("orchestra_cue_list", _auto_wrap(list_cues))
REGISTRY.register("orchestra_cue_trigger", _auto_wrap(trigger_cue))
REGISTRY.register("orchestra_sessions_list", _auto_wrap(discover_sessions))
REGISTRY.register("orchestra_sessions_search", _auto_wrap(search_sessions))
REGISTRY.register("orchestra_sessions_resume", _auto_wrap(resume_session))
from orchestra.engine.moderator import ModeratorAgent
from orchestra.engine.group_chat import GroupChatState
REGISTRY.register("orchestra_debate_start", _auto_wrap(ModeratorAgent))
REGISTRY.register("orchestra_group_chat_list", _auto_wrap(GroupChatState.list_all))

def _handle_tool_call(name: str, arguments: dict) -> dict:
    try:
        decision = PolicyEngine.from_default().evaluate_tool(name)
        if decision.action == "deny":
            return {"error": f"Tool '{name}' is denied by policy: {decision.reason}"}

        result = REGISTRY.invoke(name, arguments)
        if result is not None:
            return result

        return _tool_error(f"Unknown tool: {name}")
    except Exception as exc:
        logger.exception("Tool handler failed")
        return _tool_error(str(exc))

def _read_message() -> dict | None:
    line = sys.stdin.readline()
    if not line: return None
    try:
        return json.loads(line)
    except:
        return None

def _send_message(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def _handle_request(message: dict) -> dict | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {}) or {}

    if method == "initialize":
        return {
            "protocolVersion": DEFAULT_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "orchestra-mcp", "version": __version__},
        }
    if method == "tools/list":
        return {"tools": []} # Simplified for now
    if method == "tools/call":
        return _handle_tool_call(params.get("name", ""), params.get("arguments", {}) or {})
    
    return None

def main():
    while True:
        msg = _read_message()
        if msg is None: break
        res = _handle_request(msg)
        if res is not None and "id" in msg:
            _send_message({"jsonrpc": "2.0", "id": msg["id"], "result": res})

if __name__ == "__main__":
    main()
