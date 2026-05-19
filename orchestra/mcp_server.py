"""Minimal stdio MCP server for Orchestra."""
from __future__ import annotations

import json
import re
import sys
from typing import Any
from urllib.parse import unquote

from orchestra import __version__
from orchestra.engine.pii_scrubber import PiiScrubber
from orchestra.engine.policy_engine import PolicyEngine
from orchestra.engine.synthesizer import parse_dissent
from orchestra.protocol import parse_envelope
from orchestra.service import (
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
    execute_speculation,
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
_TRANSPORT_MODE: str | None = None

_FLEET_SKILL_CONTENT = """\
# Orchestra Fleet Workflow

Bu skill'i kullan: kullanıcı "fleet", "fleet mode" veya "/mcp__orchestra__fleet" dediğinde.

## Adımlar

### 1. Plan Al
`orchestra_get_run_plan(task=<görev>, mode=<mod>)` çağır.
`providers` listesini ve `has_postprocess` değerini kaydet.

### 2. Fleet Run Oluştur
`orchestra_create_fleet_run(task=<görev>, mode=<mod>)` çağır.
Dönen `run_id`'yi kaydet.

### 3. Her Provider İçin Background Agent Başlat (Paralel)
`providers` listesindeki her alias için `run_in_background: true` Agent başlat.
- Agent description: "Orchestra/<alias>" — örnek: "Orchestra/cdx-deep"
- Agent görevi: şu tool'u çağır ve sonucu döndür:
  `orchestra_run_fleet_agent(run_id=<run_id>, alias=<alias>)`

TÜM provider agent'larının tamamlanmasını bekle.

### 4. Finalize Agent'ı Başlat (Eğer has_postprocess: true ise)
`run_in_background: true` ile tek bir Agent başlat.
- Agent description: "Orchestra/finalize"
- Agent görevi: şu tool'u çağır ve sonucu döndür:
  `orchestra_finalize_fleet_run(run_id=<run_id>)`

Finalize agent'ının tamamlanmasını bekle.

### 5. Sonucu Sun
`orchestra_get_run(run_id=<run_id>)` çağır.
`result.answer` alanını kullanıcıya sun.

## Kritik Kurallar
- `run_in_background: true` ZORUNLU — aksi hâlde fleet view'da görünmez.
- Provider agent'ları her zaman paralel başlatılır.
- Finalize adımı yalnızca TÜM provider'lar bittikten sonra başlar.
- Herhangi bir agent failed dönerse kullanıcıyı bildir; `orchestra_get_run` ile kısmi sonuçları sun.
"""


TOOL_DEFINITIONS = [
    {
        "name": "orchestra_submit_run",
        "description": "Submit an Orchestra task as an async background job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["ask", "auto", "dual", "critical"],
                    "description": "Execution mode.",
                },
                "task": {
                    "type": "string",
                    "description": "Prompt/task to send into Orchestra.",
                },
                "alias": {
                    "type": "string",
                    "description": "Required only when mode is ask.",
                },
                "pause_after_round1": {
                    "type": "boolean",
                    "description": "Pause critical runs after round 1 so they can be resumed later.",
                    "default": False,
                },
            },
            "required": ["mode", "task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_job",
        "description": "Poll an async Orchestra job and hydrate its linked run when available.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_jobs",
        "description": "List recent async Orchestra jobs with optional status filtering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "default": 20},
                "status": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_wait_for_run",
        "description": "Convenience wrapper for short waits: poll a run or async job until it reaches a terminal or waiting-approval state. For long-running tasks, prefer orchestra_submit_run plus orchestra_get_job polling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "job_id": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 0, "default": 30},
                "poll_interval_seconds": {"type": "number", "minimum": 0.1, "default": 1.0},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_run",
        "description": "Run an Orchestra task in ask, auto, dual, or critical mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["ask", "auto", "dual", "critical"],
                    "description": "Execution mode.",
                },
                "task": {
                    "type": "string",
                    "description": "Prompt/task to send into Orchestra.",
                },
                "alias": {
                    "type": "string",
                    "description": "Required only when mode is ask.",
                },
                "pause_after_round1": {
                    "type": "boolean",
                    "description": "Pause critical runs after round 1 so they can be resumed later.",
                    "default": False,
                },
            },
            "required": ["mode", "task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_continue",
        "description": "Continue a critical Orchestra run that is waiting for approval after round 1.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_watch_run",
        "description": "Return a combined run snapshot with manifest, events, and logs for watch-style polling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "normalized": {"type": "boolean", "default": False},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_cancel",
        "description": "Cancel a running Orchestra run using the persisted PID data in its manifest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_aliases",
        "description": "List available Orchestra aliases and provider availability.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orchestra_get_stats",
        "description": "Return aggregate Orchestra run statistics.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orchestra_get_reputation",
        "description": "Get alias reputation scores and supporting metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "alias": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_submit_tool_proposal",
        "description": "Create a pending-approval Tool-Maker proposal with optional staged files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "run_id": {"type": "string"},
                "source_alias": {"type": "string"},
                "test_command": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["name", "description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_tool_proposals",
        "description": "List Tool-Maker proposals with optional status filtering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_tool_proposal",
        "description": "Read a Tool-Maker proposal and its staged bundle metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
            },
            "required": ["proposal_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_review_tool_proposal",
        "description": "Apply manual approval or rejection to a Tool-Maker proposal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "note": {"type": "string"},
            },
            "required": ["proposal_id", "approve"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_record_tool_test",
        "description": "Record the result of manual or external tests for a Tool-Maker proposal.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "status": {"type": "string", "enum": ["passed", "failed"]},
                "summary": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["proposal_id", "status", "summary"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_promote_tool_proposal",
        "description": "Promote an approved, test-passing Tool-Maker proposal into the staged tool library.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
            },
            "required": ["proposal_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_install_tool_proposal",
        "description": "Install a promoted Tool-Maker proposal into the live tool root.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
            },
            "required": ["proposal_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_installed_tools",
        "description": "List live-installed Tool-Maker tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_uninstall_tool",
        "description": "Uninstall a live Tool-Maker tool by install id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "install_id": {"type": "string"},
            },
            "required": ["install_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_submit_speculation",
        "description": "Create a speculative branching plan with named hypotheses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_root": {"type": "string"},
                "task": {"type": "string"},
                "base_ref": {"type": "string"},
                "hypothesis_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["repo_root", "task", "hypothesis_names"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_speculation",
        "description": "Read a speculative branching plan and its evaluation manifest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
            },
            "required": ["plan_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_speculations",
        "description": "List speculative branching plans.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_prepare_speculation",
        "description": "Prepare git worktrees for an existing speculative branching plan.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
            },
            "required": ["plan_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_execute_speculation",
        "description": "Execute a speculative branching plan.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["ask", "auto", "dual", "critical"],
                    "description": "Execution mode.",
                    "default": "ask",
                },
            },
            "required": ["plan_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_runs",
        "description": "List recent Orchestra runs with optional mode/status filters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "default": 20},
                "mode": {"type": "string"},
                "status": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_run",
        "description": "Get a run manifest and its event history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_logs",
        "description": "Read stored agent logs for a run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "agent": {"type": "string"},
                "normalized": {"type": "boolean", "default": False},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_run_plan",
        "description": "Return the execution plan (providers list and whether post-processing is needed) for a task and mode, without creating any run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task prompt."},
                "mode": {
                    "type": "string",
                    "enum": ["ask", "auto", "dual", "critical"],
                    "description": "Execution mode.",
                },
                "alias": {
                    "type": "string",
                    "description": "Required only when mode is ask.",
                },
            },
            "required": ["task", "mode"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_create_fleet_run",
        "description": "Create an Orchestra run record and return its run_id without executing any agents. Use before spawning fleet background agents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["ask", "auto", "dual", "critical"],
                },
                "alias": {
                    "type": "string",
                    "description": "Required only when mode is ask.",
                },
            },
            "required": ["task", "mode"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_run_fleet_agent",
        "description": "Run a single provider agent within an existing fleet run. Blocking — returns when the provider subprocess exits. Designed to be called from a background agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "alias": {"type": "string", "description": "Provider alias, e.g. cdx-deep, gmn-pro, cld-fast."},
            },
            "required": ["run_id", "alias"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_finalize_fleet_run",
        "description": "Run post-processing (round-1 review, verification loop, synthesis) for a fleet run whose provider agents have all completed. Blocking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_execute_job",
        "description": "Synchronously claim and execute a queued Orchestra job.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_list_checkpoints",
        "description": "List artifact checkpoints for a specific run.",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_get_version",
        "description": "Return the Orchestra package version.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orchestra_backfill",
        "description": "Backfill the SQLite store from filesystem run artifacts.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orchestra_config_get",
        "description": "Read a config value by dotted path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include_env": {"type": "boolean", "default": True},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_config_set",
        "description": "Set a config value by dotted path, optionally persisting it to config.toml.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value": {},
                "persist": {"type": "boolean", "default": True},
            },
            "required": ["path", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_config_validate",
        "description": "Validate the active Orchestra configuration.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orchestra_config_init",
        "description": "Initialize .orchestra/config.toml if missing, or overwrite when forced.",
        "inputSchema": {
            "type": "object",
            "properties": {"force": {"type": "boolean", "default": False}},
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_memory_add",
        "description": "Add content to Orchestra semantic memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "metadata": {"type": "object"},
                "summary": {"type": "string"},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_memory_search",
        "description": "Search Orchestra semantic memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_memory_index_project",
        "description": "Index project files into Orchestra semantic memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "glob": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_inspect_route",
        "description": "Inspect Orchestra routing and tiering decisions for a task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "preferred_alias": {"type": "string"},
                "intent_hint": {"type": "string"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_trace_run",
        "description": "Return span trace data for a run.",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_timeline_run",
        "description": "Return enriched timeline events for a run.",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_diff_runs",
        "description": "Compare two Orchestra runs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id_a": {"type": "string"},
                "run_id_b": {"type": "string"},
            },
            "required": ["run_id_a", "run_id_b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_replay_run",
        "description": "Re-synthesize a previous dual+ run using cached outputs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "synthesizer": {"type": "string", "default": "cld-fast"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_batch_run",
        "description": "Execute multiple Orchestra tasks from a structured task list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string"},
                            "task": {"type": "string"},
                            "alias": {"type": "string"},
                            "pause_after_round1": {"type": "boolean"},
                            "intent_hint": {"type": "string"},
                        },
                        "required": ["mode", "task"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_eval_batch",
        "description": "Aggregate stored evaluation results for multiple task hashes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_hashes": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["task_hashes"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_eval_stats",
        "description": "Compute overall evaluation statistics for multiple task hashes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_hashes": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["task_hashes"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_undo_run",
        "description": "Undo filesystem changes for a run using Orchestra backtracking.",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_start_daemon",
        "description": "Start the Orchestra WebSocket daemon in the background.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "default": "127.0.0.1"},
                "port": {"type": "integer", "default": 8765},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "orchestra_daemon_status",
        "description": "Return Orchestra daemon status.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "orchestra_stop_daemon",
        "description": "Stop the Orchestra WebSocket daemon if it is running.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

RESOURCE_TEMPLATES = [
    {
        "uriTemplate": "orchestra://runs/{run_id}/checkpoints",
        "name": "Run Checkpoints",
        "description": "Checkpoint list for a specific Orchestra run.",
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "orchestra://runs/{run_id}/manifest",
        "name": "Run Manifest",
        "description": "Manifest JSON for a specific Orchestra run.",
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "orchestra://runs/{run_id}/events",
        "name": "Run Events",
        "description": "JSON event stream for a specific Orchestra run.",
        "mimeType": "application/json",
    },
    {
        "uriTemplate": "orchestra://runs/{run_id}/agents/{alias}/stdout",
        "name": "Agent Stdout",
        "description": "Raw stdout artifact for a run agent.",
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "orchestra://runs/{run_id}/agents/{alias}/normalized",
        "name": "Agent Normalized Output",
        "description": "Normalized ORCH_META/ORCH_BODY artifact for a run agent.",
        "mimeType": "text/plain",
    },
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _send_message(payload: dict) -> None:
    global _TRANSPORT_MODE
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if _TRANSPORT_MODE == "jsonl":
        sys.stdout.buffer.write(body + b"\n")
        sys.stdout.buffer.flush()
        return

    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read_message() -> dict | None:
    global _TRANSPORT_MODE

    while True:
        first_line = sys.stdin.buffer.readline()
        if not first_line:
            return None
        if first_line in {b"\r\n", b"\n"}:
            continue

        stripped = first_line.strip()
        if stripped.startswith(b"{") or stripped.startswith(b"["):
            _TRANSPORT_MODE = "jsonl"
            return json.loads(stripped.decode("utf-8"))

        headers: dict[str, str] = {}
        decoded = first_line.decode("ascii", errors="ignore").strip()
        if ":" in decoded:
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if line in {b"\r\n", b"\n"}:
                break
            decoded = line.decode("ascii", errors="ignore").strip()
            if ":" not in decoded:
                continue
            key, value = decoded.split(":", 1)
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            continue

        body = sys.stdin.buffer.read(content_length)
        if not body:
            return None

        _TRANSPORT_MODE = "framed"
        return json.loads(body.decode("utf-8"))


def _send_result(request_id: Any, result: dict) -> None:
    _send_message({"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result})


def _send_error(request_id: Any, code: int, message: str) -> None:
    _send_message(
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
    )


def _tool_success(payload: dict) -> dict:
    scrubbed = PiiScrubber.scrub_dict(payload)
    return {
        "content": [{"type": "text", "text": _json_dumps(scrubbed)}],
        "structuredContent": scrubbed,
        "isError": False,
    }


def _extract_answer_fields(text: str) -> tuple[str, str]:
    body = (text or "").strip()
    if not body:
        return "", ""

    answer = body
    key_signals = ""

    answer_match = re.search(r"##\s*Answer\s*(.*?)(?:##\s*Key Signals|##\s*Dissent|$)", body, re.S | re.I)
    if answer_match:
        answer = answer_match.group(1).strip()

    key_signals_match = re.search(r"##\s*Key Signals\s*(.*?)(?:##\s*Dissent|$)", body, re.S | re.I)
    if key_signals_match:
        key_signals = key_signals_match.group(1).strip()

    return answer, key_signals


def _structured_run_result(payload: dict) -> dict:
    run = payload.get("run", {}) if isinstance(payload, dict) else {}
    run_id = run.get("run_id", "")
    status = run.get("status", "unknown")
    mode = run.get("mode", "unknown")
    agents = run.get("agents", []) if isinstance(run.get("agents"), list) else []
    summary = run.get("summary", "") or ""

    primary_agent = next(
        (
            agent for agent in agents
            if agent.get("status") == "completed" and agent.get("alias") != "verifier"
        ),
        None,
    )
    if primary_agent is None:
        primary_agent = next((agent for agent in agents if agent.get("status") == "completed"), None)
    primary_source = primary_agent.get("alias", "") if primary_agent else ""
    confidence = float(run.get("avg_confidence", 0.0) or 0.0)

    raw_text = summary
    if not raw_text and primary_agent:
        raw_text = primary_agent.get("stdout_log", "") or ""

    meta, parsed_body = parse_envelope(raw_text) if raw_text else (None, "")
    body = parsed_body or raw_text
    answer, key_signals = _extract_answer_fields(body)
    dissent = parse_dissent(body) or ""

    if meta is not None and meta.confidence:
        confidence = float(meta.confidence)
    if not answer:
        answer = body.strip()
    if not answer and status == "waiting_approval":
        answer = "Run is waiting for approval."

    result = {
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "answer": answer,
        "confidence": round(confidence, 3),
        "primary_source": primary_source,
        "dissent": dissent,
        "cost_usd": round(float(run.get("total_cost_usd", 0.0) or 0.0), 6),
    }
    if key_signals:
        result["key_signals"] = key_signals
    if meta is not None:
        result["meta"] = meta.to_dict()
    return result


def _tool_success_run(payload: dict) -> dict:
    structured = dict(payload)
    structured["result"] = _structured_run_result(payload)
    scrubbed = PiiScrubber.scrub_dict(structured)
    return {
        "content": [{"type": "text", "text": _json_dumps(scrubbed)}],
        "structuredContent": scrubbed,
        "isError": False,
    }


def _tool_error(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _resource_contents(uri: str, mime_type: str, payload: Any) -> dict:
    if isinstance(payload, str):
        text = PiiScrubber.scrub_text(payload)
    else:
        text = _json_dumps(PiiScrubber.scrub_dict(payload))
    return {
        "contents": [
            {
                "uri": uri,
                "mimeType": mime_type,
                "text": text,
            }
        ]
    }


def _recent_resource_entries() -> list[dict]:
    resources = [
        {
            "uri": "orchestra://version",
            "name": "Orchestra Version",
            "description": "Installed Orchestra package version.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://aliases",
            "name": "Orchestra Aliases",
            "description": "Available Orchestra aliases and provider availability.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://stats",
            "name": "Orchestra Stats",
            "description": "Aggregate Orchestra run statistics.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://reputation",
            "name": "Orchestra Reputation",
            "description": "Alias-level reputation scores and supporting metrics.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://toolmaker/proposals",
            "name": "Tool-Maker Proposals",
            "description": "List of Tool-Maker proposals and approval states.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://toolmaker/installed",
            "name": "Installed Tool-Maker Tools",
            "description": "List of live-installed Tool-Maker tools.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://speculations",
            "name": "Speculative Plans",
            "description": "List of speculative branching plans.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://runs/recent",
            "name": "Recent Orchestra Runs",
            "description": "Recent run summary list.",
            "mimeType": "application/json",
        },
        {
            "uri": "orchestra://jobs/recent",
            "name": "Recent Orchestra Jobs",
            "description": "Recent async Orchestra jobs.",
            "mimeType": "application/json",
        },
    ]

    recent_runs = list_runs(limit=10)["runs"]
    for run in recent_runs:
        run_id = run["run_id"]
        resources.append(
            {
                "uri": f"orchestra://runs/{run_id}/manifest",
                "name": f"Run {run_id} Manifest",
                "description": "Manifest for a recent Orchestra run.",
                "mimeType": "application/json",
            }
        )
        resources.append(
            {
                "uri": f"orchestra://runs/{run_id}/events",
                "name": f"Run {run_id} Events",
                "description": "Events for a recent Orchestra run.",
                "mimeType": "application/json",
            }
        )
        resources.append(
            {
                "uri": f"orchestra://runs/{run_id}/checkpoints",
                "name": f"Run {run_id} Checkpoints",
                "description": "Checkpoint snapshots for a recent Orchestra run.",
                "mimeType": "application/json",
            }
        )

    return resources


# ── MCP Tool Registry (auto-registered from service.py) ──────────────────────
from orchestra.engine.mcp_registry import REGISTRY, _auto_wrap

REGISTRY.register("orchestra_submit_run", _auto_wrap(submit_run))
REGISTRY.register("orchestra_get_job", _auto_wrap(get_job))
REGISTRY.register("orchestra_list_jobs", _auto_wrap(list_jobs))
REGISTRY.register("orchestra_wait_for_run", _auto_wrap(wait_for_run))
REGISTRY.register("orchestra_run", _auto_wrap(run_task))
REGISTRY.register("orchestra_continue", _auto_wrap(continue_run))
REGISTRY.register("orchestra_watch_run", _auto_wrap(watch_run_snapshot))
REGISTRY.register("orchestra_cancel", _auto_wrap(cancel_run))
REGISTRY.register("orchestra_list_aliases", _auto_wrap(list_aliases))
REGISTRY.register("orchestra_get_stats", _auto_wrap(get_stats))
REGISTRY.register("orchestra_get_reputation", _auto_wrap(get_reputation))
REGISTRY.register("orchestra_submit_tool_proposal", _auto_wrap(submit_tool_proposal))
REGISTRY.register("orchestra_list_tool_proposals", _auto_wrap(list_toolmaker_proposals))
REGISTRY.register("orchestra_get_tool_proposal", _auto_wrap(get_toolmaker_proposal))
REGISTRY.register("orchestra_review_tool_proposal", _auto_wrap(review_toolmaker_proposal))
REGISTRY.register("orchestra_record_tool_test", _auto_wrap(record_toolmaker_test))
REGISTRY.register("orchestra_promote_tool_proposal", _auto_wrap(promote_toolmaker_proposal))
REGISTRY.register("orchestra_install_tool_proposal", _auto_wrap(install_toolmaker_proposal))
REGISTRY.register("orchestra_list_installed_tools", _auto_wrap(list_installed_tools))
REGISTRY.register("orchestra_uninstall_tool", _auto_wrap(uninstall_installed_tool))
REGISTRY.register("orchestra_submit_speculation", _auto_wrap(submit_speculative_plan))
REGISTRY.register("orchestra_get_speculation", _auto_wrap(get_speculation))
REGISTRY.register("orchestra_list_speculations", _auto_wrap(list_speculations))
REGISTRY.register("orchestra_prepare_speculation", _auto_wrap(prepare_speculation))
REGISTRY.register("orchestra_execute_speculation", _auto_wrap(execute_speculation))
REGISTRY.register("orchestra_list_runs", _auto_wrap(list_runs))
REGISTRY.register("orchestra_get_run", _auto_wrap(get_run))
REGISTRY.register("orchestra_get_logs", _auto_wrap(get_logs))
REGISTRY.register("orchestra_get_run_plan", _auto_wrap(get_run_plan))
REGISTRY.register("orchestra_create_fleet_run", _auto_wrap(create_fleet_run))
REGISTRY.register("orchestra_run_fleet_agent", _auto_wrap(run_fleet_agent))
REGISTRY.register("orchestra_finalize_fleet_run", _auto_wrap(finalize_fleet_run))
REGISTRY.register("orchestra_execute_job", _auto_wrap(execute_job))
REGISTRY.register("orchestra_list_checkpoints", _auto_wrap(list_checkpoints))
REGISTRY.register("orchestra_get_version", _auto_wrap(get_version))
REGISTRY.register("orchestra_backfill", _auto_wrap(run_backfill))
REGISTRY.register("orchestra_config_get", _auto_wrap(config_get_value))
REGISTRY.register("orchestra_config_set", _auto_wrap(config_set_value))
REGISTRY.register("orchestra_config_validate", _auto_wrap(config_validate_value))
REGISTRY.register("orchestra_config_init", _auto_wrap(config_init_value))
REGISTRY.register("orchestra_memory_add", _auto_wrap(memory_add_entry))
REGISTRY.register("orchestra_memory_search", _auto_wrap(memory_search_entries))
REGISTRY.register("orchestra_memory_index_project", _auto_wrap(memory_index_project_entries))
REGISTRY.register("orchestra_inspect_route", _auto_wrap(inspect_route))
REGISTRY.register("orchestra_trace_run", _auto_wrap(get_trace))
REGISTRY.register("orchestra_timeline_run", _auto_wrap(get_timeline))
REGISTRY.register("orchestra_diff_runs", _auto_wrap(compare_runs))
REGISTRY.register("orchestra_replay_run", _auto_wrap(replay_run_output))
REGISTRY.register("orchestra_batch_run", _auto_wrap(batch_run_tasks))
REGISTRY.register("orchestra_eval_batch", _auto_wrap(get_eval_batch))
REGISTRY.register("orchestra_eval_stats", _auto_wrap(get_eval_stats))
REGISTRY.register("orchestra_undo_run", _auto_wrap(undo_run_changes))
REGISTRY.register("orchestra_start_daemon", _auto_wrap(start_daemon))
REGISTRY.register("orchestra_daemon_status", _auto_wrap(daemon_status))
REGISTRY.register("orchestra_stop_daemon", _auto_wrap(stop_daemon))


def _handle_tool_call(name: str, arguments: dict) -> dict:
    decision = PolicyEngine.from_default().evaluate_tool(name)
    if decision.action == "deny":
        return {"error": f"Tool '{name}' is denied by policy: {decision.reason}"}

    # Registry / Fast Lane path
    result = REGISTRY.invoke(name, arguments)
    if result is not None:
        return result

    # Legacy fallback (should never hit if all tools are registered above)
    if name == "orchestra_submit_run":
        return _tool_success_run(
            submit_run(
                mode=arguments["mode"],
                task=arguments["task"],
                alias=arguments.get("alias"),
                pause_after_round1=arguments.get("pause_after_round1", False),
            )
        )
    if name == "orchestra_get_job":
        return _tool_success_run(get_job(arguments["job_id"]))
    if name == "orchestra_list_jobs":
        return _tool_success(
            list_jobs(
                limit=arguments.get("limit", 20),
                status=arguments.get("status"),
            )
        )
    if name == "orchestra_wait_for_run":
        return _tool_success_run(
            wait_for_run(
                run_id=arguments.get("run_id"),
                job_id=arguments.get("job_id"),
                timeout_seconds=arguments.get("timeout_seconds", 30),
                poll_interval_seconds=arguments.get("poll_interval_seconds", 1.0),
            )
        )
    if name == "orchestra_run":
        return _tool_success_run(
            run_task(
                mode=arguments["mode"],
                task=arguments["task"],
                alias=arguments.get("alias"),
                pause_after_round1=arguments.get("pause_after_round1", False),
            )
        )
    if name == "orchestra_continue":
        return _tool_success_run(continue_run(arguments["run_id"]))
    if name == "orchestra_watch_run":
        return _tool_success(
            watch_run_snapshot(
                arguments["run_id"],
                normalized=arguments.get("normalized", False),
            )
        )
    if name == "orchestra_cancel":
        return _tool_success(cancel_run(arguments["run_id"]))
    if name == "orchestra_list_aliases":
        return _tool_success(list_aliases())
    if name == "orchestra_get_stats":
        return _tool_success(get_stats())
    if name == "orchestra_get_reputation":
        return _tool_success(get_reputation(arguments.get("alias")))
    if name == "orchestra_submit_tool_proposal":
        return _tool_success(
            submit_tool_proposal(
                name=arguments["name"],
                description=arguments["description"],
                files=arguments.get("files"),
                run_id=arguments.get("run_id"),
                source_alias=arguments.get("source_alias"),
                test_command=arguments.get("test_command", ""),
            )
        )
    if name == "orchestra_list_tool_proposals":
        return _tool_success(list_toolmaker_proposals(arguments.get("status")))
    if name == "orchestra_get_tool_proposal":
        return _tool_success(get_toolmaker_proposal(arguments["proposal_id"]))
    if name == "orchestra_review_tool_proposal":
        return _tool_success(
            review_toolmaker_proposal(
                arguments["proposal_id"],
                arguments["approve"],
                arguments.get("note", ""),
            )
        )
    if name == "orchestra_record_tool_test":
        return _tool_success(
            record_toolmaker_test(
                arguments["proposal_id"],
                status=arguments["status"],
                summary=arguments["summary"],
                command=arguments.get("command", ""),
            )
        )
    if name == "orchestra_promote_tool_proposal":
        return _tool_success(promote_toolmaker_proposal(arguments["proposal_id"]))
    if name == "orchestra_install_tool_proposal":
        return _tool_success(install_toolmaker_proposal(arguments["proposal_id"]))
    if name == "orchestra_list_installed_tools":
        return _tool_success(list_installed_tools(arguments.get("status")))
    if name == "orchestra_uninstall_tool":
        return _tool_success(uninstall_installed_tool(arguments["install_id"]))
    if name == "orchestra_submit_speculation":
        return _tool_success(
            submit_speculative_plan(
                repo_root=arguments["repo_root"],
                task=arguments["task"],
                base_ref=arguments.get("base_ref", "HEAD"),
                hypothesis_names=arguments["hypothesis_names"],
            )
        )
    if name == "orchestra_get_speculation":
        return _tool_success(get_speculation(arguments["plan_id"]))
    if name == "orchestra_list_speculations":
        return _tool_success(list_speculations(arguments.get("status")))
    if name == "orchestra_prepare_speculation":
        return _tool_success(prepare_speculation(arguments["plan_id"]))
    if name == "orchestra_execute_speculation":
        return _tool_success(
            execute_speculation(
                arguments["plan_id"],
                mode=arguments.get("mode", "ask"),
            )
        )
    if name == "orchestra_list_runs":
        return _tool_success(
            list_runs(
                limit=arguments.get("limit", 20),
                mode=arguments.get("mode"),
                status=arguments.get("status"),
            )
        )
    if name == "orchestra_get_run":
        return _tool_success_run(get_run(arguments["run_id"]))
    if name == "orchestra_get_logs":
        return _tool_success(
            get_logs(
                arguments["run_id"],
                agent=arguments.get("agent"),
                normalized=arguments.get("normalized", False),
            )
        )
    if name == "orchestra_get_run_plan":
        return _tool_success(
            get_run_plan(
                task=arguments["task"],
                mode=arguments["mode"],
                alias=arguments.get("alias"),
            )
        )
    if name == "orchestra_create_fleet_run":
        return _tool_success(
            create_fleet_run(
                task=arguments["task"],
                mode=arguments["mode"],
                alias=arguments.get("alias"),
            )
        )
    if name == "orchestra_run_fleet_agent":
        return _tool_success(
            run_fleet_agent(
                run_id=arguments["run_id"],
                alias=arguments["alias"],
            )
        )
    if name == "orchestra_finalize_fleet_run":
        return _tool_success(finalize_fleet_run(run_id=arguments["run_id"]))
    if name == "orchestra_execute_job":
        return _tool_success_run(execute_job(arguments["job_id"]))
    if name == "orchestra_list_checkpoints":
        return _tool_success(list_checkpoints(arguments["run_id"]))
    if name == "orchestra_get_version":
        return _tool_success(get_version())
    if name == "orchestra_backfill":
        return _tool_success(run_backfill())
    if name == "orchestra_config_get":
        return _tool_success(
            config_get_value(
                arguments["path"],
                include_env=arguments.get("include_env", True),
            )
        )
    if name == "orchestra_config_set":
        return _tool_success(
            config_set_value(
                arguments["path"],
                arguments["value"],
                persist=arguments.get("persist", True),
            )
        )
    if name == "orchestra_config_validate":
        return _tool_success(config_validate_value())
    if name == "orchestra_config_init":
        return _tool_success(config_init_value(force=arguments.get("force", False)))
    if name == "orchestra_memory_add":
        return _tool_success(
            memory_add_entry(
                arguments["content"],
                metadata=arguments.get("metadata"),
                summary=arguments.get("summary"),
            )
        )
    if name == "orchestra_memory_search":
        return _tool_success(
            memory_search_entries(
                arguments["query"],
                limit=arguments.get("limit", 5),
            )
        )
    if name == "orchestra_memory_index_project":
        return _tool_success(
            memory_index_project_entries(
                path=arguments.get("path", "."),
                glob_pattern=arguments.get("glob", "**/*.md"),
            )
        )
    if name == "orchestra_inspect_route":
        return _tool_success(
            inspect_route(
                arguments["task"],
                preferred_alias=arguments.get("preferred_alias"),
                intent_hint=arguments.get("intent_hint", ""),
            )
        )
    if name == "orchestra_trace_run":
        return _tool_success(get_trace(arguments["run_id"]))
    if name == "orchestra_timeline_run":
        return _tool_success(get_timeline(arguments["run_id"]))
    if name == "orchestra_diff_runs":
        return _tool_success(compare_runs(arguments["run_id_a"], arguments["run_id_b"]))
    if name == "orchestra_replay_run":
        return _tool_success(
            replay_run_output(
                arguments["run_id"],
                synthesizer=arguments.get("synthesizer", "cld-fast"),
            )
        )
    if name == "orchestra_batch_run":
        return _tool_success(batch_run_tasks(arguments["tasks"]))
    if name == "orchestra_eval_batch":
        return _tool_success(get_eval_batch(arguments["task_hashes"]))
    if name == "orchestra_eval_stats":
        return _tool_success(get_eval_stats(arguments["task_hashes"]))
    if name == "orchestra_undo_run":
        return _tool_success(undo_run_changes(arguments["run_id"]))
    if name == "orchestra_start_daemon":
        return _tool_success(
            start_daemon(
                host=arguments.get("host", "127.0.0.1"),
                port=arguments.get("port", 8765),
            )
        )
    if name == "orchestra_daemon_status":
        return _tool_success(daemon_status())
    if name == "orchestra_stop_daemon":
        return _tool_success(stop_daemon())
    return _tool_error(f"Unknown tool: {name}")


def _handle_resource_read(uri: str) -> dict:
    if uri == "orchestra://aliases":
        return _resource_contents(uri, "application/json", list_aliases())
    if uri == "orchestra://version":
        return _resource_contents(uri, "application/json", get_version())
    if uri == "orchestra://stats":
        return _resource_contents(uri, "application/json", get_stats())
    if uri == "orchestra://reputation":
        return _resource_contents(uri, "application/json", get_reputation())
    if uri == "orchestra://toolmaker/proposals":
        return _resource_contents(uri, "application/json", list_toolmaker_proposals())
    if uri == "orchestra://toolmaker/installed":
        return _resource_contents(uri, "application/json", list_installed_tools())
    if uri == "orchestra://speculations":
        return _resource_contents(uri, "application/json", list_speculations())
    if uri == "orchestra://runs/recent":
        return _resource_contents(uri, "application/json", list_runs(limit=20))
    if uri == "orchestra://jobs/recent":
        return _resource_contents(uri, "application/json", list_jobs(limit=20))

    proposal_match = re.fullmatch(r"orchestra://toolmaker/proposals/([^/]+)", uri)
    if proposal_match:
        proposal_id = unquote(proposal_match.group(1))
        return _resource_contents(uri, "application/json", get_toolmaker_proposal(proposal_id))

    speculation_match = re.fullmatch(r"orchestra://speculations/([^/]+)", uri)
    if speculation_match:
        plan_id = unquote(speculation_match.group(1))
        return _resource_contents(uri, "application/json", get_speculation(plan_id))

    manifest_match = re.fullmatch(r"orchestra://runs/([^/]+)/manifest", uri)
    if manifest_match:
        run_id = unquote(manifest_match.group(1))
        return _resource_contents(uri, "application/json", get_run(run_id)["run"])

    events_match = re.fullmatch(r"orchestra://runs/([^/]+)/events", uri)
    if events_match:
        run_id = unquote(events_match.group(1))
        return _resource_contents(uri, "application/json", get_run(run_id)["events"])

    checkpoints_match = re.fullmatch(r"orchestra://runs/([^/]+)/checkpoints", uri)
    if checkpoints_match:
        run_id = unquote(checkpoints_match.group(1))
        return _resource_contents(uri, "application/json", list_checkpoints(run_id))

    stdout_match = re.fullmatch(r"orchestra://runs/([^/]+)/agents/([^/]+)/stdout", uri)
    if stdout_match:
        run_id = unquote(stdout_match.group(1))
        alias = unquote(stdout_match.group(2))
        logs = get_logs(run_id, agent=alias, normalized=False)
        if alias not in logs["logs"]:
            raise ValueError(f"Stdout log not found for {run_id}/{alias}")
        return _resource_contents(uri, "text/plain", logs["logs"][alias])

    normalized_match = re.fullmatch(r"orchestra://runs/([^/]+)/agents/([^/]+)/normalized", uri)
    if normalized_match:
        run_id = unquote(normalized_match.group(1))
        alias = unquote(normalized_match.group(2))
        logs = get_logs(run_id, agent=alias, normalized=True)
        if alias not in logs["logs"]:
            raise ValueError(f"Normalized log not found for {run_id}/{alias}")
        return _resource_contents(uri, "text/plain", logs["logs"][alias])

    raise ValueError(f"Unknown resource: {uri}")


def _handle_request(message: dict) -> dict | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {}) or {}

    if method == "initialize":
        client_version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
        return {
            "protocolVersion": client_version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": "orchestra-mcp",
                "version": __version__,
            },
        }

    if method == "ping":
        return {}

    if method == "tools/list":
        return {"tools": TOOL_DEFINITIONS}

    if method == "tools/call":
        result = _handle_tool_call(params.get("name", ""), params.get("arguments", {}) or {})
        run_id = (params.get("arguments", {}) or {}).get("run_id")
        if run_id and config.replay_enabled():
            from orchestra.engine.replay import get_recorder
            get_recorder(run_id).record_tool_call(
                params.get("name", ""), params.get("arguments", {}) or {}, result
            )
        return result

    if method == "resources/list":
        try:
            return {"resources": _recent_resource_entries()}
        except Exception:
            return {"resources": []}

    if method == "resources/templates/list":
        return {"resourceTemplates": RESOURCE_TEMPLATES}

    if method == "resources/read":
        return _handle_resource_read(params["uri"])

    if method == "prompts/list":
        return {"prompts": [
            {
                "name": "fleet",
                "description": "Orchestra Fleet View workflow — spawns parallel background agents for each provider, review, and synthesis step so all operations appear in the Claude Code fleet panel.",
            }
        ]}

    if method == "prompts/get":
        prompt_name = params.get("name", "")
        if prompt_name == "fleet":
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {"type": "text", "text": _FLEET_SKILL_CONTENT},
                    }
                ]
            }
        raise ValueError(f"Unknown prompt: {prompt_name}")

    if method == "notifications/initialized":
        return None

    if method and method.startswith("notifications/"):
        return None

    if request_id is None:
        return None
    raise ValueError(f"Method not found: {method}")


def run_stdio_server() -> None:
    while True:
        try:
            message = _read_message()
        except json.JSONDecodeError as exc:
            _send_error(None, -32700, f"Parse error: {exc}")
            continue

        if message is None:
            return

        request_id = message.get("id")

        try:
            result = _handle_request(message)
            if request_id is not None and result is not None:
                _send_result(request_id, result)
        except ValueError as exc:
            if request_id is not None:
                _send_error(request_id, -32000, str(exc))
        except Exception as exc:  # pragma: no cover - defensive server boundary
            if request_id is not None:
                _send_error(request_id, -32603, f"Internal error: {exc}")


def main() -> None:
    run_stdio_server()


if __name__ == "__main__":
    main()
