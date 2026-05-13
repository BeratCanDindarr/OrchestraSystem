"""Minimal stdio MCP server for Orchestra."""
from __future__ import annotations

import json
import re
import sys
from typing import Any
from urllib.parse import unquote

from orchestra import __version__
from orchestra.engine.synthesizer import parse_dissent
from orchestra.protocol import parse_envelope
from orchestra.service import (
    cancel_run,
    continue_run,
    get_logs,
    get_job,
    get_reputation,
    get_toolmaker_proposal,
    get_run,
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
    run_task,
    submit_speculative_plan,
    submit_run,
    submit_tool_proposal,
    uninstall_installed_tool,
    wait_for_run,
)


JSONRPC_VERSION = "2.0"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
_TRANSPORT_MODE: str | None = None


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
        "description": "Poll a run or async job until it reaches a terminal or waiting-approval state.",
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
]

RESOURCE_TEMPLATES = [
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
    return {
        "content": [{"type": "text", "text": _json_dumps(payload)}],
        "structuredContent": payload,
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
    return {
        "content": [{"type": "text", "text": _json_dumps(structured)}],
        "structuredContent": structured,
        "isError": False,
    }


def _tool_error(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _resource_contents(uri: str, mime_type: str, payload: Any) -> dict:
    if isinstance(payload, str):
        text = payload
    else:
        text = _json_dumps(payload)
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

    return resources


def _handle_tool_call(name: str, arguments: dict) -> dict:
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
    if name == "orchestra_cancel":
        return _tool_success(cancel_run(arguments["run_id"]))
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
    return _tool_error(f"Unknown tool: {name}")


def _handle_resource_read(uri: str) -> dict:
    if uri == "orchestra://aliases":
        return _resource_contents(uri, "application/json", list_aliases())
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
        return _handle_tool_call(params.get("name", ""), params.get("arguments", {}) or {})

    if method == "resources/list":
        return {"resources": _recent_resource_entries()}

    if method == "resources/templates/list":
        return {"resourceTemplates": RESOURCE_TEMPLATES}

    if method == "resources/read":
        return _handle_resource_read(params["uri"])

    if method == "prompts/list":
        return {"prompts": []}

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
