"""Headless service API for Orchestra automation surfaces such as MCP."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.process_group import ProcessGroupManager
from orchestra.engine.runner import resume_run, run_ask, run_critical, run_dual
from orchestra.models import RunStatus
from orchestra.providers.fallback import available_aliases
from orchestra.router import classify, task_to_mode
from orchestra.storage.db import backfill, get_db
from orchestra.storage.jobs import (
    claim_job,
    get_job as get_job_record,
    insert_job,
    list_jobs as list_job_records,
    mark_job_completed,
    mark_job_failed,
    mark_job_run_started,
)
from orchestra.storage.reputation import list_alias_reputation
from orchestra.storage.speculation import (
    create_speculative_plan,
    get_speculative_plan,
    list_speculative_plans,
    prepare_speculative_worktrees,
)
from orchestra.storage.toolmaker import (
    create_tool_proposal,
    get_tool_proposal,
    install_tool_proposal,
    list_tool_installs,
    list_tool_proposals,
    promote_tool_proposal,
    record_tool_test_result,
    review_tool_proposal,
    uninstall_tool,
)


VALID_MODES = {"ask", "auto", "dual", "critical"}
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _serialize_rows(rows) -> list[dict]:
    return [dict(row) for row in rows]


def run_task(
    *,
    mode: str,
    task: str,
    alias: Optional[str] = None,
    pause_after_round1: bool = False,
) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    route_class = None
    routed_mode = mode

    if mode == "ask":
        if not alias:
            raise ValueError("alias is required when mode='ask'")
        run = run_ask(
            alias,
            task,
            show_live=False,
            emit_console=False,
            install_signal_handlers=False,
        )
    elif mode == "dual":
        run = run_dual(
            task,
            show_live=False,
            emit_console=False,
            install_signal_handlers=False,
        )
    elif mode == "critical":
        run = run_critical(
            task,
            require_approval=pause_after_round1,
            show_live=False,
            emit_console=False,
            install_signal_handlers=False,
            approval_behavior="pause",
        )
    else:
        route_class = classify(task)
        routed_mode = task_to_mode(task)

        if routed_mode.startswith("ask "):
            selected_alias = routed_mode.split(" ", 1)[1].strip()
            run = run_ask(
                selected_alias,
                task,
                show_live=False,
                emit_console=False,
                install_signal_handlers=False,
            )
        elif routed_mode == "dual":
            run = run_dual(
                task,
                show_live=False,
                emit_console=False,
                install_signal_handlers=False,
            )
        elif routed_mode == "critical":
            run = run_critical(
                task,
                require_approval=pause_after_round1,
                show_live=False,
                emit_console=False,
                install_signal_handlers=False,
                approval_behavior="pause" if pause_after_round1 else "continue",
            )
        else:
            raise ValueError(f"Unknown auto route: {routed_mode}")

    return {
        "route_class": route_class,
        "routed_mode": routed_mode,
        "run": run.to_manifest(),
    }


def continue_run(run_id: str) -> dict:
    run = resume_run(
        run_id,
        show_live=False,
        emit_console=False,
        install_signal_handlers=False,
    )
    return {"run": run.to_manifest()}


def submit_run(
    *,
    mode: str,
    task: str,
    alias: Optional[str] = None,
    pause_after_round1: bool = False,
) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "ask" and not alias:
        raise ValueError("alias is required when mode='ask'")

    connection = get_db()
    with connection:
        job = insert_job(
            connection,
            mode=mode,
            task=task,
            alias=alias,
            pause_after_round1=pause_after_round1,
        )
    connection.close()

    subprocess.Popen(
        [sys.executable, "-m", "orchestra", "job-worker", job["job_id"]],
        cwd=str(PACKAGE_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"job": job}


def execute_job(job_id: str) -> dict:
    connection = get_db()
    worker_pid = os.getpid()
    with connection:
        claimed = claim_job(connection, job_id, worker_pid=worker_pid)
        if not claimed:
            job = get_job_record(connection, job_id)
            connection.close()
            return {"job": job, "claimed": False}
        job = get_job_record(connection, job_id)
    connection.close()

    try:
        result = run_task(
            mode=job["mode"],
            task=job["task"],
            alias=job.get("alias"),
            pause_after_round1=bool(job.get("pause_after_round1")),
        )
        run_manifest = result["run"]
        connection = get_db()
        with connection:
            mark_job_run_started(connection, job_id, run_manifest["run_id"], worker_pid=worker_pid)
            if run_manifest.get("status") in {RunStatus.COMPLETED.value, RunStatus.WAITING_APPROVAL.value}:
                mark_job_completed(connection, job_id, run_manifest["run_id"])
            else:
                error_message = ((run_manifest.get("failure") or {}).get("message") or run_manifest.get("status") or "run_failed")
                mark_job_failed(connection, job_id, "run_failed", error_message)
        connection.close()
        return {"job": get_job(job_id)["job"], "run": run_manifest}
    except Exception as exc:
        connection = get_db()
        with connection:
            mark_job_failed(connection, job_id, "worker_exception", str(exc))
        connection.close()
        raise


def get_job(job_id: str) -> dict:
    connection = get_db()
    job = get_job_record(connection, job_id)
    connection.close()
    payload = {"job": job}
    run_id = job.get("run_id")
    if run_id:
        try:
            run_payload = get_run(run_id)
            payload["run"] = run_payload["run"]
            payload["events"] = run_payload["events"]
        except ValueError:
            pass
    return payload


def list_jobs(limit: int = 20, status: Optional[str] = None) -> dict:
    connection = get_db()
    jobs = list_job_records(connection, status=status, limit=limit)
    connection.close()
    return {"jobs": jobs}


def list_checkpoints(run_id: str) -> dict:
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "checkpoints": artifacts.list_checkpoints(run_id),
    }


def cancel_run(run_id: str) -> dict:
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")

    status = manifest.get("status", "unknown")
    if status in {
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.CANCELLED.value,
    }:
        return {"run": manifest, "already_finished": True}

    manager = ProcessGroupManager()
    for agent in manifest.get("agents", []):
        pid = agent.get("pid")
        if pid:
            manager.register(int(pid))
    manager.kill_all()

    now = datetime.now(timezone.utc).isoformat()
    manifest["status"] = RunStatus.CANCELLED.value
    manifest["updated_at"] = now
    manifest["interrupt_state"] = "cancelled"

    for agent in manifest.get("agents", []):
        if agent.get("status") in {"queued", "started"} or agent.get("pid"):
            agent["status"] = RunStatus.CANCELLED.value
            agent["end_time"] = agent.get("end_time") or now
            agent["error"] = "cancelled by user"
        agent["pid"] = None

    artifacts.write_manifest_data(run_id, manifest)
    artifacts.append_event(run_id, {"event": "run_cancelled", "reason": "service_cancel"})
    return {"run": manifest, "already_finished": False}


def list_runs(limit: int = 20, mode: Optional[str] = None, status: Optional[str] = None) -> dict:
    backfill()
    connection = get_db()

    query = [
        "SELECT run_id, mode, status, task, created_at, updated_at,",
        "       latest_review_stage, latest_review_status, latest_review_winner, latest_review_reason,",
        "       approval_state, interrupt_state",
        "FROM runs",
        "WHERE 1=1",
    ]
    params: list[object] = []

    if mode:
        query.append("AND mode = ?")
        params.append(mode)
    if status:
        query.append("AND status = ?")
        params.append(status)

    query.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)

    rows = connection.execute(" ".join(query), params).fetchall()
    connection.close()
    return {"runs": _serialize_rows(rows)}


def get_run(run_id: str) -> dict:
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")

    return {
        "run": manifest,
        "events": artifacts.read_events(run_id),
    }


def get_logs(run_id: str, agent: Optional[str] = None, normalized: bool = False) -> dict:
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")

    aliases = [agent] if agent else [item.get("alias", "") for item in manifest.get("agents", [])]
    logs: dict[str, str] = {}
    missing: list[str] = []

    for alias_name in aliases:
        if not alias_name:
            continue
        content = artifacts.read_agent_log(run_id, alias_name, normalized=normalized)
        if content is None:
            missing.append(alias_name)
            continue
        logs[alias_name] = content

    return {
        "run_id": run_id,
        "normalized": normalized,
        "logs": logs,
        "missing": missing,
    }


def list_aliases() -> dict:
    return {"aliases": available_aliases()}


def get_stats() -> dict:
    backfill()
    connection = get_db()
    total_runs = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    completed_runs = connection.execute(
        "SELECT COUNT(*) FROM runs WHERE status = 'completed'"
    ).fetchone()[0]
    failed_runs = connection.execute(
        "SELECT COUNT(*) FROM runs WHERE status = 'failed'"
    ).fetchone()[0]

    success_rows = connection.execute(
        """
        SELECT alias,
               SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
               SUM(CASE WHEN status IN ('completed', 'failed', 'cancelled') THEN 1 ELSE 0 END) AS total_count
        FROM agents
        GROUP BY alias
        HAVING total_count > 0
        ORDER BY alias
        """
    ).fetchall()
    reputation_rows = list_alias_reputation(connection)
    connection.close()

    return {
        "total_runs": total_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "provider_success": _serialize_rows(success_rows),
        "alias_reputation": reputation_rows,
    }


def get_reputation(alias: Optional[str] = None) -> dict:
    backfill()
    connection = get_db()
    rows = list_alias_reputation(connection)
    connection.close()
    if alias:
        rows = [row for row in rows if row["alias"] == alias]
    return {"reputation": rows}


def submit_tool_proposal(
    *,
    name: str,
    description: str,
    files: Optional[list[dict]] = None,
    run_id: Optional[str] = None,
    source_alias: Optional[str] = None,
    test_command: str = "",
) -> dict:
    connection = get_db()
    with connection:
        proposal = create_tool_proposal(
            connection,
            name=name,
            description=description,
            files=files,
            run_id=run_id,
            source_alias=source_alias,
            test_command=test_command,
        )
    connection.close()
    return {"proposal": proposal}


def list_toolmaker_proposals(status: Optional[str] = None) -> dict:
    connection = get_db()
    proposals = list_tool_proposals(connection, status=status)
    connection.close()
    return {"proposals": proposals}


def get_toolmaker_proposal(proposal_id: str) -> dict:
    connection = get_db()
    proposal = get_tool_proposal(connection, proposal_id)
    connection.close()
    return {"proposal": proposal}


def review_toolmaker_proposal(proposal_id: str, approve: bool, note: str = "") -> dict:
    connection = get_db()
    with connection:
        proposal = review_tool_proposal(
            connection,
            proposal_id=proposal_id,
            approve=approve,
            note=note,
        )
    connection.close()
    return {"proposal": proposal}


def record_toolmaker_test(
    proposal_id: str,
    *,
    status: str,
    summary: str,
    command: str = "",
) -> dict:
    connection = get_db()
    with connection:
        proposal = record_tool_test_result(
            connection,
            proposal_id=proposal_id,
            status=status,
            summary=summary,
            command=command,
        )
    connection.close()
    return {"proposal": proposal}


def promote_toolmaker_proposal(proposal_id: str) -> dict:
    connection = get_db()
    with connection:
        proposal = promote_tool_proposal(connection, proposal_id)
    connection.close()
    return {"proposal": proposal}


def install_toolmaker_proposal(proposal_id: str) -> dict:
    connection = get_db()
    with connection:
        install = install_tool_proposal(connection, proposal_id)
    connection.close()
    return {"install": install}


def list_installed_tools(status: Optional[str] = None) -> dict:
    connection = get_db()
    installs = list_tool_installs(connection, status=status)
    connection.close()
    return {"installs": installs}


def uninstall_installed_tool(install_id: str) -> dict:
    connection = get_db()
    with connection:
        install = uninstall_tool(connection, install_id)
    connection.close()
    return {"install": install}


def submit_speculative_plan(
    *,
    repo_root: str,
    task: str,
    base_ref: str,
    hypothesis_names: list[str],
) -> dict:
    connection = get_db()
    with connection:
        plan = create_speculative_plan(
            connection,
            config.orchestra_root(),
            repo_root=repo_root,
            task=task,
            base_ref=base_ref,
            hypothesis_names=hypothesis_names,
        )
    connection.close()
    return {"plan": plan}


def get_speculation(plan_id: str) -> dict:
    connection = get_db()
    plan = get_speculative_plan(connection, plan_id)
    connection.close()
    return {"plan": plan}


def list_speculations(status: Optional[str] = None) -> dict:
    connection = get_db()
    plans = list_speculative_plans(connection, status=status)
    connection.close()
    return {"plans": plans}


def prepare_speculation(plan_id: str) -> dict:
    connection = get_db()
    with connection:
        plan = prepare_speculative_worktrees(connection, plan_id)
    connection.close()
    return {"plan": plan}
