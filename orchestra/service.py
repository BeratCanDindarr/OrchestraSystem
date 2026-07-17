"""Headless service API for Orchestra automation surfaces such as MCP."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.process_group import ProcessGroupManager
from orchestra.engine.runners.ask_runner import run_ask
from orchestra.engine.runners.dual_runner import run_dual
from orchestra.engine.runners.critical_runner import run_critical
from orchestra.engine.runners.resume_runner import resume_run
from orchestra.engine.runners.planned_runner import run_planned
from orchestra.models import RunStatus
from orchestra.providers.fallback import available_aliases
from orchestra.router import classify, route_task, task_to_mode
from orchestra.router.classifier import agents_for_tier, classify_to_tier
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


def _serialize_rows(rows) -> list[dict]:
    return [dict(row) for row in rows]


def _daemon_pid_path() -> Path:
    return config.orchestra_root() / "orchestra-daemon.pid"


from orchestra.utils.config_values import (
    parse_cli_value as _parse_cli_value,
    toml_literal as _toml_literal,
    flatten_toml_lines as _flatten_toml_lines,
)


def _config_tree(include_env: bool = True) -> dict:
    source = config.raw_config() if include_env else config.file_config()
    return json.loads(json.dumps(source, ensure_ascii=False))


def _config_at_path(tree: dict, path: str):
    cursor = tree
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def _set_config_path(tree: dict, path: str, value) -> None:
    parts = path.split(".")
    cursor = tree
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def run_task(
    *,
    mode: str,
    task: str,
    alias: Optional[str] = None,
    previous_alias: Optional[str] = None,
    pause_after_round1: bool = False,
    intent_hint: Optional[str] = None,
) -> dict:
    from orchestra import live_status
    live_status.install()

    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    route_class = None
    routed_mode = mode

    if mode == "ask":
        selected_alias = alias or previous_alias
        if not selected_alias:
            decision = route_task(task, preferred_alias=alias, previous_alias=previous_alias, intent_hint=intent_hint or "")
            selected_alias = decision.alias or "cdx-fast"
            
        run = run_ask(
            selected_alias,
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
        # Priority order for auto routing:
        # 1. Outcome-weighted routing (historical EV, activates after ≥50 outcomes)
        # 2. Cascade routing (cheap-first escalation, opt-in via [cascade] enabled=true)
        # 3. Keyword/heuristic routing (always-available fallback)

        outcome_mode: str | None = None
        try:
            from orchestra.router.outcome_router import OutcomeRouter
            outcome_mode = OutcomeRouter().suggest_mode(task)
        except Exception:
            outcome_mode = None

        cascade_run = None
        if outcome_mode and outcome_mode in {"ask", "dual", "critical"}:
            route_class = "outcome"
            routed_mode = outcome_mode
            decision = None
        else:
            cascade_enabled = False
            try:
                cascade_cfg = config.cascade_config()
                cascade_enabled = bool(cascade_cfg.get("enabled", False))
            except Exception:
                pass

            if cascade_enabled:
                from orchestra.router.cascade_router import CascadeRouter
                route_class = "cascade"
                routed_mode = "ask"
                decision = None
                cascade_run = CascadeRouter().run(
                    task,
                    emit_console=False,
                    show_live=False,
                    install_signal_handlers=False,
                )
            else:
                route_class = classify(task)
                decision = route_task(task, preferred_alias=alias, previous_alias=previous_alias, intent_hint=intent_hint or "")
                routed_mode = decision.mode

        if cascade_run is not None:
            run = cascade_run
        elif routed_mode == "ask":
            selected_alias = (decision.alias if decision else None) or alias or previous_alias or "cdx-fast"
            run = run_ask(
                selected_alias,
                task,
                show_live=False,
                emit_console=False,
                install_signal_handlers=False,
            )
        elif routed_mode == "dual":
            tier = classify_to_tier(task)
            run = run_dual(
                task,
                agents=agents_for_tier(tier),
                show_live=False,
                emit_console=False,
                install_signal_handlers=False,
            )
        elif routed_mode == "critical":
            require_approval = (decision.require_approval if decision else False) or pause_after_round1
            run = run_critical(
                task,
                require_approval=require_approval,
                show_live=False,
                emit_console=False,
                install_signal_handlers=False,
                approval_behavior="pause" if require_approval else "continue",
            )
        else:
            raise ValueError(f"Unknown auto route: {routed_mode}")

    return {
        "route_class": route_class,
        "routed_mode": routed_mode,
        "run": run.to_manifest(),
    }


def continue_run(run_id: str) -> dict:
    from orchestra import live_status
    live_status.install()
    run = resume_run(
        run_id,
        force_continue=True,
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

    cmd = [sys.executable, "-m", "orchestra", "job-worker", job["job_id"]]
    cwd = str(config.project_root())
    logger.info("spawning %s (cwd=%s)", cmd, cwd)
    subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"job": job}


def execute_job(job_id: str) -> dict:
    from orchestra import live_status
    live_status.install()

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


def wait_for_run(
    *,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    timeout_seconds: int = 30,
    poll_interval_seconds: float = 1.0,
) -> dict:
    """Wait briefly for a run/job to settle.

    This is a convenience wrapper for short synchronous waits. For long-running
    tasks, callers should prefer submit_run() and poll with get_job()/get_run().
    """
    if not run_id and not job_id:
        raise ValueError("run_id or job_id is required")

    deadline = time.time() + max(0, timeout_seconds)
    last_payload: dict | None = None

    while True:
        if job_id:
            payload = get_job(job_id)
            last_payload = payload
            run_payload = payload.get("run")
            if run_payload:
                status = run_payload.get("status")
                if status in {
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                    RunStatus.WAITING_APPROVAL.value,
                }:
                    payload["timed_out"] = False
                    return payload
            else:
                job = payload.get("job") or {}
                if job.get("status") in {"failed", "completed"}:
                    payload["timed_out"] = False
                    return payload
        else:
            payload = get_run(run_id)
            last_payload = payload
            status = (payload.get("run") or {}).get("status")
            if status in {
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
                RunStatus.WAITING_APPROVAL.value,
            }:
                payload["timed_out"] = False
                return payload

        if time.time() >= deadline:
            result = dict(last_payload or {})
            result["timed_out"] = True
            return result

        time.sleep(max(0.1, poll_interval_seconds))


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
    from orchestra import live_status
    live_status.install()
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


def get_version() -> dict:
    from orchestra import __version__

    return {"version": __version__}


def watch_run_snapshot(run_id: str, normalized: bool = False) -> dict:
    run_payload = get_run(run_id)
    logs_payload = get_logs(run_id, normalized=normalized)
    return {
        "run_id": run_id,
        "run": run_payload["run"],
        "events": run_payload["events"],
        "logs": logs_payload["logs"],
        "missing_logs": logs_payload["missing"],
        "normalized": normalized,
    }


def run_backfill() -> dict:
    processed = backfill()
    return {"processed_runs": processed}


def start_daemon(host: str = "127.0.0.1", port: int = 8765) -> dict:
    pid_path = _daemon_pid_path()
    current = daemon_status()
    if current["running"]:
        return current

    cmd = [sys.executable, "-m", "orchestra", "daemon", "--host", host, "--port", str(port)]
    cwd = str(config.project_root())
    logger.info("spawning %s (cwd=%s)", cmd, cwd)
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path.write_text(json.dumps({"pid": process.pid, "host": host, "port": port}), encoding="utf-8")
    return {"running": True, "pid": process.pid, "host": host, "port": port, "started": True}


def daemon_status() -> dict:
    pid_path = _daemon_pid_path()
    if not pid_path.exists():
        return {"running": False}
    try:
        payload = json.loads(pid_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid"))
        os.kill(pid, 0)
        return {
            "running": True,
            "pid": pid,
            "host": payload.get("host", "127.0.0.1"),
            "port": payload.get("port", 8765),
        }
    except Exception:
        try:
            pid_path.unlink()
        except OSError:
            pass
        return {"running": False}


def stop_daemon() -> dict:
    pid_path = _daemon_pid_path()
    status = daemon_status()
    if not status.get("running"):
        return {"running": False, "stopped": False}
    pid = int(status["pid"])
    try:
        os.kill(pid, 15)
    except OSError:
        pass
    try:
        pid_path.unlink()
    except OSError:
        pass
    return {"running": False, "stopped": True, "pid": pid}


def config_get_value(path: str, include_env: bool = True) -> dict:
    tree = _config_tree(include_env=include_env)
    return {
        "path": path,
        "include_env": include_env,
        "value": _config_at_path(tree, path),
    }


def config_set_value(path: str, value, persist: bool = True) -> dict:
    parsed_value = _parse_cli_value(value)
    if persist:
        file_tree = _config_tree(include_env=False)
        _set_config_path(file_tree, path, parsed_value)
        target_path = config.config_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("\n".join(_flatten_toml_lines(file_tree)).strip() + "\n", encoding="utf-8")
        config.reload_config()
        return {"path": path, "value": parsed_value, "persisted": True, "config_path": str(target_path)}

    section, _, key = path.rpartition(".")
    if not section or not key:
        raise ValueError("path must contain a section and key, e.g. review.tie_margin")
    config.set(section, key, parsed_value)
    return {"path": path, "value": parsed_value, "persisted": False}


def config_validate_value() -> dict:
    config.reload_config()
    is_valid, errors = config.validate_config()
    return {
        "valid": is_valid and not bool(config.config_error()),
        "config_error": config.config_error(),
        "errors": errors,
        "config_path": str(config.config_path()),
    }


def config_init_value(force: bool = False) -> dict:
    path = config.config_path()
    if path.exists() and not force:
        return {"initialized": False, "config_path": str(path), "reason": "exists"}

    path.parent.mkdir(parents=True, exist_ok=True)
    template = _config_tree(include_env=False) or _config_tree(include_env=True)
    path.write_text("\n".join(_flatten_toml_lines(template)).strip() + "\n", encoding="utf-8")
    config.reload_config()
    return {"initialized": True, "config_path": str(path), "force": force}


def memory_add_entry(content: str, metadata: Optional[dict] = None, summary: Optional[str] = None) -> dict:
    from orchestra.storage.memory import get_memory

    mem = get_memory()
    ok = mem.add(content, metadata or {}, summary=summary, memory_type="semantic")
    return {"stored": ok, "content_length": len(content), "metadata": metadata or {}, "summary": summary}


def memory_search_entries(query: str, limit: int = 5) -> dict:
    from orchestra.storage.memory import get_memory

    mem = get_memory()
    return {"query": query, "limit": limit, "results": mem.search(query, limit=limit)}


def memory_index_project_entries(path: str = ".", glob_pattern: str = "**/*.md") -> dict:
    import re
    from orchestra.storage.memory import get_memory

    root = Path(path)
    files = list(root.glob(glob_pattern))
    mem = get_memory()
    indexed = 0
    failed: list[dict] = []

    for file_path in files:
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
            if not content.strip():
                continue

            if len(content) > 2000:
                chunks = [content[i:i + 2000] for i in range(0, len(content), 1800)]
                for chunk_index, chunk in enumerate(chunks):
                    if mem.add(chunk, {"path": str(file_path), "chunk": chunk_index}, memory_type="semantic"):
                        indexed += 1
            else:
                if mem.add(content, {"path": str(file_path)}, memory_type="semantic"):
                    indexed += 1

            if str(file_path).endswith(".cs"):
                usings = re.findall(r"using\s+([a-zA-Z0-9_.]+);", content)
                for using_name in usings:
                    if not using_name.startswith("System") and not using_name.startswith("UnityEngine"):
                        mem.add_relation(str(file_path), using_name, "imports_namespace")
        except Exception as exc:  # noqa: BLE001
            failed.append({"path": str(file_path), "error": str(exc)})

    return {
        "root": str(root),
        "glob": glob_pattern,
        "files_seen": len(files),
        "indexed_entries": indexed,
        "failed": failed,
    }


def inspect_route(task: str, preferred_alias: Optional[str] = None, intent_hint: str = "") -> dict:
    decision = route_task(task, preferred_alias=preferred_alias, intent_hint=intent_hint)
    route_class = classify(task)
    tier = classify_to_tier(task)
    aliases = agents_for_tier(tier)
    return {
        "task": task,
        "route_class": route_class,
        "task_to_mode": task_to_mode(task),
        "decision": {
            "mode": decision.mode,
            "alias": decision.alias,
            "require_approval": decision.require_approval,
            "reason": decision.reason,
            "classifier": decision.classifier,
            "confidence": decision.confidence,
        },
        "tier": tier,
        "tier_aliases": aliases,
    }


def get_trace(run_id: str) -> dict:
    from orchestra.engine.artifacts import read_events

    events = read_events(run_id)
    spans = [event for event in events if event.get("event") == "span_finished"]
    return {"run_id": run_id, "spans": spans}


def get_timeline(run_id: str) -> dict:
    from orchestra.engine.observability import (
        enrich_timeline_events,
        load_events,
        load_manifest,
        resolve_run_id,
    )

    resolved_id = resolve_run_id(run_id) or run_id
    manifest = load_manifest(resolved_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")
    events = load_events(resolved_id)
    if not events:
        raise ValueError(f"No events found for run: {resolved_id}")
    return {
        "run_id": resolved_id,
        "run": manifest.to_manifest(),
        "events": enrich_timeline_events(events),
    }


def compare_runs(run_id_a: str, run_id_b: str) -> dict:
    from orchestra.engine.observability import load_manifest, resolve_run_id

    resolved_a = resolve_run_id(run_id_a) or run_id_a
    resolved_b = resolve_run_id(run_id_b) or run_id_b
    manifest_a = load_manifest(resolved_a)
    manifest_b = load_manifest(resolved_b)
    if not manifest_a or not manifest_b:
        raise ValueError("One or both runs could not be loaded")

    cost_a = manifest_a.total_cost_usd or 0.0
    cost_b = manifest_b.total_cost_usd or 0.0
    conf_a = manifest_a.avg_confidence or 0.0
    conf_b = manifest_b.avg_confidence or 0.0
    agents_a = len(manifest_a.agents or [])
    agents_b = len(manifest_b.agents or [])

    return {
        "run_a": manifest_a.to_manifest(),
        "run_b": manifest_b.to_manifest(),
        "delta": {
            "cost_usd": round(cost_b - cost_a, 6),
            "confidence": round(conf_b - conf_a, 4),
            "agents": agents_b - agents_a,
            "cost_percent": round(((cost_b - cost_a) / cost_a * 100), 1) if cost_a > 0 else 0.0,
        },
    }


def replay_run_output(run_id: str, synthesizer: str = "cld-fast") -> dict:
    from orchestra.engine.observability import resolve_run_id
    from orchestra.engine.synthesizer import build_synthesis_prompt, outputs_sufficient
    from orchestra.engine.runner import run_ask
    from orchestra.models import AgentRun, AgentStatus
    from orchestra.providers.fallback import available_aliases

    resolved_id = resolve_run_id(run_id) or run_id
    manifest = artifacts.load_manifest(resolved_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")
    agents_meta = manifest.get("agents") or []
    completed_meta = [agent for agent in agents_meta if agent.get("status") == "completed" and not agent.get("soft_failed")]
    if len(completed_meta) < 2:
        raise ValueError("Replay requires dual+ runs with at least 2 completed agents")
    if synthesizer not in set(available_aliases()):
        raise ValueError(f"Unknown synthesizer: {synthesizer}")

    reconstructed: list[AgentRun] = []
    for meta in completed_meta[:2]:
        alias = meta.get("alias", "")
        log = artifacts.read_agent_log(resolved_id, alias) or ""
        agent = AgentRun(alias=alias, provider=meta.get("provider", ""), model=meta.get("model", ""))
        agent.status = AgentStatus.COMPLETED
        agent.confidence = float(meta.get("confidence") or 0.5)
        agent.stdout_log = log
        agent.soft_failed = bool(meta.get("soft_failed", False))
        agent.validation_status = meta.get("validation_status", "not_run")
        reconstructed.append(agent)

    agent_a, agent_b = reconstructed[0], reconstructed[1]
    if not outputs_sufficient(agent_a, agent_b):
        raise ValueError("Cached outputs are insufficient for replay")

    synth_prompt = build_synthesis_prompt(agent_a, agent_b)
    synth_run = run_ask(synthesizer, synth_prompt, emit_console=False, show_live=False)
    if not synth_run.agents or not synth_run.agents[0].stdout_log:
        raise ValueError("Replay synthesis produced no output")

    return {
        "run_id": resolved_id,
        "synthesizer": synthesizer,
        "replay_run_id": synth_run.run_id,
        "original_output": (manifest.get("summary") or "").strip(),
        "replayed_output": synth_run.agents[0].stdout_log,
        "original_cost_usd": round(float(manifest.get("total_cost_usd") or 0.0), 6),
        "replayed_cost_usd": round(float(synth_run.total_cost_usd or 0.0), 6),
        "original_confidence": round(float(manifest.get("avg_confidence") or 0.0), 3),
        "replayed_confidence": round(float(synth_run.avg_confidence or 0.0), 3),
        "agents": completed_meta[:2],
    }


def batch_run_tasks(tasks: list[dict]) -> dict:
    results: list[dict] = []
    for index, entry in enumerate(tasks, 1):
        try:
            result = run_task(
                mode=entry["mode"],
                task=entry["task"],
                alias=entry.get("alias"),
                pause_after_round1=bool(entry.get("pause_after_round1", False)),
                intent_hint=entry.get("intent_hint"),
            )
            results.append({"index": index, "status": "ok", **result})
        except Exception as exc:  # noqa: BLE001
            results.append({"index": index, "status": "error", "error": str(exc), "task": entry.get("task", "")})
    return {"tasks": tasks, "results": results}


def get_eval_batch(task_hashes: list[str]) -> dict:
    from orchestra.engine.eval_harness import batch_evaluate

    return {"results": [result.to_dict() for result in batch_evaluate(task_hashes)]}


def get_eval_stats(task_hashes: list[str]) -> dict:
    from orchestra.engine.eval_harness import compute_eval_stats

    return compute_eval_stats(task_hashes)


def undo_run_changes(run_id: str) -> dict:
    from orchestra.engine.backtrack import undo_last_run

    return {"run_id": run_id, "undone": undo_last_run(run_id)}


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


# ---------------------------------------------------------------------------
# Fleet View API — used by the orchestra_fleet MCP prompt skill
# ---------------------------------------------------------------------------

def get_run_plan(task: str, mode: str, alias: Optional[str] = None) -> dict:
    """Return agent execution plan for a mode without creating any run."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "ask":
        if not alias:
            raise ValueError("alias is required when mode='ask'")
        return {"mode": "ask", "providers": [alias], "has_postprocess": False}
    if mode == "dual":
        return {"mode": "dual", "providers": ["cdx-deep", "gmn-pro"], "has_postprocess": True}
    if mode == "critical":
        return {"mode": "critical", "providers": ["cdx-fast", "gmn-fast"], "has_postprocess": True}
    # auto: use tier routing
    tier = classify_to_tier(task)
    providers = agents_for_tier(tier)
    if len(providers) < 2:
        providers = (providers + ["gmn-pro"])[:2]
    return {"mode": "auto", "providers": providers[:2], "has_postprocess": True}


def create_fleet_run(task: str, mode: str, alias: Optional[str] = None) -> dict:
    """Create an Orchestra run record and persist manifest without executing agents."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "ask" and not alias:
        raise ValueError("alias is required when mode='ask'")
    from orchestra.models import OrchestraRun, RunStatus as RS
    run = OrchestraRun(mode=mode, task=task)
    run.status = RS.RUNNING
    artifacts.write_manifest(run)
    artifacts.append_event(run.run_id, {"event": "run_started", "mode": mode, "source": "fleet"})
    return {"run_id": run.run_id, "run": run.to_manifest()}


def run_fleet_agent(run_id: str, alias: str) -> dict:
    """Run a single provider agent and merge its result into the fleet run manifest."""
    import shutil
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")
    task = manifest["task"]
    temp_run = run_ask(alias, task, show_live=False, emit_console=False, install_signal_handlers=False)
    if not temp_run.agents:
        raise ValueError(f"No agent result returned for alias={alias}")
    agent = temp_run.agents[0]
    # Copy agent log files from temp run to fleet run directory so finalize can find them
    from orchestra import config as _cfg
    temp_agents_dir = _cfg.artifact_root() / temp_run.run_id / "agents"
    fleet_agents_dir = _cfg.artifact_root() / run_id / "agents"
    fleet_agents_dir.mkdir(parents=True, exist_ok=True)
    if temp_agents_dir.exists():
        for src in temp_agents_dir.iterdir():
            dst = fleet_agents_dir / src.name.replace(temp_run.run_id, run_id) if temp_run.run_id in src.name else fleet_agents_dir / src.name
            shutil.copy2(src, dst)
    _merge_agent_to_fleet_manifest(run_id, agent)
    return {
        "run_id": run_id,
        "alias": alias,
        "status": agent.status.value,
        "confidence": round(agent.confidence, 3),
        "elapsed": agent.elapsed,
    }


def finalize_fleet_run(run_id: str) -> dict:
    """Run post-processing (review + verification + synthesis) on a completed fleet run."""
    manifest = artifacts.load_manifest(run_id)
    if not manifest:
        raise ValueError(f"Run not found: {run_id}")
    from orchestra.models import OrchestraRun
    from orchestra.engine.runner import (
        _record_round1_review,
        _run_verification_loop,
        _synthesize_if_possible,
        _finalize,
    )
    run = OrchestraRun.from_manifest(manifest)
    _record_round1_review(run)
    _run_verification_loop(run, run.task, show_live=False, emit_console=False, install_signal_handlers=False)
    _synthesize_if_possible(run, run.task, show_live=False, emit_console=False, install_signal_handlers=False)
    _finalize(run)
    return {"run_id": run_id, "run": run.to_manifest()}


def _merge_agent_to_fleet_manifest(run_id: str, agent) -> None:
    """Atomically update one agent entry in the shared fleet run manifest using file locking."""
    import fcntl
    import json as _json
    agent_dict = {
        "alias": agent.alias,
        "provider": agent.provider,
        "model": agent.model,
        "status": agent.status.value,
        "start_time": agent.start_time,
        "end_time": agent.end_time,
        "elapsed": agent.elapsed,
        "error": agent.error,
        "pid": None,
        "estimated_completion_tokens": agent.estimated_completion_tokens,
        "estimated_cost_usd": agent.estimated_cost_usd,
        "confidence": agent.confidence,
        "soft_failed": agent.soft_failed,
        "validation_status": agent.validation_status,
        "validation_reason": agent.validation_reason,
    }
    path = artifacts.manifest_path(run_id)
    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = _json.load(f)
            agents = data.get("agents", [])
            idx = next((i for i, a in enumerate(agents) if a.get("alias") == agent.alias), None)
            if idx is not None:
                agents[idx] = agent_dict
            else:
                agents.append(agent_dict)
            data["agents"] = agents
            f.seek(0)
            f.write(_json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    try:
        from orchestra.storage.db import upsert_run
        upsert_run(data)
    except Exception:
        pass

def discover_sessions() -> dict:
    from orchestra.engine.session_discovery import SessionDiscovery
    import dataclasses
    sd = SessionDiscovery()
    return {"sessions": [dataclasses.asdict(s) for s in sd.discover_all()]}

def search_sessions(query: str) -> dict:
    from orchestra.engine.session_discovery import SessionDiscovery
    import dataclasses
    sd = SessionDiscovery()
    return {"sessions": [dataclasses.asdict(s) for s in sd.search(query)]}

def resume_session(session_id: str) -> dict:
    from orchestra.engine.session_discovery import SessionDiscovery
    sd = SessionDiscovery()
    run_id = sd.resume_into_orchestra(session_id)
    return {"run_id": run_id}

def list_cues() -> dict:
    from orchestra.engine.cue import CueEngine
    engine = CueEngine()
    import dataclasses
    return {"rules": [dataclasses.asdict(r) for r in engine.rules]}

def trigger_cue(event_type: str, context: dict = None) -> dict:
    from orchestra.engine.cue import CueEngine
    engine = CueEngine()
    engine.trigger(event_type, context or {})
    return {"status": "triggered"}

def run_playbook(path: str, alias: str = "cld-fast") -> dict:
    from orchestra.engine.playbook import PlaybookRunner
    PlaybookRunner(alias=alias).run(path)
    return {"status": "completed"}

def get_playbook_status(path: str) -> dict:
    from orchestra.engine.playbook import PlaybookParser
    import dataclasses
    playbook = PlaybookParser.parse(path)
    return {"playbook": dataclasses.asdict(playbook)}
