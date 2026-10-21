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
from orchestra.engine.runners.planned_runner import run_planned
from orchestra.engine.runners.resume_runner import resume_run
from orchestra.models import OrchestraRun, RunStatus
from orchestra.providers.fallback import available_aliases
from orchestra.router import classify, route_task
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


VALID_MODES = {"ask", "auto", "dual", "consensus"}


def _serialize_rows(rows) -> list[dict]:
    return [dict(row) for row in rows]




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


def execute_task(
    *,
    mode: str,
    task: str,
    alias: Optional[str] = None,
    previous_alias: Optional[str] = None,
    intent_hint: Optional[str] = None,
    fresh: bool = False,
    follow_up_to: Optional[str] = None,
    run_id: Optional[str] = None,
    interactive: bool = False,
) -> tuple:
    """Run a task; returns (run, route_class, routed_mode). The one place a task is routed and run.

    interactive: the terminal shows the live agent table and the runners' console output, and
    Ctrl-C cancels (the CLI); otherwise nothing is printed (MCP, the job worker).

    fresh: ask the model again even if this alias has answered this exact task before.
    It applies to single-agent (ask) runs, the only ones answered from stored results.
    follow_up_to: the id of an earlier run this task continues. The agents are given that
    discussion (the chain of runs it belongs to) before the new task. In ask mode the agent
    that answered the earlier run is used unless an alias is given.
    """
    from orchestra import live_status
    live_status.install()

    if mode not in VALID_MODES | {"planned"}:
        raise ValueError(f"Unknown mode: {mode}")
    display = {"show_live": interactive, "emit_console": interactive, "install_signal_handlers": interactive}

    # run_id: the id submit_run handed out; the run is created under it.
    follow: dict = {"run_id": run_id} if run_id else {}
    if follow_up_to:
        from orchestra.engine import follow_up
        from orchestra.engine.observability import resolve_run_id
        followed = resolve_run_id(follow_up_to) or follow_up_to
        if not artifacts.load_manifest(followed):
            raise ValueError(f"Run to follow up not found: {follow_up_to}")
        follow = {**follow, "follows_run_id": followed}
        if mode == "ask" and not alias:
            alias = follow_up.previous_alias(followed)

    route_class = None
    routed_mode = mode

    if mode == "planned":
        run = run_planned(task, **display)
    elif mode == "ask":
        selected_alias = alias or previous_alias
        if not selected_alias:
            decision = route_task(task, preferred_alias=alias, previous_alias=previous_alias, intent_hint=intent_hint or "")
            selected_alias = decision.alias or config.default_light_alias()
            
        run = run_ask(
            selected_alias,
            task,
            **display,
            **follow,
            fresh=fresh,
        )
    elif mode in ("dual", "consensus"):
        run = run_dual(
            task,
            stances=mode == "consensus",
            **display,
            **follow,
        )
    else:
        # Priority order for auto routing:
        # 1. Outcome routing (the mode that worked on similar tasks; opt-in via
        #    [outcome_routing] enabled=true, silent until there is enough history)
        # 2. Cascade routing (cheap-first escalation, opt-in via [cascade] enabled=true)
        # 3. Keyword/heuristic routing (always-available fallback)

        outcome_mode: str | None = None
        try:
            from orchestra.router.outcome_router import suggest_mode_for
            outcome_mode = suggest_mode_for(task)
        except Exception:
            logger.warning("outcome routing failed; the heuristic router decides", exc_info=True)

        cascade_run = None
        if outcome_mode and outcome_mode in {"ask", "dual"}:
            route_class = "outcome"
            routed_mode = outcome_mode
            # The history picks the mode; which agent answers is still the router's call.
            decision = route_task(task, preferred_alias=alias, previous_alias=previous_alias, intent_hint=intent_hint or "")
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
                    **display,
                )
            else:
                route_class = classify(task)
                decision = route_task(task, preferred_alias=alias, previous_alias=previous_alias, intent_hint=intent_hint or "")
                routed_mode = decision.mode

        if cascade_run is not None:
            run = cascade_run
        elif routed_mode == "ask":
            selected_alias = (decision.alias if decision else None) or alias or previous_alias or config.default_light_alias()
            run = run_ask(
                selected_alias,
                task,
                **display,
                **follow,
                fresh=fresh,
            )
        elif routed_mode == "dual":
            tier = classify_to_tier(task)
            run = run_dual(
                task,
                agents=agents_for_tier(tier),
                **display,
                **follow,
            )
        elif routed_mode == "planned":
            run = run_planned(task, **display)
        else:
            raise ValueError(f"Unknown auto route: {routed_mode}")

    return run, route_class, routed_mode


def run_task(*, mode: str, **kwargs) -> dict:
    """Run a task and return its manifest (MCP and the job worker; planned is not offered there).

    fresh: ask the model again even if this alias has answered this exact task before.
    follow_up_to: the id of an earlier run this task continues (see execute_task).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    run, route_class, routed_mode = execute_task(mode=mode, **kwargs)
    return {
        "route_class": route_class,
        "routed_mode": routed_mode,
        "run": run.to_manifest(),
        # The MCP payload reads which aliases another model answered for from these.
        "events": artifacts.read_events(run.run_id),
    }


# What the CLI runs besides tasks: the engine does the work, the terminal only prints.
from orchestra.engine.eval_compare import evaluate_modes  # noqa: E402
from orchestra.engine.replay import replay_synthesis  # noqa: E402


def resume(run_id: str, *, reject: bool = False) -> None:
    """Resume a paused or interrupted run in the terminal, or reject (fail) a paused one."""
    resume_run(run_id, approval_decision="reject" if reject else "approve")


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
    fresh: bool = False,
    follow_up_to: Optional[str] = None,
) -> dict:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "ask" and not alias and follow_up_to:
        from orchestra.engine import follow_up
        from orchestra.engine.observability import resolve_run_id
        alias = follow_up.previous_alias(resolve_run_id(follow_up_to) or follow_up_to)
    if mode == "ask" and not alias:
        raise ValueError("alias is required when mode='ask'")

    # The run exists from now on: the caller can watch it, or cancel it before it starts.
    queued = OrchestraRun(mode="dual" if mode == "consensus" else mode, task=task)
    artifacts.write_manifest(queued)
    connection = get_db()
    with connection:
        job = insert_job(
            connection,
            mode=mode,
            task=task,
            alias=alias,
            fresh=fresh,
            follow_up_to=follow_up_to,
            run_id=queued.run_id,
        )
    connection.close()

    cmd = [sys.executable, "-m", "orchestra", "job-worker", job["job_id"]]
    cwd = str(config.project_root())
    logger.info("spawning %s (cwd=%s)", cmd, cwd)
    try:
        subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        # No worker will ever pick it up: the queued run must not wait forever.
        queued.status = RunStatus.FAILED
        artifacts.write_manifest(queued)
        connection = get_db()
        with connection:
            mark_job_failed(connection, job["job_id"], "worker_spawn_failed", "the job worker could not be started")
        connection.close()
        raise
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

    run_id = job.get("run_id")
    if run_id and (artifacts.load_manifest(run_id) or {}).get("status") == RunStatus.CANCELLED.value:
        connection = get_db()
        with connection:
            mark_job_failed(connection, job_id, "cancelled", "cancelled before it started")
        connection.close()
        return {"job": get_job(job_id)["job"], "run": artifacts.load_manifest(run_id)}

    try:
        result = run_task(
            mode=job["mode"],
            task=job["task"],
            alias=job.get("alias"),
            fresh=bool(job.get("fresh")),
            follow_up_to=job.get("follow_up_to"),
            **({"run_id": run_id} if run_id else {}),
        )
        run_manifest = result["run"]
        if run_id and run_manifest.get("run_id") != run_id:
            # Answered by a stored run: the id the caller holds shows that answer too.
            run_manifest = {**run_manifest, "run_id": run_id, "replayed_from": run_manifest.get("run_id")}
            artifacts.write_manifest_data(run_id, run_manifest)
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

    # The cancel is on disk before the processes die: an agent whose CLI is killed returns
    # empty output, and the retry loop, which reads the manifest, must not take that for a
    # failure and start the next model.
    artifacts.write_manifest_data(run_id, manifest)
    manager.kill_all()
    artifacts.append_event(run_id, {"event": "run_cancelled", "reason": "service_cancel"})
    # The steps this run started (verifier, synthesiser) are cancelled with it.
    for child_id in artifacts.child_run_ids(run_id):
        cancel_run(child_id)
    return {"run": manifest, "already_finished": False}


def _pid_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)  # type: ignore[call-overload]
    except (OSError, TypeError, ValueError):
        return False
    return True


def reap_stale_runs() -> list[str]:
    """Fail the runs that are recorded as running although nothing is working on them.

    A run whose process was killed (or whose machine went down) keeps the status it had. A run
    counts as abandoned when its manifest has not changed for `config.stale_run_seconds()` and
    none of its agents has a live process. Returns the run ids that were failed.
    """
    limit = config.stale_run_seconds()
    connection = get_db()
    try:
        candidates = [row[0] for row in connection.execute(
            "SELECT run_id FROM runs WHERE status IN ('running', 'queued')")]
    finally:
        connection.close()
    now = datetime.now(timezone.utc)
    reaped: list[str] = []
    for run_id in candidates:
        manifest = artifacts.load_manifest(run_id)
        if not manifest or manifest.get("status") not in ("running", "queued"):
            continue
        try:
            idle = (now - datetime.fromisoformat(str(manifest.get("updated_at") or manifest.get("created_at")))).total_seconds()
        except (TypeError, ValueError):
            continue
        agents = manifest.get("agents", [])
        if idle < limit or any(agent.get("pid") and _pid_alive(agent["pid"]) for agent in agents):
            continue
        stamp = now.isoformat()
        for agent in agents:
            if agent.get("status") in ("queued", "started", "retrying"):
                agent.update(status="failed", end_time=agent.get("end_time") or stamp,
                             error=agent.get("error") or "interrupted", pid=None)
        manifest.update(status=RunStatus.FAILED.value, updated_at=stamp, failure={
            "kind": "UNKNOWN_RUNTIME_ERROR", "retryable": True, "source": "stale_run_reaper",
            "message": f"interrupted: no activity for {int(idle)}s and no agent process is alive",
        })
        artifacts.write_manifest_data(run_id, manifest)
        artifacts.append_event(run_id, {"event": "run_finished", "status": "failed", "reason": "stale"})
        reaped.append(run_id)
    return reaped


def _index_if_empty() -> None:
    """Rebuild the run index from the run folders when it holds nothing.

    Runs and events are indexed as they are written, so reading the history needs no
    reindex. Reindexing on every read re-parsed every run folder and replaced the events of
    runs that were still being written. An index that is empty next to existing run folders
    (a deleted database, a fresh checkout of old runs) is the one case worth repairing here;
    `orchestra backfill` does it on request.
    """
    connection = get_db()
    try:
        indexed = connection.execute("SELECT 1 FROM runs LIMIT 1").fetchone()
    finally:
        connection.close()
    if indexed is None:
        backfill()


def list_runs(limit: int = 20, mode: Optional[str] = None, status: Optional[str] = None,
              include_steps: bool = False) -> dict:
    """Recent runs. A run's own steps (verifier, synthesiser, planned nodes) are left out
    unless include_steps is set."""
    _index_if_empty()
    reap_stale_runs()
    connection = get_db()

    query = [
        "SELECT run_id, mode, status, task, created_at, updated_at,",
        "       latest_review_stage, latest_review_status, latest_review_winner, latest_review_reason,",
        "       approval_state, interrupt_state",
        "FROM runs",
        "WHERE 1=1",
    ]
    params: list[object] = []

    if not include_steps:
        query.append("AND (parent_run_id IS NULL OR parent_run_id = '')")
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


MEMORY_SNIPPET_CHARS = 500


def _tables_with_run_id(connection) -> list[str]:
    rows = connection.execute(
        "SELECT m.name FROM sqlite_master m JOIN pragma_table_info(m.name) p "
        "WHERE m.type = 'table' AND p.name = 'run_id'"
    ).fetchall()
    return [row[0] for row in rows]


def _forget_stored_answers(run_ids: list[str]) -> None:
    """Remove the runs' answers from the two stores that hand answers back: the replay store
    (the whole run, kept under one shared id) and the answer cache."""
    import sqlite3

    from orchestra.engine.runners import common

    wanted = set(run_ids)
    replay_db = config.artifact_root() / "event_log.db"
    if replay_db.exists():
        connection = sqlite3.connect(str(replay_db))
        try:
            connection.execute("PRAGMA busy_timeout=10000;")
            doomed = []
            for event_id, payload in connection.execute(
                    "SELECT id, payload FROM events WHERE run_id = 'global_idempotency'"):
                try:
                    stored = (json.loads(payload or "{}").get("result") or {}).get("run_id")
                except (TypeError, ValueError, AttributeError):
                    continue
                if stored in wanted:
                    doomed.append((event_id,))
            with connection:
                connection.executemany("DELETE FROM events WHERE id = ?", doomed)
        except sqlite3.Error:
            logger.warning("forget_runs: the replay store could not be cleaned", exc_info=True)
        finally:
            connection.close()
    try:
        cache = common._get_fts_cache()
        for run_id in run_ids if cache is not None else []:
            cache.forget_run(run_id)
    except Exception:
        logger.warning("forget_runs: the answer cache could not be cleaned", exc_info=True)


def forget_runs(run_ids: list) -> dict:
    """Erase runs for good: their folder, their database rows, their steps, their memory rows
    and the answers kept for replay and in the answer cache.

    Removing only the folder leaves the run in `ps` and its task text in memory. Steps of a
    run (verifier, synthesis, planned nodes) go with it. A run that is still running is
    left alone and reported.
    """
    import shutil
    import sqlite3

    from orchestra.engine.observability import resolve_run_id

    wanted: list[str] = []
    skipped: list[dict] = []
    for given in run_ids:
        run_id = resolve_run_id(str(given)) or str(given)
        manifest = artifacts.load_manifest(run_id)
        if not manifest:
            skipped.append({"run_id": str(given), "reason": "not found"})
            continue
        if manifest.get("status") in (RunStatus.RUNNING.value, RunStatus.QUEUED.value):
            skipped.append({"run_id": run_id, "reason": "still running: cancel it first"})
            continue
        wanted.append(run_id)
    for run_id in list(wanted):
        wanted.extend(child for child in artifacts.child_run_ids(run_id) if child not in wanted)
    if not wanted:
        return {"forgotten": [], "skipped": skipped}

    marks = ",".join("?" for _ in wanted)
    databases = [config.db_path(), config.artifact_root() / "event_log.db"]
    for path in databases:
        if not Path(path).exists():
            continue
        connection = sqlite3.connect(str(path))
        try:
            connection.execute("PRAGMA busy_timeout=10000;")
            with connection:
                for table in _tables_with_run_id(connection):
                    connection.execute(f"DELETE FROM {table} WHERE run_id IN ({marks})", wanted)
        finally:
            connection.close()

    try:
        from orchestra.storage.memory import get_memory
        memory = get_memory()
        for run_id in wanted:
            memory.delete_by_source(run_id)
    except Exception:
        logger.warning("forget_runs: the memory rows of the runs could not be removed", exc_info=True)
    _forget_stored_answers(wanted)

    for run_id in wanted:
        shutil.rmtree(config.artifact_root() / run_id, ignore_errors=True)
    return {"forgotten": wanted, "skipped": skipped}


def review_changes(base: str = "HEAD", paths: Optional[list] = None, focus: str = "", diff: Optional[str] = None) -> dict:
    """Review a code change with two agents and return findings that were checked against it.

    base: what to compare the working tree with (default HEAD: everything not yet committed;
    e.g. "main" for the whole branch). paths: limit the review to these files or folders.
    focus: what to look at in particular. diff: review this unified diff instead of asking git.
    Each finding has severity, file, line, the quoted line, the problem and a fix; findings
    whose quote is not in the diff are discarded and only counted.
    """
    import dataclasses

    from orchestra import live_status
    from orchestra.engine import code_review

    live_status.install()
    text = diff if diff else code_review.collect_diff(base, [str(p) for p in paths] if paths else None)
    if not text.strip():
        raise ValueError(f"Nothing to review: no changes against {base}" + (f" in {paths}" if paths else ""))
    limit = int(config.get("review_changes", "max_diff_chars", 120000))
    if len(text) > limit:
        raise ValueError(f"The diff is {len(text)} characters (limit {limit}): narrow it with paths.")
    title = f"Review of changes against {base}" + (f" in {', '.join(map(str, paths))}" if paths else "")
    run, findings, discarded = code_review.run_review(text, focus=focus, title=title)
    reviewed_by, not_reviewed_by = code_review.who_reviewed(run)
    return {
        "run": run.to_manifest(),
        "findings": [dataclasses.asdict(finding) for finding in findings],
        "discarded": len(discarded),
        # An empty findings list only means "sound" when reviewed_by is not empty.
        "reviewed_by": reviewed_by,
        "not_reviewed_by": not_reviewed_by,
    }


def memory_search_entries(query: str, limit: int = 5, include_tasks: bool = False, full: bool = False) -> dict:
    """Search memory. By default: notes only, each cut to a snippet.

    include_tasks: also return the task text stored for earlier runs.
    full: return whole entries instead of snippets (an entry can be a whole source file).
    """
    from orchestra.storage import memory as memory_store

    mem = memory_store.get_memory()
    rows = (mem.search(query, limit=limit) or []) if include_tasks else memory_store.search_notes(mem, query, limit=limit)
    if not full:
        rows = [_memory_snippet(row) for row in rows]
    return {"query": query, "limit": limit, "results": rows}


def _memory_snippet(row: dict) -> dict:
    content = str(row.get("content") or "")
    kept = {key: row[key] for key in ("id", "similarity", "source", "pinned", "created_at", "file_path") if key in row}
    return {**kept, "content": content[:MEMORY_SNIPPET_CHARS], "content_length": len(content)}


# ---------------------------------------------------------------------------
# Fleet View API — used by the orchestra_fleet MCP prompt skill
# ---------------------------------------------------------------------------
