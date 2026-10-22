"""Close the runs nothing is working on any more (moved from service.py unchanged)."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from orchestra import config
from orchestra.engine import artifacts
from orchestra.models import RunStatus
from orchestra.storage.db import get_db


def _pid_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)  # type: ignore[call-overload]
    except (OSError, TypeError, ValueError):
        return False
    return True


def _expire_approval(run_id: str, manifest: dict, idle: float, now: datetime) -> None:
    """Cancel a run nobody approved in time; it can no longer be resumed."""
    from orchestra.engine.state_suspension import delete_paused_run

    manifest.update(status=RunStatus.CANCELLED.value, updated_at=now.isoformat(), approval_state="expired", failure={
        "kind": "POLICY_DENIED", "retryable": True, "source": "approval_timeout",
        "message": f"no approval within {config.approval_timeout_hours():g} h (waited {int(idle // 3600)} h): "
                   "cancelled; run the task again if it is still wanted",
    })
    artifacts.write_manifest_data(run_id, manifest)
    artifacts.append_event(run_id, {"event": "run_cancelled", "reason": "approval_timeout"})
    connection = get_db()
    try:
        delete_paused_run(connection, run_id)
    finally:
        connection.close()


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
            "SELECT run_id FROM runs WHERE status IN ('running', 'queued', 'waiting_approval')")]
    finally:
        connection.close()
    now = datetime.now(timezone.utc)
    reaped: list[str] = []
    for run_id in candidates:
        manifest = artifacts.load_manifest(run_id)
        if not manifest or manifest.get("status") not in ("running", "queued", "waiting_approval"):
            continue
        try:
            idle = (now - datetime.fromisoformat(str(manifest.get("updated_at") or manifest.get("created_at")))).total_seconds()
        except (TypeError, ValueError):
            continue
        if manifest.get("status") == "waiting_approval":
            if idle >= config.approval_timeout_hours() * 3600:
                _expire_approval(run_id, manifest, idle, now)
                reaped.append(run_id)
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
