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
