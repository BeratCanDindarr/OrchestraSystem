"""HITL state suspension and resumption for approval gates."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from orchestra.config import checkpoint_ttl_hours
from orchestra.models import OrchestraRun


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_at(hours: int | None = None) -> str:
    if hours is None:
        hours = checkpoint_ttl_hours()
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def suspend_run(
    connection: sqlite3.Connection,
    run: OrchestraRun,
    paused_by: str = "system",
) -> None:
    """Serialize and persist a paused run to database.

    Captures:
    - All agent state (stdout_log, confidence, validation status)
    - Review history and decisions
    - Partial outputs for synthesis continuation
    """
    # The whole manifest, plus what a manifest leaves out: each agent's answer. A paused run
    # goes on with exactly what it had (stances, the run it follows, costs, review state).
    checkpoint_data = run.to_manifest()
    for stored, agent in zip(checkpoint_data["agents"], run.agents):
        stored["stdout_log"] = agent.stdout_log

    connection.execute(
        """
        INSERT INTO paused_runs
            (run_id, checkpoint_data, paused_at, paused_by, expires_at, status, resume_owner, resume_lease_expires_at)
        VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL)
        ON CONFLICT(run_id) DO UPDATE SET
            checkpoint_data = excluded.checkpoint_data,
            paused_at = excluded.paused_at,
            paused_by = excluded.paused_by,
            expires_at = excluded.expires_at,
            status = 'pending',
            resume_owner = NULL,
            resume_lease_expires_at = NULL
        """,
        (
            run.run_id,
            json.dumps(checkpoint_data),
            _now(),
            paused_by,
            _expires_at(),
        ),
    )
    connection.commit()


def resume_run(
    connection: sqlite3.Connection,
    run_id: str,
    owner_id: Optional[str] = None,
    lease_minutes: int = 5,
) -> Optional[OrchestraRun]:
    """Load and claim a paused run from database, reconstructing OrchestraRun.

    Implements atomic compare-and-swap (CAS) to ensure only one process can
    resume a given run at a time. Returns None if:
    - Run not found
    - Already claimed by another process (status='resuming')
    - Checkpoint has expired (>TTL old)
    - Data corruption detected

    Args:
        connection: SQLite connection
        run_id: Run ID to resume
        owner_id: Optional process owner ID (defaults to os.getpid())
        lease_minutes: Lease duration in minutes before stale claim is reclaimable

    Returns:
        OrchestraRun if claim succeeded, None if already claimed or not found
    """
    import os
    from orchestra.models import AgentRun, AgentStatus, RunStatus
    from orchestra.state import ApprovalState as ApprovalStateEnum
    from orchestra.state import InterruptState as InterruptStateEnum
    from orchestra.state import FailureState

    if owner_id is None:
        owner_id = str(os.getpid())
    lease_expires = (datetime.now(timezone.utc) + timedelta(minutes=lease_minutes)).isoformat()

    # Atomic claim: succeed only if status='pending' OR stale lease expired
    cursor = connection.execute(
        """UPDATE paused_runs
           SET status = 'resuming',
               resume_owner = ?,
               resume_lease_expires_at = ?
           WHERE run_id = ?
             AND (status = 'pending'
                  OR (status = 'resuming' AND resume_lease_expires_at < ?))""",
        (owner_id, lease_expires, run_id, _now()),
    )
    connection.commit()
    if cursor.rowcount == 0:
        return None  # already claimed or not found

    # Now safe to read (we own the claim)
    row = connection.execute(
        "SELECT checkpoint_data, expires_at FROM paused_runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()

    if not row:
        return None

    checkpoint_data_str, expires_at = row
    expires_dt = datetime.fromisoformat(expires_at)
    if datetime.now(timezone.utc) > expires_dt:
        # Checkpoint expired, delete it
        connection.execute("DELETE FROM paused_runs WHERE run_id = ?", (run_id,))
        connection.commit()
        return None

    try:
        data = json.loads(checkpoint_data_str)
    except json.JSONDecodeError:
        return None

    try:
        run = OrchestraRun.from_manifest({"run_id": run_id, "status": "running", **data})
    except (TypeError, ValueError):
        return None  # a checkpoint this version cannot read
    for agent, stored in zip(run.agents, data.get("agents", [])):
        agent.stdout_log = stored.get("stdout_log", "") or ""
    return run


def delete_paused_run(connection: sqlite3.Connection, run_id: str) -> None:
    """Remove a paused run from suspension storage."""
    connection.execute("DELETE FROM paused_runs WHERE run_id = ?", (run_id,))
    connection.commit()


def release_resume_claim(connection: sqlite3.Connection, run_id: str) -> None:
    """Reset resume claim to 'pending' if resume failed partway through.

    Call this if resume_run() succeeded but the run execution failed before
    completing, to allow another process to retry the resume.

    Args:
        connection: SQLite connection
        run_id: Run ID to release
    """
    connection.execute(
        "UPDATE paused_runs SET status = 'pending', resume_owner = NULL, "
        "resume_lease_expires_at = NULL WHERE run_id = ?",
        (run_id,),
    )
    connection.commit()


def cleanup_expired_paused_runs(connection: sqlite3.Connection) -> int:
    """Delete all paused runs that have exceeded their 48h expiry.

    Returns count of deleted rows.
    """
    cursor = connection.execute(
        "SELECT run_id FROM paused_runs WHERE expires_at < ?",
        (_now(),),
    )
    expired_ids = [row[0] for row in cursor.fetchall()]

    for run_id in expired_ids:
        connection.execute("DELETE FROM paused_runs WHERE run_id = ?", (run_id,))

    if expired_ids:
        connection.commit()

    return len(expired_ids)
