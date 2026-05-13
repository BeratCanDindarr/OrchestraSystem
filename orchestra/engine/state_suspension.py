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
    checkpoint_data = {
        "run_id": run.run_id,
        "mode": run.mode,
        "task": run.task,
        "status": run.status.value,
        "approval_state": run.approval_state.value,
        "checkpoint_version": run.checkpoint_version,
        "agents": [
            {
                "alias": a.alias,
                "provider": a.provider,
                "model": a.model,
                "status": a.status.value,
                "stdout_log": a.stdout_log,
                "confidence": a.confidence,
                "soft_failed": a.soft_failed,
                "validation_status": a.validation_status,
                "validation_reason": a.validation_reason,
            }
            for a in run.agents
        ],
        "reviews": run.reviews,
        "latest_review_stage": run.latest_review_stage,
        "latest_review_status": run.latest_review_status,
        "latest_review_winner": run.latest_review_winner,
        "summary": run.summary,
        "total_cost_usd": run.total_cost_usd or 0.0,
        "avg_confidence": run.avg_confidence or 0.0,
        "interrupt_state": run.interrupt_state.value if run.interrupt_state else None,
        "failure": run.failure.to_dict() if run.failure else None,
        "turns": run.turns,
    }

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

    run = OrchestraRun(mode=data.get("mode", "ask"), task=data.get("task", ""))
    run.run_id = data.get("run_id", run_id)
    run.status = RunStatus(data.get("status", "running"))
    run.approval_state = ApprovalStateEnum(data.get("approval_state", "not_required"))
    run.checkpoint_version = data.get("checkpoint_version", 0)
    run.reviews = data.get("reviews", [])
    run.latest_review_stage = data.get("latest_review_stage", "")
    run.latest_review_status = data.get("latest_review_status", "")
    run.latest_review_winner = data.get("latest_review_winner", "")
    run.summary = data.get("summary", "")
    run.total_cost_usd = data.get("total_cost_usd", 0.0)
    run.avg_confidence = data.get("avg_confidence", 0.0)
    run.turns = data.get("turns", 0)

    # Restore interrupt_state
    if data.get("interrupt_state"):
        run.interrupt_state = InterruptStateEnum(data["interrupt_state"])

    # Restore failure
    if data.get("failure"):
        run.failure = FailureState.from_dict(data["failure"])

    # Reconstruct agents
    for agent_data in data.get("agents", []):
        agent = AgentRun(
            alias=agent_data.get("alias", "unknown"),
            provider=agent_data.get("provider", ""),
            model=agent_data.get("model", ""),
        )
        agent.status = AgentStatus(agent_data.get("status", "pending"))
        agent.stdout_log = agent_data.get("stdout_log", "")
        agent.confidence = agent_data.get("confidence", 0.0)
        agent.soft_failed = agent_data.get("soft_failed", False)
        agent.validation_status = agent_data.get("validation_status", "not_run")
        agent.validation_reason = agent_data.get("validation_reason", "")
        run.agents.append(agent)

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
