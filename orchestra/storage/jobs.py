"""Persisted async job helpers for Orchestra."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_job(
    connection: sqlite3.Connection,
    *,
    mode: str,
    task: str,
    alias: str | None = None,
    pause_after_round1: bool = False,
) -> dict:
    job_id = uuid.uuid4().hex[:10]
    created_at = _now()
    connection.execute(
        """
        INSERT INTO jobs (
            job_id, run_id, kind, status, mode, task, alias, pause_after_round1,
            created_at, started_at, finished_at, updated_at, worker_pid, error_code, error_message
        )
        VALUES (?, NULL, 'run', 'queued', ?, ?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, NULL)
        """,
        (
            job_id,
            mode,
            task,
            alias,
            1 if pause_after_round1 else 0,
            created_at,
            created_at,
        ),
    )
    return get_job(connection, job_id)


def get_job(connection: sqlite3.Connection, job_id: str) -> dict:
    row = connection.execute(
        """
        SELECT
            job_id, run_id, kind, status, mode, task, alias, pause_after_round1,
            created_at, started_at, finished_at, updated_at, worker_pid, error_code, error_message
        FROM jobs
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Job not found: {job_id}")
    return dict(row)


def list_jobs(connection: sqlite3.Connection, status: str | None = None, limit: int = 20) -> list[dict]:
    query = [
        """
        SELECT
            job_id, run_id, kind, status, mode, task, alias, pause_after_round1,
            created_at, started_at, finished_at, updated_at, worker_pid, error_code, error_message
        FROM jobs
        WHERE 1=1
        """
    ]
    params: list[object] = []
    if status:
        query.append("AND status = ?")
        params.append(status)
    query.append("ORDER BY created_at DESC LIMIT ?")
    params.append(limit)
    rows = connection.execute(" ".join(query), params).fetchall()
    return [dict(row) for row in rows]


def claim_job(connection: sqlite3.Connection, job_id: str, worker_pid: int | None = None) -> bool:
    now = _now()
    cursor = connection.execute(
        """
        UPDATE jobs
        SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ?, worker_pid = ?
        WHERE job_id = ? AND status = 'queued'
        """,
        (now, now, worker_pid, job_id),
    )
    return cursor.rowcount > 0


def mark_job_run_started(connection: sqlite3.Connection, job_id: str, run_id: str, worker_pid: int | None = None) -> None:
    connection.execute(
        """
        UPDATE jobs
        SET run_id = ?, status = 'running', updated_at = ?, worker_pid = COALESCE(?, worker_pid)
        WHERE job_id = ?
        """,
        (run_id, _now(), worker_pid, job_id),
    )


def mark_job_completed(connection: sqlite3.Connection, job_id: str, run_id: str) -> None:
    now = _now()
    connection.execute(
        """
        UPDATE jobs
        SET run_id = ?, status = 'completed', finished_at = ?, updated_at = ?
        WHERE job_id = ?
        """,
        (run_id, now, now, job_id),
    )


def mark_job_failed(connection: sqlite3.Connection, job_id: str, error_code: str, error_message: str) -> None:
    now = _now()
    connection.execute(
        """
        UPDATE jobs
        SET status = 'failed', finished_at = ?, updated_at = ?, error_code = ?, error_message = ?
        WHERE job_id = ?
        """,
        (now, now, error_code, error_message[:2000], job_id),
    )
