"""Alias-level reputation scoring derived from persisted Orchestra runs."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _score_row(row: sqlite3.Row) -> float:
    terminal_runs = max(
        1,
        int(row["completed_runs"]) + int(row["failed_runs"]) + int(row["cancelled_runs"]),
    )
    total_runs = max(1, int(row["total_runs"]))
    success_rate = int(row["completed_runs"]) / terminal_runs
    cancel_rate = int(row["cancelled_runs"]) / total_runs
    soft_fail_rate = int(row["soft_failures"]) / total_runs
    validation_fail_rate = int(row["validation_failures"]) / total_runs
    review_win_rate = int(row["review_wins"]) / total_runs
    avg_confidence = float(row["avg_confidence"] or 0.0)

    score = 35.0
    score += success_rate * 30.0
    score += avg_confidence * 20.0
    score += review_win_rate * 10.0
    score -= soft_fail_rate * 15.0
    score -= validation_fail_rate * 15.0
    score -= cancel_rate * 5.0

    if total_runs >= 5:
        score += 3.0
    elif total_runs == 1:
        score -= 2.0

    return round(_clamp(score), 3)


def refresh_alias_reputation(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT
            a.alias,
            MAX(a.provider) AS provider,
            COUNT(*) AS total_runs,
            SUM(CASE WHEN a.status = 'completed' THEN 1 ELSE 0 END) AS completed_runs,
            SUM(CASE WHEN a.status = 'failed' THEN 1 ELSE 0 END) AS failed_runs,
            SUM(CASE WHEN a.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_runs,
            SUM(CASE WHEN COALESCE(a.soft_failed, 0) = 1 THEN 1 ELSE 0 END) AS soft_failures,
            SUM(CASE WHEN COALESCE(a.validation_status, 'not_run') = 'failed' THEN 1 ELSE 0 END) AS validation_failures,
            SUM(CASE WHEN COALESCE(r.latest_review_winner, '') = a.alias THEN 1 ELSE 0 END) AS review_wins,
            AVG(COALESCE(a.confidence, 0.0)) AS avg_confidence,
            AVG(COALESCE(a.estimated_cost_usd, 0.0)) AS avg_cost_usd,
            MAX(COALESCE(a.end_time, a.start_time, r.created_at)) AS last_run_at
        FROM agents a
        LEFT JOIN runs r ON r.run_id = a.run_id
        GROUP BY a.alias
        ORDER BY a.alias
        """
    ).fetchall()

    connection.execute("DELETE FROM alias_reputation")
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        connection.execute(
            """
            INSERT INTO alias_reputation (
                alias, provider, total_runs, completed_runs, failed_runs, cancelled_runs,
                soft_failures, validation_failures, review_wins,
                avg_confidence, avg_cost_usd, reputation_score, last_run_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["alias"],
                row["provider"],
                row["total_runs"],
                row["completed_runs"],
                row["failed_runs"],
                row["cancelled_runs"],
                row["soft_failures"],
                row["validation_failures"],
                row["review_wins"],
                round(float(row["avg_confidence"] or 0.0), 3),
                round(float(row["avg_cost_usd"] or 0.0), 6),
                _score_row(row),
                row["last_run_at"],
                now,
            ),
        )


def list_alias_reputation(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        """
        SELECT
            alias,
            provider,
            total_runs,
            completed_runs,
            failed_runs,
            cancelled_runs,
            soft_failures,
            validation_failures,
            review_wins,
            avg_confidence,
            avg_cost_usd,
            reputation_score,
            last_run_at,
            updated_at
        FROM alias_reputation
        ORDER BY reputation_score DESC, total_runs DESC, alias ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def choose_alias_by_reputation(
    connection: sqlite3.Connection,
    aliases: list[str],
    *,
    task: str = "",
    min_runs: int = 2,
    min_score_gap: float = 3.0,
    explore_ratio: int = 10,
) -> str | None:
    unique_aliases = [alias for alias in dict.fromkeys(aliases) if alias]
    if len(unique_aliases) < 2:
        return unique_aliases[0] if unique_aliases else None

    placeholders = ",".join("?" for _ in unique_aliases)
    rows = connection.execute(
        f"""
        SELECT alias, total_runs, reputation_score
        FROM alias_reputation
        WHERE alias IN ({placeholders})
        """,
        unique_aliases,
    ).fetchall()
    if not rows:
        return unique_aliases[0]

    ranked = sorted(
        (dict(row) for row in rows if int(row["total_runs"] or 0) >= min_runs),
        key=lambda row: (-float(row["reputation_score"] or 0.0), row["alias"]),
    )
    if not ranked:
        return unique_aliases[0]
    if len(ranked) == 1:
        return ranked[0]["alias"]

    top = ranked[0]
    second = ranked[1]
    if float(top["reputation_score"]) - float(second["reputation_score"]) < min_score_gap:
        return unique_aliases[0]

    digest = hashlib.sha256(task.encode("utf-8")).hexdigest() if task else ""
    should_explore = bool(digest) and int(digest[:8], 16) % max(1, explore_ratio) == 0
    return second["alias"] if should_explore else top["alias"]
