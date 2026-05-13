"""SQLite event-store helpers for Orchestra."""
from __future__ import annotations

import json
import gzip
import sqlite3
from pathlib import Path

from orchestra import config
from orchestra.storage.migrations import ensure_schema
from orchestra.storage.reputation import refresh_alias_reputation
from orchestra.storage.event_log import EventLog

# Singleton EventLog instance for thread-safe event logging (Block 1)
_event_log_instance: EventLog | None = None

def get_event_log() -> EventLog:
    """Get or create the singleton EventLog instance."""
    global _event_log_instance
    if _event_log_instance is None:
        _event_log_instance = EventLog()
    return _event_log_instance


def get_db() -> sqlite3.Connection:
    """Return a ready-to-use SQLite connection for .orchestra/orchestra.db."""
    db_path = config.db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    ensure_schema(connection)
    return connection


def upsert_run(manifest: dict) -> None:
    """Insert or update a run row and fully refresh its agent rows."""
    run_id = manifest.get("run_id")
    if not run_id:
        return

    connection = get_db()
    with connection:
        connection.execute(
            """
            INSERT INTO runs (
                run_id, mode, status, task, created_at, updated_at,
                total_cost_usd, avg_confidence,
                latest_review_stage, latest_review_status, latest_review_winner, latest_review_reason,
                approval_state, interrupt_state,
                checkpoint_version, failure_kind, failure_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                mode = excluded.mode,
                status = excluded.status,
                task = excluded.task,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                total_cost_usd = excluded.total_cost_usd,
                avg_confidence = excluded.avg_confidence,
                latest_review_stage = excluded.latest_review_stage,
                latest_review_status = excluded.latest_review_status,
                latest_review_winner = excluded.latest_review_winner,
                latest_review_reason = excluded.latest_review_reason,
                approval_state = excluded.approval_state,
                interrupt_state = excluded.interrupt_state,
                checkpoint_version = excluded.checkpoint_version,
                failure_kind = excluded.failure_kind,
                failure_message = excluded.failure_message
            """,
            (
                run_id,
                manifest.get("mode"),
                manifest.get("status"),
                manifest.get("task"),
                manifest.get("created_at"),
                manifest.get("updated_at"),
                manifest.get("total_cost_usd", 0.0),
                manifest.get("avg_confidence", 0.0),
                manifest.get("latest_review_stage", ""),
                manifest.get("latest_review_status", "not_run"),
                manifest.get("latest_review_winner", ""),
                manifest.get("latest_review_reason", ""),
                manifest.get("approval_state", "not_required"),
                manifest.get("interrupt_state", "idle"),
                manifest.get("checkpoint_version", 0),
                (manifest.get("failure") or {}).get("kind"),
                (manifest.get("failure") or {}).get("message"),
            ),
        )
        connection.execute("DELETE FROM agents WHERE run_id = ?", (run_id,))
        for agent in manifest.get("agents", []):
            connection.execute(
                """
                INSERT INTO agents (
                    run_id, alias, provider, model, status, start_time, end_time, elapsed, error,
                    estimated_completion_tokens, estimated_cost_usd, confidence, soft_failed,
                    validation_status, validation_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    agent.get("alias"),
                    agent.get("provider"),
                    agent.get("model"),
                    agent.get("status"),
                    agent.get("start_time"),
                    agent.get("end_time"),
                    agent.get("elapsed"),
                    agent.get("error"),
                    agent.get("estimated_completion_tokens", 0),
                    agent.get("estimated_cost_usd", 0.0),
                    agent.get("confidence", 0.0),
                    1 if agent.get("soft_failed", False) else 0,
                    agent.get("validation_status", "not_run"),
                    agent.get("validation_reason", ""),
                ),
            )
        refresh_alias_reputation(connection)
    connection.close()


def insert_event(run_id: str, event_payload: dict) -> None:
    """Insert a single event row."""
    event_name = event_payload.get("event", "")
    ts = event_payload.get("ts")
    extra = {
        key: value
        for key, value in event_payload.items()
        if key not in {"event", "ts"}
    }

    connection = get_db()
    with connection:
        connection.execute(
            "INSERT INTO events (run_id, event, ts, data) VALUES (?, ?, ?, ?)",
            (run_id, event_name, ts, json.dumps(extra, ensure_ascii=False)),
        )
    connection.close()


def insert_events(run_id: str, events_jsonl_path: Path) -> None:
    """Replace a run's event rows from an events.jsonl artifact."""
    connection = get_db()
    rows: list[tuple[str, str, str | None, str]] = []

    if events_jsonl_path.exists():
        if events_jsonl_path.suffix == ".gz":
            with gzip.open(events_jsonl_path, "rt", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = events_jsonl_path.read_text(encoding="utf-8").splitlines()
            
        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            extra = {
                key: value
                for key, value in payload.items()
                if key not in {"event", "ts"}
            }
            rows.append(
                (
                    run_id,
                    payload.get("event", ""),
                    payload.get("ts"),
                    json.dumps(extra, ensure_ascii=False),
                )
            )

    with connection:
        connection.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
        if rows:
            connection.executemany(
                "INSERT INTO events (run_id, event, ts, data) VALUES (?, ?, ?, ?)",
                rows,
            )
    connection.close()


def backfill() -> int:
    """Load all existing JSONL artifacts into SQLite and return processed run count."""
    root = config.artifact_root()
    if not root.exists():
        get_db().close()
        return 0

    processed = 0
    import itertools
    for manifest_path in itertools.chain(root.glob("*/manifest.json.gz"), root.glob("*/manifest.json")):
        try:
            if manifest_path.suffix == ".gz":
                with gzip.open(manifest_path, "rt", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        run_id = manifest.get("run_id")
        if not run_id:
            continue

        upsert_run(manifest)
        events_gz = manifest_path.parent / "events.jsonl.gz"
        events_path = events_gz if events_gz.exists() else manifest_path.parent / "events.jsonl"
        insert_events(run_id, events_path)
        processed += 1

    return processed
