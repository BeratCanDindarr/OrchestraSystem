"""SQLite schema migrations for Orchestra (Autonomous Evolution Version)."""
from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS agents (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS alias_reputation (alias TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tool_proposals (proposal_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS blackboard (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tool_installs (install_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS speculative_plans (plan_id TEXT PRIMARY KEY);
"""

# Table -> Column -> Type/Constraint
COLUMNS = {
    "runs": {
        "mode": "TEXT", "status": "TEXT", "task": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
        "total_cost_usd": "REAL DEFAULT 0.0", "avg_confidence": "REAL DEFAULT 0.0",
        "latest_review_stage": "TEXT DEFAULT ''", "latest_review_status": "TEXT DEFAULT 'not_run'",
        "latest_review_winner": "TEXT DEFAULT ''", "latest_review_reason": "TEXT DEFAULT ''",
        "approval_state": "TEXT DEFAULT 'not_required'", "interrupt_state": "TEXT DEFAULT 'idle'",
        "checkpoint_version": "INTEGER DEFAULT 0", "last_seq_id": "INTEGER DEFAULT 0",
        "failure_kind": "TEXT", "failure_message": "TEXT",
        "dissent_text": "TEXT DEFAULT ''"  # Phase 7: Dissent Tracking
    },
    "agents": {
        "run_id": "TEXT", "alias": "TEXT", "provider": "TEXT", "model": "TEXT", "status": "TEXT",
        "start_time": "TEXT", "end_time": "TEXT", "elapsed": "TEXT", "error": "TEXT",
        "estimated_completion_tokens": "INTEGER DEFAULT 0", "estimated_cost_usd": "REAL DEFAULT 0.0",
        "confidence": "REAL DEFAULT 0.0", "soft_failed": "INTEGER DEFAULT 0",
        "validation_status": "TEXT DEFAULT 'not_run'", "validation_reason": "TEXT DEFAULT ''"
    },
    "events": {
        "run_id": "TEXT", "seq": "INTEGER DEFAULT 0", "event_type": "TEXT", "version": "INTEGER DEFAULT 1",
        "causation_id": "TEXT", "idempotency_key": "TEXT", "ts": "TEXT", "data": "TEXT"
    },
    "blackboard": {
        "run_id": "TEXT", "alias": "TEXT", "content": "TEXT", "tags": "TEXT", "ts": "TEXT"
    }
}

def _ensure_robust_columns(connection: sqlite3.Connection):
    for table, cols in COLUMNS.items():
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, spec in cols.items():
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")

def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    _ensure_robust_columns(connection)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_blackboard_run_id ON blackboard(run_id)")
