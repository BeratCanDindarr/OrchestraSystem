"""SQLite schema migrations for Orchestra (Autonomous Evolution Version)."""
from __future__ import annotations

import importlib
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS agents (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT,
    ts REAL NOT NULL DEFAULT (unixepoch('subsec')),
    seq INTEGER UNIQUE NOT NULL,
    created_at REAL DEFAULT (unixepoch('subsec'))
);
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_run_id_seq ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_seq ON events(seq);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE TABLE IF NOT EXISTS alias_reputation (alias TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tool_proposals (proposal_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS blackboard (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tool_installs (install_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS speculative_plans (plan_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS paused_runs (run_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS planned_checkpoints (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    result_json TEXT,
    status TEXT NOT NULL,
    completed_at REAL,
    error_message TEXT,
    cost_usd REAL DEFAULT 0.0,
    PRIMARY KEY (run_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_planned_checkpoints_run_id ON planned_checkpoints(run_id);
CREATE TABLE IF NOT EXISTS eval_runs (run_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS eval_scenarios (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS eval_results (id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS skill_reputation (skill_name TEXT PRIMARY KEY);
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
        "event_type": "TEXT NOT NULL",
        "run_id": "TEXT NOT NULL",
        "payload": "TEXT",
        "ts": "REAL NOT NULL DEFAULT (unixepoch('subsec'))",
        "seq": "INTEGER UNIQUE NOT NULL",
        "created_at": "REAL DEFAULT (unixepoch('subsec'))",
        "id": "TEXT PRIMARY KEY"
    },
    "blackboard": {
        "run_id": "TEXT", "alias": "TEXT", "content": "TEXT", "tags": "TEXT", "ts": "TEXT"
    },
    "jobs": {
        "run_id": "TEXT", "kind": "TEXT DEFAULT 'run'", "status": "TEXT DEFAULT 'queued'",
        "mode": "TEXT", "task": "TEXT", "alias": "TEXT",
        "pause_after_round1": "INTEGER DEFAULT 0",
        "created_at": "TEXT", "started_at": "TEXT", "finished_at": "TEXT", "updated_at": "TEXT",
        "worker_pid": "INTEGER", "error_code": "TEXT", "error_message": "TEXT"
    },
    "alias_reputation": {
        "provider": "TEXT", "total_runs": "INTEGER DEFAULT 0",
        "completed_runs": "INTEGER DEFAULT 0", "failed_runs": "INTEGER DEFAULT 0",
        "cancelled_runs": "INTEGER DEFAULT 0", "soft_failures": "INTEGER DEFAULT 0",
        "validation_failures": "INTEGER DEFAULT 0", "review_wins": "INTEGER DEFAULT 0",
        "avg_confidence": "REAL DEFAULT 0.0", "avg_cost_usd": "REAL DEFAULT 0.0",
        "reputation_score": "REAL DEFAULT 50.0", "last_run_at": "TEXT", "updated_at": "TEXT",
        # Explicit per-review-gate outcome counters (for ±7.5% delta in reviewer)
        "outcome_wins": "INTEGER DEFAULT 0", "outcome_losses": "INTEGER DEFAULT 0",
        "outcome_soft_fails": "INTEGER DEFAULT 0", "outcome_ties": "INTEGER DEFAULT 0",
    },
    "paused_runs": {
        "checkpoint_data": "TEXT NOT NULL", "paused_at": "TEXT NOT NULL",
        "paused_by": "TEXT DEFAULT 'system'", "expires_at": "TEXT NOT NULL",
        "resume_attempt_count": "INTEGER DEFAULT 0", "last_resume_error": "TEXT",
        "status": "TEXT DEFAULT 'pending'",
        "resume_owner": "TEXT",
        "resume_lease_expires_at": "TEXT",
    },
    "eval_runs": {
        "task": "TEXT", "mode": "TEXT", "status": "TEXT", "created_at": "TEXT"
    },
    "eval_scenarios": {
        "scenario_id": "TEXT", "task_name": "TEXT", "gold_standard_answer": "TEXT"
    },
    "eval_results": {
        "run_id": "TEXT", "scenario_id": "TEXT", "outcome": "TEXT",
        "tokens_used": "INTEGER DEFAULT 0", "pass_at_1": "REAL DEFAULT 0.0",
        "created_at": "TEXT",
        "task_hash": "TEXT",
        "answer_quality": "REAL", "factual_grounding": "REAL",
        "source_attribution": "REAL", "conciseness": "REAL",
        "calibration": "REAL", "composite_score": "REAL",
    },
    "speculative_plans": {
        "repo_root": "TEXT", "base_ref": "TEXT", "task": "TEXT", "status": "TEXT",
        "manifest_path": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
    },
    "skill_reputation": {
        "total_runs": "INTEGER DEFAULT 0",
        "completed_runs": "INTEGER DEFAULT 0",
        "failed_runs": "INTEGER DEFAULT 0",
        "reputation_score": "REAL DEFAULT 50.0",
        "updated_at": "TEXT",
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
    run_migrations(connection)  # Run numbered migrations (idempotent; skips already-applied)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_blackboard_run_id ON blackboard(run_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_paused_runs_expires ON paused_runs(expires_at)")


class MigrationError(Exception):
    """Raised on migration failure."""
    pass


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Read PRAGMA user_version (0 = fresh database)."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Write PRAGMA user_version atomically (no transaction needed for PRAGMA)."""
    conn.execute(f"PRAGMA user_version = {version}")


def _get_migration_modules() -> list[str]:
    """Discover NNN_*.py files in the migrations/ subdirectory, sorted."""
    migrations_dir = Path(__file__).parent / "migrations"
    modules = []
    for path in sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.py")):
        if path.stem != "__init__":
            modules.append(path.stem)
    return modules


def run_migrations(
    conn: sqlite3.Connection,
    target_version: int | None = None,
) -> dict:
    """Run all pending migrations up to target_version (or latest if None).

    Returns:
        {
            "migrations_run": int,
            "final_version": int,
            "errors": list[str],
            "migration_details": list[{"module": str, "result": dict}],
        }
    """
    conn.execute("PRAGMA foreign_keys = ON")
    modules = _get_migration_modules()
    if target_version is None:
        target_version = len(modules)

    migrations_run = 0
    details = []

    current = _get_schema_version(conn)

    for i, module_name in enumerate(modules):
        version_after = i + 1
        if version_after > target_version:
            break
        if version_after <= current:
            continue  # already applied

        # Import the module dynamically
        mod = importlib.import_module(
            f"orchestra.storage.migrations.{module_name}"
        )

        try:
            result = mod.upgrade(conn)  # upgrade() manages its own BEGIN IMMEDIATE
            _set_schema_version(conn, version_after)
            migrations_run += 1
            details.append({"module": module_name, "result": result})
            current = version_after
        except Exception as e:
            raise MigrationError(
                f"Migration {module_name} failed: {e}"
            ) from e

    return {
        "migrations_run": migrations_run,
        "final_version": _get_schema_version(conn),
        "errors": [],
        "migration_details": details,
    }


def rollback_migration(conn: sqlite3.Connection, from_version: int) -> dict:
    """Roll back one migration (from_version → from_version - 1)."""
    if from_version <= 0:
        return {"rollback_completed": False, "new_version": 0}

    modules = _get_migration_modules()
    module_name = modules[from_version - 1]
    mod = importlib.import_module(f"orchestra.storage.migrations.{module_name}")
    mod.downgrade(conn)
    _set_schema_version(conn, from_version - 1)
    return {"rollback_completed": True, "new_version": from_version - 1}


def get_migration_status(conn: sqlite3.Connection) -> dict:
    """Return current schema version and count of pending migrations."""
    current = _get_schema_version(conn)
    modules = _get_migration_modules()
    pending = len([m for i, m in enumerate(modules) if (i + 1) > current])
    return {
        "current_version": current,
        "pending_count": pending,
        "available_modules": modules,
    }



def migrate_v2_to_v3(conn: sqlite3.Connection) -> bool:
    """
    Migrate EventLog from v2 (optional seq) to v3 (mandatory seq with AUTOINCREMENT).

    Changes:
    - seq becomes INTEGER UNIQUE NOT NULL
    - Populate seq via ROW_NUMBER() based on ts order
    - Create new indexes for seq-based queries
    - Add created_at timestamp field

    Args:
        conn: SQLite connection

    Returns:
        True if migration succeeded

    Raises:
        MigrationError: On migration failure
    """
    cursor = conn.cursor()

    cursor.execute("BEGIN IMMEDIATE")
    try:
        # Check if events table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            # Table doesn't exist; schema initialization will handle it
            logger.info("Events table does not exist; skipping v2→v3 migration")
            conn.commit()
            return True

        # Check if seq is already NOT NULL (already v3)
        cursor.execute("PRAGMA table_info(events)")
        seq_info = cursor.fetchall()
        seq_not_null = any(col[1] == "seq" and col[3] == 1 for col in seq_info)
        if seq_not_null:
            logger.info("Events table already at v3 schema; skipping migration")
            conn.commit()
            return True

        logger.info("Starting v2→v3 migration for events table")

        # Create new events_v3 table with strict schema
        cursor.execute("""
            CREATE TABLE events_v3 (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                run_id TEXT NOT NULL,
                payload TEXT,
                ts REAL NOT NULL,
                seq INTEGER UNIQUE NOT NULL,
                created_at REAL DEFAULT (unixepoch('subsec'))
            )
        """)

        # Migrate data: assign seq via ROW_NUMBER() ordered by ts
        cursor.execute("""
            INSERT INTO events_v3 (id, event_type, run_id, payload, ts, seq)
            SELECT
                id,
                event_type,
                run_id,
                payload,
                ts,
                ROW_NUMBER() OVER (ORDER BY ts ASC) as seq
            FROM events
            ORDER BY ts ASC
        """)

        # Drop old table
        cursor.execute("DROP TABLE events")

        # Rename new table
        cursor.execute("ALTER TABLE events_v3 RENAME TO events")

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id_seq ON events(run_id, seq)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_seq ON events(seq)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type)")

        conn.commit()
        logger.info("v2→v3 migration completed successfully")
        return True

    except sqlite3.Error as e:
        conn.rollback()
        raise MigrationError(f"v2→v3 migration failed: {str(e)}") from e
