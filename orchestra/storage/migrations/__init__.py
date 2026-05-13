"""Migration runner: auto-discovers and executes versioned migrations.

Migrations follow the pattern: NNN_name_v1_to_v2.py
Each migration module must provide:
- upgrade(conn) -> dict: Upgrade function returning metadata
- downgrade(conn) -> dict: Downgrade function for rollback
- validate(conn) -> bool: Validation function

The runner:
1. Discovers all migration modules in sorted order
2. Tracks applied migrations via schema_version PRAGMA
3. Executes only unapplied migrations
4. Validates after each migration
5. Provides rollback on failure
"""
from __future__ import annotations

import importlib
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Re-export legacy schema initialization for backward compatibility
def ensure_schema(connection: sqlite3.Connection) -> None:
    """Legacy schema initialization (v0)."""
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
    """

    COLUMNS = {
        "runs": {
            "mode": "TEXT", "status": "TEXT", "task": "TEXT", "created_at": "TEXT", "updated_at": "TEXT",
            "total_cost_usd": "REAL DEFAULT 0.0", "avg_confidence": "REAL DEFAULT 0.0",
            "latest_review_stage": "TEXT DEFAULT ''", "latest_review_status": "TEXT DEFAULT 'not_run'",
            "latest_review_winner": "TEXT DEFAULT ''", "latest_review_reason": "TEXT DEFAULT ''",
            "approval_state": "TEXT DEFAULT 'not_required'", "interrupt_state": "TEXT DEFAULT 'idle'",
            "checkpoint_version": "INTEGER DEFAULT 0", "last_seq_id": "INTEGER DEFAULT 0",
            "failure_kind": "TEXT", "failure_message": "TEXT",
            "dissent_text": "TEXT DEFAULT ''"
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
            "causation_id": "TEXT", "idempotency_key": "TEXT", "ts": "TEXT", "data": "TEXT",
            "id": "TEXT UNIQUE", "payload": "TEXT"
        },
        "blackboard": {
            "run_id": "TEXT", "alias": "TEXT", "content": "TEXT", "tags": "TEXT", "ts": "TEXT"
        },
        "alias_reputation": {
            "provider": "TEXT", "total_runs": "INTEGER DEFAULT 0",
            "completed_runs": "INTEGER DEFAULT 0", "failed_runs": "INTEGER DEFAULT 0",
            "cancelled_runs": "INTEGER DEFAULT 0", "soft_failures": "INTEGER DEFAULT 0",
            "validation_failures": "INTEGER DEFAULT 0", "review_wins": "INTEGER DEFAULT 0",
            "avg_confidence": "REAL DEFAULT 0.0", "avg_cost_usd": "REAL DEFAULT 0.0",
            "reputation_score": "REAL DEFAULT 50.0", "last_run_at": "TEXT", "updated_at": "TEXT",
            "outcome_wins": "INTEGER DEFAULT 0", "outcome_losses": "INTEGER DEFAULT 0",
            "outcome_soft_fails": "INTEGER DEFAULT 0", "outcome_ties": "INTEGER DEFAULT 0",
        },
        "paused_runs": {
            "checkpoint_data": "TEXT NOT NULL", "paused_at": "TEXT NOT NULL",
            "paused_by": "TEXT DEFAULT 'system'", "expires_at": "TEXT NOT NULL",
            "resume_attempt_count": "INTEGER DEFAULT 0", "last_resume_error": "TEXT"
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
            "created_at": "TEXT"
        }
    }

    def _ensure_robust_columns(conn: sqlite3.Connection):
        for table, cols in COLUMNS.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, spec in cols.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")

    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    _ensure_robust_columns(connection)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_blackboard_run_id ON blackboard(run_id)")


class MigrationError(Exception):
    """Raised on migration runner failure."""
    pass


def _get_migration_modules() -> list[str]:
    """
    Auto-discover migration modules in this directory.

    Returns:
        List of module names (e.g., ['001_eventlog_v1_to_v2', '002_artifacts_v1_to_v2'])
    """
    migrations_dir = Path(__file__).parent
    modules = []

    for py_file in sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.py")):
        module_name = py_file.stem
        modules.append(module_name)

    logger.info(f"Discovered {len(modules)} migration modules: {modules}")
    return modules


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """
    Get current schema version from PRAGMA user_version.

    Args:
        conn: SQLite connection

    Returns:
        Schema version (0 if not set)
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA user_version")
    return cursor.fetchone()[0]


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """
    Set schema version via PRAGMA user_version.

    Args:
        conn: SQLite connection
        version: New schema version
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def run_migrations(conn: sqlite3.Connection, target_version: Optional[int] = None) -> dict:
    """
    Run all pending migrations up to target_version.

    Flow:
    1. Auto-discover migration modules
    2. Determine current schema version
    3. Execute each migration in order
    4. Validate each migration
    5. Update schema version
    6. Rollback on error

    Args:
        conn: SQLite connection
        target_version: Optional target version (default: latest available)

    Returns:
        dict with migration results:
        {
            'migrations_run': int,
            'migration_details': list[dict],
            'final_version': int,
            'errors': list[str] or empty
        }

    Raises:
        MigrationError: On migration failure with rollback
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    modules = _get_migration_modules()
    current_version = _get_schema_version(conn)

    if target_version is None:
        target_version = len(modules)

    logger.info(
        f"Starting migration runner: current_version={current_version}, "
        f"target_version={target_version}, available_migrations={len(modules)}"
    )

    if current_version >= target_version:
        logger.info(f"Database already at v{current_version}; no migrations needed")
        return {
            'migrations_run': 0,
            'migration_details': [],
            'final_version': current_version,
            'errors': [],
        }

    migration_details = []
    errors = []

    try:
        # Execute migrations
        for idx, module_name in enumerate(modules, start=1):
            if idx <= current_version:
                logger.debug(f"Skipping {module_name} (already applied)")
                continue

            if idx > target_version:
                logger.debug(f"Stopping at {module_name} (reached target_version={target_version})")
                break

            logger.info(f"Applying migration {idx}: {module_name}")

            try:
                # Import migration module
                migration = importlib.import_module(
                    f"orchestra.storage.migrations.{module_name}"
                )

                # Run upgrade
                result = migration.upgrade(conn)
                logger.info(f"Migration {module_name} upgrade completed: {result}")

                # Validate
                migration.validate(conn)
                logger.info(f"Migration {module_name} validation passed")

                migration_details.append({
                    'module': module_name,
                    'version': idx,
                    'status': 'applied',
                    'result': result,
                })

                # Update schema version after each successful migration
                _set_schema_version(conn, idx)
                logger.info(f"Schema version updated to {idx}")

            except Exception as e:
                error_msg = f"Migration {module_name} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                raise MigrationError(error_msg) from e

        # All migrations completed
        final_version = _get_schema_version(conn)
        logger.info(f"Migration runner completed. Final version: {final_version}")

        return {
            'migrations_run': len(migration_details),
            'migration_details': migration_details,
            'final_version': final_version,
            'errors': errors,
        }

    except MigrationError as e:
        # Rollback on any migration error
        conn.rollback()
        logger.error(f"Migration failed and rolled back: {e}")
        raise


def rollback_migration(conn: sqlite3.Connection, from_version: int) -> dict:
    """
    Rollback from a specific version.

    Args:
        conn: SQLite connection
        from_version: Version to rollback from

    Returns:
        dict with rollback results

    Raises:
        MigrationError: If rollback fails
    """
    if from_version <= 0:
        logger.info("Already at version 0; no rollback needed")
        return {
            'rollback_completed': False,
            'reason': 'Already at version 0',
        }

    modules = _get_migration_modules()
    if from_version > len(modules):
        raise MigrationError(f"Cannot rollback from v{from_version}: only {len(modules)} migrations available")

    module_name = modules[from_version - 1]
    logger.info(f"Rolling back migration {from_version}: {module_name}")

    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        migration = importlib.import_module(
            f"orchestra.storage.migrations.{module_name}"
        )

        # Run downgrade
        result = migration.downgrade(conn)
        logger.info(f"Migration {module_name} downgrade completed: {result}")

        # Update schema version
        _set_schema_version(conn, from_version - 1)
        logger.info(f"Schema version rolled back to {from_version - 1}")

        return {
            'rollback_completed': True,
            'module': module_name,
            'new_version': from_version - 1,
            'result': result,
        }

    except Exception as e:
        conn.rollback()
        raise MigrationError(f"Rollback of migration {module_name} failed: {str(e)}") from e


def get_migration_status(conn: sqlite3.Connection) -> dict:
    """
    Get current migration status and available migrations.

    Args:
        conn: SQLite connection

    Returns:
        dict with status:
        {
            'current_version': int,
            'available_migrations': list[str],
            'pending_count': int
        }
    """
    current_version = _get_schema_version(conn)
    modules = _get_migration_modules()

    return {
        'current_version': current_version,
        'available_migrations': modules,
        'pending_count': max(0, len(modules) - current_version),
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
                payload TEXT NOT NULL,
                ts REAL NOT NULL,
                seq INTEGER UNIQUE NOT NULL,
                created_at REAL DEFAULT (unixepoch('subsec')),
                CHECK (event_type IN (
                    'run_started','run_completed','run_failed',
                    'agent_started','agent_completed','agent_failed','agent_retrying',
                    'budget_check','budget_exceeded',
                    'approval_requested','approval_approved','approval_rejected',
                    'suspension_created','suspension_resumed',
                    'checkpoint_written','checkpoint_loaded',
                    'synthesis_started','synthesis_completed',
                    'cache_hit','cache_miss',
                    'error_logged','validation_failed',
                    'span_started','span_completed',
                    'cost_tracked','token_count_recorded',
                    'ext_1','ext_2','ext_3','ext_4'
                ))
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
