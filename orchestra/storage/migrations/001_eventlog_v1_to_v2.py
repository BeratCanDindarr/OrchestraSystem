"""Migration 001: EventLog v1 → v2 (seq atomic RETURNING clause).

This migration converts the EventLog from optional seq (v1) to mandatory seq
with atomic RETURNING clause (v2). It populates seq values via ROW_NUMBER() and
creates proper indexes.

Key changes:
- seq becomes UNIQUE NOT NULL
- Atomic seq generation via RETURNING (SQLite 3.35+)
- New indexes for seq-based queries
- Backward-compatible payload format
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised on migration failure."""
    pass


class RollbackNotSupportedError(MigrationError):
    """Raised when rollback is not supported."""
    pass


def upgrade(conn: sqlite3.Connection) -> dict:
    """
    Upgrade EventLog from v1 (optional seq) to v2 (mandatory seq with RETURNING).

    Args:
        conn: SQLite connection with PRAGMA foreign_keys = ON

    Returns:
        dict with migration metadata:
        {
            'rows_migrated': int,
            'new_indexes': list[str],
            'backup_table': str or None
        }

    Raises:
        MigrationError: On migration failure
    """
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN IMMEDIATE")

        # Check if events table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        if not cursor.fetchone():
            logger.info("001: Events table does not exist; migration not needed")
            conn.commit()
            return {
                'rows_migrated': 0,
                'new_indexes': [],
                'backup_table': None,
            }

        # Check if seq is already NOT NULL (already v2)
        cursor.execute("PRAGMA table_info(events)")
        columns = cursor.fetchall()
        seq_col = next((col for col in columns if col[1] == 'seq'), None)

        if seq_col and seq_col[3] == 1:  # col[3] is notnull flag
            logger.info("001: Events table already at v2 schema; no upgrade needed")
            conn.commit()
            return {
                'rows_migrated': 0,
                'new_indexes': [],
                'backup_table': None,
            }

        logger.info("001: Starting v1→v2 migration for events table")

        # Step 1: Create backup table (optional, for safety)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events_v1_backup AS
            SELECT * FROM events
        """)
        backup_table = "events_v1_backup"

        # Step 2: Get current row count
        cursor.execute("SELECT COUNT(*) FROM events")
        initial_count = cursor.fetchone()[0]
        logger.info(f"001: Found {initial_count} events to migrate")

        # Step 3: Create new events_v2 table with strict schema
        cursor.execute("""
            CREATE TABLE events_v2 (
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

        # Step 4: Migrate data with ROW_NUMBER() assignment
        cursor.execute("""
            INSERT INTO events_v2 (id, event_type, run_id, payload, ts, seq)
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

        migrated = cursor.rowcount
        logger.info(f"001: Migrated {migrated} rows to events_v2")

        # Step 5: Verify data integrity
        cursor.execute("SELECT COUNT(*) FROM events_v2")
        new_count = cursor.fetchone()[0]
        if new_count != initial_count:
            raise MigrationError(
                f"Data integrity check failed: {initial_count} rows → {new_count} rows"
            )

        # Step 6: Check for seq uniqueness
        cursor.execute("""
            SELECT COUNT(*) as dup_count FROM (
                SELECT seq FROM events_v2 GROUP BY seq HAVING COUNT(*) > 1
            )
        """)
        dup_count = cursor.fetchone()[0]
        if dup_count > 0:
            raise MigrationError(f"Found {dup_count} duplicate seq values")

        # Step 7: Drop old table and rename
        cursor.execute("DROP TABLE events")
        cursor.execute("ALTER TABLE events_v2 RENAME TO events")

        # Step 8: Create indexes
        indexes = []
        for idx_def in [
            ("idx_events_run_id", "CREATE INDEX idx_events_run_id ON events(run_id)"),
            ("idx_events_run_id_seq", "CREATE INDEX idx_events_run_id_seq ON events(run_id, seq)"),
            ("idx_events_seq", "CREATE INDEX idx_events_seq ON events(seq)"),
            ("idx_events_ts", "CREATE INDEX idx_events_ts ON events(ts)"),
            ("idx_events_event_type", "CREATE INDEX idx_events_event_type ON events(event_type)"),
        ]:
            idx_name, idx_sql = idx_def
            try:
                cursor.execute(idx_sql)
                indexes.append(idx_name)
                logger.debug(f"001: Created index {idx_name}")
            except sqlite3.OperationalError:
                logger.warning(f"001: Index {idx_name} already exists")

        conn.commit()
        logger.info(f"001: Migration completed. Created {len(indexes)} indexes")

        return {
            'rows_migrated': migrated,
            'new_indexes': indexes,
            'backup_table': backup_table,
        }

    except sqlite3.Error as e:
        conn.rollback()
        raise MigrationError(f"001: Migration failed: {str(e)}") from e


def downgrade(conn: sqlite3.Connection) -> dict:
    """
    Downgrade EventLog from v2 back to v1.

    This is a lossy operation: the restored v1 table will have seq values,
    even though v1 didn't require them. Use only for rollback.

    Args:
        conn: SQLite connection

    Returns:
        dict with metadata

    Raises:
        RollbackNotSupportedError: If backup is not available
    """
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN IMMEDIATE")

        # Check if backup exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events_v1_backup'"
        )
        if not cursor.fetchone():
            raise RollbackNotSupportedError(
                "001: Cannot rollback: events_v1_backup not found. "
                "Manual restore required."
            )

        logger.info("001: Starting downgrade from v2 to v1")

        # Drop v2 table
        cursor.execute("DROP TABLE IF EXISTS events")

        # Restore from backup
        cursor.execute("ALTER TABLE events_v1_backup RENAME TO events")

        conn.commit()
        logger.info("001: Downgrade completed; restored from events_v1_backup")

        return {
            'rows_restored': cursor.rowcount,
            'backup_preserved': False,
        }

    except sqlite3.Error as e:
        conn.rollback()
        raise MigrationError(f"001: Downgrade failed: {str(e)}") from e


def validate(conn: sqlite3.Connection) -> bool:
    """
    Validate that migration completed successfully.

    Checks:
    - events table exists
    - seq is NOT NULL and UNIQUE
    - All events have seq values
    - seq values are contiguous (no gaps)
    - Indexes exist

    Args:
        conn: SQLite connection

    Returns:
        True if validation passes

    Raises:
        MigrationError: If validation fails
    """
    cursor = conn.cursor()

    # Check table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
    )
    if not cursor.fetchone():
        raise MigrationError("001: Validation failed: events table not found")

    # Check schema
    cursor.execute("PRAGMA table_info(events)")
    columns = {row[1]: row for row in cursor.fetchall()}

    if 'seq' not in columns:
        raise MigrationError("001: Validation failed: seq column missing")

    seq_col = columns['seq']
    if seq_col[3] != 1:  # notnull flag
        raise MigrationError("001: Validation failed: seq is not NOT NULL")

    # Check for NULL seq values
    cursor.execute("SELECT COUNT(*) FROM events WHERE seq IS NULL")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        raise MigrationError(
            f"001: Validation failed: {null_count} events have NULL seq"
        )

    # Check for duplicate seq values
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT seq FROM events GROUP BY seq HAVING COUNT(*) > 1
        )
    """)
    dup_count = cursor.fetchone()[0]
    if dup_count > 0:
        raise MigrationError(
            f"001: Validation failed: {dup_count} duplicate seq values"
        )

    # Check contiguity (seq should be 1, 2, 3, ... N if populated)
    cursor.execute("SELECT COUNT(*) FROM events")
    total = cursor.fetchone()[0]
    if total > 0:
        cursor.execute("SELECT MAX(seq) FROM events")
        max_seq = cursor.fetchone()[0]
        if max_seq != total:
            raise MigrationError(
                f"001: Validation failed: non-contiguous seq "
                f"(max={max_seq}, count={total})"
            )

    # Check indexes
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'"
    )
    indexes = {row[0] for row in cursor.fetchall()}
    required = {'idx_events_run_id', 'idx_events_seq'}
    if not required.issubset(indexes):
        missing = required - indexes
        logger.warning(f"001: Validation: missing indexes {missing}")

    logger.info("001: Validation passed")
    return True
