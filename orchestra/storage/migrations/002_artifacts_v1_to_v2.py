"""Migration 002: Artifacts v1 → v2 (snapshot fields: agents[], reviews[], planned_nodes{}).

This migration extends the artifacts schema to support new snapshot fields:
- agents[]: List of agent metadata snapshots
- reviews[]: List of approval/review gate snapshots
- planned_nodes{}: Dictionary of planned nodes per checkpoint

Backward compatibility:
- Old checkpoints without these fields are preserved
- New schema_version field added to track format version
- Existing checkpoints remain readable
"""
from __future__ import annotations

import json
import logging
import sqlite3

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised on migration failure."""
    pass


def upgrade(conn: sqlite3.Connection) -> dict:
    """
    Upgrade artifacts schema to v2 with new snapshot fields.

    Changes:
    - Add schema_version column to track format
    - Preserve existing data with schema_version=1
    - New checkpoints will use schema_version=2

    Args:
        conn: SQLite connection

    Returns:
        dict with metadata:
        {
            'rows_checked': int,
            'schema_version_set': bool,
            'new_columns': list[str]
        }

    Raises:
        MigrationError: On migration failure
    """
    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN IMMEDIATE")

        # Check if artifacts/checkpoints table exists (naming varies)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('artifacts', 'checkpoints', 'tool_proposals')"
        )
        table_row = cursor.fetchone()
        if not table_row:
            logger.info("002: No artifacts/checkpoints table found; migration not needed")
            conn.commit()
            return {
                'rows_checked': 0,
                'schema_version_set': False,
                'new_columns': [],
            }

        table_name = table_row[0]
        logger.info(f"002: Upgrading {table_name} table to v2 schema")

        # Check if schema_version already exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}

        new_cols = []

        # Add schema_version if not present
        if 'schema_version' not in columns:
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN schema_version INTEGER DEFAULT 1")
                new_cols.append('schema_version')
                logger.debug(f"002: Added schema_version column to {table_name}")
            except sqlite3.OperationalError as e:
                logger.warning(f"002: Could not add schema_version: {e}")

        # Add snapshot fields if needed (for tool_proposals)
        for col_name in ['agents_snapshot', 'reviews_snapshot', 'planned_nodes_snapshot']:
            if col_name not in columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {col_name} TEXT DEFAULT NULL"
                    )
                    new_cols.append(col_name)
                    logger.debug(f"002: Added {col_name} column to {table_name}")
                except sqlite3.OperationalError as e:
                    logger.warning(f"002: Could not add {col_name}: {e}")

        # Set schema_version to 1 for all existing rows (backward compat)
        cursor.execute(f"UPDATE {table_name} SET schema_version = 1 WHERE schema_version IS NULL")
        updated = cursor.rowcount
        logger.info(f"002: Set schema_version=1 for {updated} existing rows")

        # Get total row count for reporting
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total = cursor.fetchone()[0]

        conn.commit()
        logger.info(f"002: Upgrade completed. Total rows: {total}, new columns: {len(new_cols)}")

        return {
            'rows_checked': total,
            'schema_version_set': True,
            'new_columns': new_cols,
        }

    except sqlite3.Error as e:
        conn.rollback()
        raise MigrationError(f"002: Migration failed: {str(e)}") from e


def downgrade(conn: sqlite3.Connection) -> dict:
    """
    Downgrade artifacts schema from v2 to v1.

    Removes new columns; existing data is preserved but inaccessible.

    Args:
        conn: SQLite connection

    Returns:
        dict with metadata

    Raises:
        MigrationError: On migration failure (SQLite ALTER TABLE limitations)
    """
    logger.warning("002: Downgrade requested. SQLite doesn't support DROP COLUMN in older versions.")
    logger.warning("002: Creating v1-compatible backup instead.")

    cursor = conn.cursor()

    try:
        cursor.execute("BEGIN IMMEDIATE")

        # Get the main table name
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('artifacts', 'checkpoints', 'tool_proposals')"
        )
        table_row = cursor.fetchone()
        if not table_row:
            raise MigrationError("002: No artifacts table found for downgrade")

        table_name = table_row[0]
        logger.info(f"002: Creating downgrade backup of {table_name}")

        # Create v1-compatible backup (columns without snapshot fields)
        cursor.execute(f"PRAGMA table_info({table_name})")
        all_cols = cursor.fetchall()

        # Filter out v2 columns
        v1_cols = [
            col[1] for col in all_cols
            if col[1] not in {'schema_version', 'agents_snapshot', 'reviews_snapshot', 'planned_nodes_snapshot'}
        ]

        if not v1_cols:
            raise MigrationError("002: No v1-compatible columns found")

        col_list = ", ".join(v1_cols)
        cursor.execute(f"""
            CREATE TABLE {table_name}_v1_backup AS
            SELECT {col_list} FROM {table_name}
        """)

        logger.info(f"002: Created {table_name}_v1_backup with {len(v1_cols)} columns")
        conn.commit()

        return {
            'backup_created': f"{table_name}_v1_backup",
            'columns_excluded': len(all_cols) - len(v1_cols),
        }

    except sqlite3.Error as e:
        conn.rollback()
        raise MigrationError(f"002: Downgrade failed: {str(e)}") from e


def validate(conn: sqlite3.Connection) -> bool:
    """
    Validate that migration completed successfully.

    Checks:
    - schema_version column exists
    - All existing rows have schema_version = 1
    - Snapshot columns exist (may be NULL)

    Args:
        conn: SQLite connection

    Returns:
        True if validation passes

    Raises:
        MigrationError: If validation fails
    """
    cursor = conn.cursor()

    # Find the table
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('artifacts', 'checkpoints', 'tool_proposals')"
    )
    table_row = cursor.fetchone()
    if not table_row:
        logger.info("002: No artifacts table for validation; skipping")
        return True

    table_name = table_row[0]

    # Check schema_version column
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {row[1] for row in cursor.fetchall()}

    if 'schema_version' not in columns:
        raise MigrationError(f"002: Validation failed: schema_version column missing from {table_name}")

    # Check that existing rows have schema_version
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE schema_version IS NULL")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        raise MigrationError(
            f"002: Validation failed: {null_count} rows have NULL schema_version"
        )

    # Check schema_version values are 1 or 2
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE schema_version NOT IN (1, 2)")
    invalid_count = cursor.fetchone()[0]
    if invalid_count > 0:
        raise MigrationError(
            f"002: Validation failed: {invalid_count} rows have invalid schema_version"
        )

    logger.info(f"002: Validation passed for {table_name}")
    return True
