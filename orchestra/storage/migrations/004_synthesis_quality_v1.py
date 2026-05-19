"""Add synthesis_quality table for tracking synthesizer output quality."""
from __future__ import annotations

import sqlite3


def upgrade(conn: sqlite3.Connection) -> dict:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS synthesis_quality (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            score REAL NOT NULL,
            length INTEGER NOT NULL,
            has_answer INTEGER DEFAULT 0,
            has_key_signals INTEGER DEFAULT 0,
            has_dissent INTEGER DEFAULT 0,
            has_failure_report INTEGER DEFAULT 0,
            has_provenance INTEGER DEFAULT 0,
            divergence_index REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synthesis_quality_run_id ON synthesis_quality(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_synthesis_quality_score ON synthesis_quality(score)")
    return {"table": "synthesis_quality", "indexes": 2}


def downgrade(conn: sqlite3.Connection) -> dict:
    conn.execute("DROP TABLE IF EXISTS synthesis_quality")
    return {"table_dropped": "synthesis_quality"}


def validate(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='synthesis_quality'"
    ).fetchone()
    return row is not None
