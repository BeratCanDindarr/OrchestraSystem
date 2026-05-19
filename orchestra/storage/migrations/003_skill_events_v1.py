"""Add skill_events table for immutable skill lifecycle audit trail."""
from __future__ import annotations

import sqlite3


def upgrade(conn: sqlite3.Connection) -> dict:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            actor TEXT DEFAULT 'orchestra',
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_events_name ON skill_events(skill_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_events_type ON skill_events(event_type)")
    return {"table": "skill_events", "indexes": 2}


def downgrade(conn: sqlite3.Connection) -> dict:
    conn.execute("DROP TABLE IF EXISTS skill_events")
    return {"table_dropped": "skill_events"}


def validate(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='skill_events'"
    ).fetchone()
    return row is not None
