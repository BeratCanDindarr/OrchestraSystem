"""Unit tests for web dashboard incident counts (count_incidents over the events log)."""
from __future__ import annotations

import sqlite3

from orchestra.monitoring import web_dashboard as wd


def _events_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE events (event_type TEXT, run_id TEXT)")
    conn.executemany("INSERT INTO events VALUES (?, ?)", rows)
    conn.commit()
    return conn


def test_count_incidents_counts_only_incident_types():
    conn = _events_conn([
        ("run_started", "r1"),
        ("agent_failed", "r1"),
        ("agent_failed", "r2"),
        ("budget_exceeded", "r3"),
        ("run_completed", "r4"),      # not an incident
        ("validation_failed", "r5"),
    ])
    out = wd.count_incidents(conn)
    assert out == {"agent_failed": 2, "budget_exceeded": 1, "validation_failed": 1}
    assert "run_started" not in out and "run_completed" not in out


def test_count_incidents_empty_when_none():
    conn = _events_conn([("run_started", "r1"), ("run_completed", "r1")])
    assert wd.count_incidents(conn) == {}


def test_count_incidents_handles_missing_table():
    conn = sqlite3.connect(":memory:")   # no events table
    assert wd.count_incidents(conn) == {}
