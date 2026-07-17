"""Unit tests for web dashboard metrics aggregation (compute_metrics)."""
from __future__ import annotations

import sqlite3

from orchestra.monitoring import web_dashboard as wd


def _runs_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE runs (run_id TEXT, mode TEXT, status TEXT, total_cost_usd REAL)")
    conn.executemany("INSERT INTO runs VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return conn


def test_compute_metrics_empty():
    conn = _runs_conn([])
    m = wd.compute_metrics(conn)
    assert m["total_runs"] == 0
    assert m["avg_cost_usd"] == 0.0 and m["failure_rate"] == 0.0


def test_compute_metrics_aggregates_counts_cost_and_failure_rate():
    conn = _runs_conn([
        ("r1", "ask", "completed", 0.01),
        ("r2", "dual", "completed", 0.03),
        ("r3", "ask", "failed", 0.02),
        ("r4", "critical", "failed", None),   # null cost tolerated
    ])
    m = wd.compute_metrics(conn)
    assert m["total_runs"] == 4
    assert m["by_status"] == {"completed": 2, "failed": 2}
    assert m["by_mode"] == {"ask": 2, "dual": 1, "critical": 1}
    assert m["total_cost_usd"] == 0.06
    assert m["avg_cost_usd"] == 0.015
    assert m["failure_rate"] == 0.5


def test_compute_metrics_handles_missing_table():
    conn = sqlite3.connect(":memory:")   # no runs table
    m = wd.compute_metrics(conn)
    assert m["total_runs"] == 0          # zeroed summary, no raise


def test_dashboard_html_renders_metrics_panel():
    html = wd._DASHBOARD_HTML
    assert 'id="metrics"' in html                 # panel container present
    assert "/api/metrics" in html                 # fetched from the metrics endpoint
    assert "failure rate" in html and "total cost" in html
