"""Unit tests for web dashboard eval summary (compute_eval_summary over eval_runs)."""
from __future__ import annotations

import sqlite3

from orchestra.monitoring import web_dashboard as wd


def _eval_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE eval_runs (run_id TEXT, task TEXT, mode TEXT, status TEXT, created_at TEXT)")
    conn.executemany(
        "INSERT INTO eval_runs VALUES (?, ?, ?, ?, ?)",
        [(rid, "t", mode, status, "2026-07-17") for rid, mode, status in rows],
    )
    conn.commit()
    return conn


def test_eval_summary_empty():
    conn = _eval_conn([])
    s = wd.compute_eval_summary(conn)
    assert s == {"total": 0, "passed": 0, "pass_rate": 0.0, "by_mode": {}}


def test_eval_summary_overall_and_by_mode():
    conn = _eval_conn([
        ("r1", "ask", "PASS"),
        ("r2", "ask", "FAIL"),
        ("r3", "dual", "PASS"),
        ("r4", "dual", "PASS"),
    ])
    s = wd.compute_eval_summary(conn)
    assert s["total"] == 4 and s["passed"] == 3
    assert s["pass_rate"] == 0.75
    assert s["by_mode"]["ask"] == {"passed": 1, "total": 2, "pass_rate": 0.5}
    assert s["by_mode"]["dual"] == {"passed": 2, "total": 2, "pass_rate": 1.0}


def test_eval_summary_missing_table():
    conn = sqlite3.connect(":memory:")   # no eval_runs table
    assert wd.compute_eval_summary(conn)["total"] == 0


def test_dashboard_html_renders_eval_panel():
    html = wd._DASHBOARD_HTML
    assert 'id="eval"' in html
    assert "/api/eval" in html
    assert "pass rate" in html and "no evals yet" in html
