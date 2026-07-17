"""Unit tests for web dashboard HITL logic (list paused runs, apply approval)."""
from __future__ import annotations

import sqlite3

from orchestra import config
from orchestra.monitoring import web_dashboard as wd


def _paused_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE paused_runs (run_id TEXT PRIMARY KEY, paused_at TEXT, "
        "paused_by TEXT, status TEXT)"
    )
    conn.execute("INSERT INTO paused_runs VALUES ('r1', '2026-07-17T00:00:00', 'approval_gate', 'pending')")
    conn.execute("INSERT INTO paused_runs VALUES ('r2', '2026-07-17T00:01:00', 'approval_gate', 'resuming')")
    conn.commit()
    return conn


def test_list_paused_runs_returns_only_pending():
    rows = wd.list_paused_runs(_paused_conn())
    assert [r["run_id"] for r in rows] == ["r1"]     # r2 is 'resuming', not pending
    assert rows[0]["paused_by"] == "approval_gate"


def test_list_paused_runs_empty_on_missing_table():
    conn = sqlite3.connect(":memory:")   # no paused_runs table
    assert wd.list_paused_runs(conn) == []


def test_apply_approval_rejects_bad_decision():
    called = []
    out = wd.apply_approval("r1", "maybe", resume_fn=lambda rid, d: called.append((rid, d)))
    assert out["ok"] is False and called == []


def test_apply_approval_requires_run_id():
    out = wd.apply_approval("", "approve", resume_fn=lambda rid, d: None)
    assert out["ok"] is False


def test_apply_approval_resumes_on_valid_decision():
    called = []
    out = wd.apply_approval("r1", "approve", resume_fn=lambda rid, d: called.append((rid, d)))
    assert out == {"ok": True, "run_id": "r1", "decision": "approve"}
    assert called == [("r1", "approve")]


def test_apply_approval_reports_resume_failure():
    def boom(rid, d):
        raise RuntimeError("resume exploded")

    out = wd.apply_approval("r1", "reject", resume_fn=boom)
    assert out["ok"] is False and "resume exploded" in out["error"]


def test_dashboard_approvals_disabled_by_default():
    assert config.dashboard_approvals_enabled() is False
