"""Tests for EvalTracker score persistence + regression delta (Faz 4 Block C/D)."""
import sqlite3

from orchestra.storage.migrations import ensure_schema
from orchestra.storage.eval_tracker import EvalTracker


def _fresh_db(tmp_path):
    dbfile = str(tmp_path / "eval.db")
    conn = sqlite3.connect(dbfile)
    ensure_schema(conn)
    conn.commit()
    conn.close()
    return dbfile


_SCORES = {
    "answer_quality": 0.8, "factual_grounding": 0.7, "source_attribution": 1.0,
    "conciseness": 0.9, "calibration": 0.6, "composite_score": 0.8,
}


def test_log_score_persists_row(tmp_path, monkeypatch):
    dbfile = _fresh_db(tmp_path)
    monkeypatch.setattr("orchestra.storage.eval_tracker.get_db", lambda: sqlite3.connect(dbfile))
    EvalTracker().log_score("r1", "h1", _SCORES)
    conn = sqlite3.connect(dbfile)
    row = conn.execute(
        "SELECT task_hash, composite_score, calibration FROM eval_results WHERE run_id='r1'"
    ).fetchone()
    conn.close()
    assert row == ("h1", 0.8, 0.6)


def test_regression_delta_compares_latest_vs_history(tmp_path, monkeypatch):
    dbfile = _fresh_db(tmp_path)
    monkeypatch.setattr("orchestra.storage.eval_tracker.get_db", lambda: sqlite3.connect(dbfile))
    tracker = EvalTracker()
    for composite in (0.6, 0.8, 0.4):  # insertion order = chronological (id ascending)
        tracker.log_score("r", "h1", {**_SCORES, "composite_score": composite})
    delta = tracker.regression_delta("h1")
    assert delta["n"] == 3
    assert delta["latest"] == 0.4
    assert abs(delta["historical_mean"] - 0.7) < 1e-9
    assert abs(delta["delta"] - (-0.3)) < 1e-9


def test_regression_delta_insufficient_history(tmp_path, monkeypatch):
    dbfile = _fresh_db(tmp_path)
    monkeypatch.setattr("orchestra.storage.eval_tracker.get_db", lambda: sqlite3.connect(dbfile))
    tracker = EvalTracker()
    tracker.log_score("r", "h1", _SCORES)
    delta = tracker.regression_delta("h1")
    assert delta["n"] == 1
    assert delta["delta"] == 0.0
