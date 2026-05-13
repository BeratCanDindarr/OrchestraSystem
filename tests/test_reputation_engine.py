"""Tests for Reputation Engine — record_outcome, get_reputation_delta, reviewer integration."""
from __future__ import annotations

import sqlite3

import pytest

from orchestra.storage.migrations import ensure_schema
from orchestra.storage.reputation import get_reputation_delta, record_outcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    """In-memory SQLite with full orchestra schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# record_outcome — basic write path
# ---------------------------------------------------------------------------

def test_record_win_creates_row(db):
    record_outcome(db, "cdx-deep", "win")
    row = db.execute("SELECT outcome_wins FROM alias_reputation WHERE alias='cdx-deep'").fetchone()
    assert row is not None
    assert int(row["outcome_wins"]) == 1


def test_record_loss_creates_row(db):
    record_outcome(db, "gmn-pro", "loss")
    row = db.execute("SELECT outcome_losses FROM alias_reputation WHERE alias='gmn-pro'").fetchone()
    assert int(row["outcome_losses"]) == 1


def test_record_soft_fail_creates_row(db):
    record_outcome(db, "cld-fast", "soft_fail")
    row = db.execute("SELECT outcome_soft_fails FROM alias_reputation WHERE alias='cld-fast'").fetchone()
    assert int(row["outcome_soft_fails"]) == 1


def test_record_tie_creates_row(db):
    record_outcome(db, "cdx-deep", "tie")
    row = db.execute("SELECT outcome_ties FROM alias_reputation WHERE alias='cdx-deep'").fetchone()
    assert int(row["outcome_ties"]) == 1


def test_record_outcome_accumulates(db):
    for _ in range(3):
        record_outcome(db, "cdx-deep", "win")
    for _ in range(2):
        record_outcome(db, "cdx-deep", "loss")
    row = db.execute(
        "SELECT outcome_wins, outcome_losses FROM alias_reputation WHERE alias='cdx-deep'"
    ).fetchone()
    assert int(row["outcome_wins"]) == 3
    assert int(row["outcome_losses"]) == 2


def test_record_unknown_outcome_is_noop(db):
    record_outcome(db, "cdx-deep", "invalid_outcome")
    row = db.execute("SELECT * FROM alias_reputation WHERE alias='cdx-deep'").fetchone()
    assert row is None


# ---------------------------------------------------------------------------
# get_reputation_delta — neutrality and delta calculation
# ---------------------------------------------------------------------------

def test_delta_neutral_below_min_outcomes(db):
    for _ in range(9):
        record_outcome(db, "cdx-deep", "win")
    assert get_reputation_delta(db, "cdx-deep") == 0.0


def test_delta_neutral_for_unknown_alias(db):
    assert get_reputation_delta(db, "unknown-alias") == 0.0


def test_delta_positive_for_high_win_rate(db):
    # 10 wins, 0 losses -> win_rate = 1.0 -> delta = 0.075
    for _ in range(10):
        record_outcome(db, "cdx-deep", "win")
    delta = get_reputation_delta(db, "cdx-deep")
    assert delta == pytest.approx(0.075, abs=1e-4)


def test_delta_negative_for_zero_win_rate(db):
    # 10 losses -> win_rate = 0.0 -> delta = -0.075
    for _ in range(10):
        record_outcome(db, "cdx-deep", "loss")
    delta = get_reputation_delta(db, "cdx-deep")
    assert delta == pytest.approx(-0.075, abs=1e-4)


def test_delta_neutral_for_even_record(db):
    # 5 wins, 5 losses -> win_rate = 0.5 -> delta = 0.0
    for _ in range(5):
        record_outcome(db, "cdx-deep", "win")
    for _ in range(5):
        record_outcome(db, "cdx-deep", "loss")
    delta = get_reputation_delta(db, "cdx-deep")
    assert delta == pytest.approx(0.0, abs=1e-4)


def test_delta_ties_count_as_half_win(db):
    # 10 ties -> effective_wins=5, win_rate=0.5 -> delta=0.0
    for _ in range(10):
        record_outcome(db, "cdx-deep", "tie")
    delta = get_reputation_delta(db, "cdx-deep")
    assert delta == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# reviewer integration — review_pair uses delta when connection provided
# ---------------------------------------------------------------------------

def test_review_pair_reputation_boosts_score(db):
    """Agent with strong reputation should have higher score_a than identical base-score agent."""
    from orchestra.engine.reviewer import review_pair
    from orchestra.models import AgentRun, AgentStatus

    # Give cdx-deep a strong positive reputation
    for _ in range(10):
        record_outcome(db, "cdx-deep", "win")

    agent_a = AgentRun(alias="cdx-deep", provider="codex", model="gpt-4")
    agent_a.status = AgentStatus.COMPLETED
    agent_a.confidence = 0.70
    agent_a.stdout_log = "answer"

    agent_b = AgentRun(alias="gmn-pro", provider="gemini", model="gemini-pro")
    agent_b.status = AgentStatus.COMPLETED
    agent_b.confidence = 0.70  # identical base confidence
    agent_b.stdout_log = "answer"

    decision = review_pair("round1", agent_a, agent_b, connection=db)
    assert decision.score_a > decision.score_b


def test_review_pair_without_connection_no_delta():
    """Without a connection, review_pair works as before — no reputation delta."""
    from orchestra.engine.reviewer import review_pair
    from orchestra.models import AgentRun, AgentStatus

    agent_a = AgentRun(alias="cdx-deep", provider="codex", model="gpt-4")
    agent_a.status = AgentStatus.COMPLETED
    agent_a.confidence = 0.80
    agent_a.stdout_log = "a"

    agent_b = AgentRun(alias="gmn-pro", provider="gemini", model="gemini-pro")
    agent_b.status = AgentStatus.COMPLETED
    agent_b.confidence = 0.60
    agent_b.stdout_log = "b"

    decision = review_pair("round1", agent_a, agent_b)
    assert decision.winner == "cdx-deep"
