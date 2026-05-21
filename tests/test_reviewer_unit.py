"""Unit tests for orchestra.engine.reviewer — deterministic, no external deps."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestra.engine.reviewer import ReviewDecision, _content_score, review_pair
from orchestra.models import AgentRun, AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_agent(
    alias: str = "cdx-deep",
    status: AgentStatus = AgentStatus.COMPLETED,
    confidence: float = 0.8,
    stdout_log: str = "some output here",
    soft_failed: bool = False,
    validation_status: str = "not_run",
) -> AgentRun:
    return AgentRun(
        alias=alias,
        provider="codex",
        model="gpt-5.4",
        status=status,
        confidence=confidence,
        stdout_log=stdout_log,
        soft_failed=soft_failed,
        validation_status=validation_status,
    )


# ---------------------------------------------------------------------------
# _content_score
# ---------------------------------------------------------------------------
class TestContentScore:
    @patch("orchestra.engine.reviewer.config.review_config", return_value={})
    def test_base_score_is_confidence(self, _mock_cfg):
        agent = _make_agent(confidence=0.75)
        assert _content_score(agent) == pytest.approx(0.75 + len("some output here".strip()) / 4000.0)

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_partial_bonus": 0.05,
            "validation_failed_penalty": 0.25,
            "soft_failed_penalty": 0.35,
            "length_bonus_divisor": 4000.0,
            "length_bonus_cap": 0.10,
        },
    )
    def test_validation_passed_bonus(self, _mock_cfg):
        agent = _make_agent(confidence=0.5, validation_status="passed")
        assert _content_score(agent) == pytest.approx(0.5 + 0.20 + len("some output here".strip()) / 4000.0)

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_partial_bonus": 0.05,
            "validation_failed_penalty": 0.25,
            "soft_failed_penalty": 0.35,
            "length_bonus_divisor": 4000.0,
            "length_bonus_cap": 0.10,
        },
    )
    def test_validation_failed_penalty(self, _mock_cfg):
        agent = _make_agent(confidence=0.5, validation_status="failed")
        assert _content_score(agent) == pytest.approx(0.5 - 0.25 + len("some output here".strip()) / 4000.0)

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_partial_bonus": 0.05,
            "validation_failed_penalty": 0.25,
            "soft_failed_penalty": 0.35,
            "length_bonus_divisor": 4000.0,
            "length_bonus_cap": 0.10,
        },
    )
    def test_validation_partial_bonus(self, _mock_cfg):
        agent = _make_agent(confidence=0.5, validation_status="partial")
        assert _content_score(agent) == pytest.approx(0.5 + 0.05 + len("some output here".strip()) / 4000.0)

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_partial_bonus": 0.05,
            "validation_failed_penalty": 0.25,
            "soft_failed_penalty": 0.35,
            "length_bonus_divisor": 4000.0,
            "length_bonus_cap": 0.10,
        },
    )
    def test_soft_failed_penalty(self, _mock_cfg):
        agent = _make_agent(confidence=0.5, soft_failed=True)
        assert _content_score(agent) == pytest.approx(0.5 - 0.35 + len("some output here".strip()) / 4000.0)

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_partial_bonus": 0.05,
            "validation_failed_penalty": 0.25,
            "soft_failed_penalty": 0.35,
            "length_bonus_divisor": 4000.0,
            "length_bonus_cap": 0.10,
        },
    )
    def test_reputation_delta_applied(self, _mock_cfg):
        agent = _make_agent(confidence=0.5)
        assert _content_score(agent, reputation_delta=0.15) == pytest.approx(0.5 + 0.15 + len("some output here".strip()) / 4000.0)

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "length_bonus_divisor": 1000.0,
            "length_bonus_cap": 0.05,
        },
    )
    def test_length_bonus_capped(self, _mock_cfg):
        agent = _make_agent(confidence=0.5, stdout_log="x" * 10000)
        score = _content_score(agent)
        length_bonus = min(10000 / 1000.0, 0.05)
        assert score == pytest.approx(0.5 + length_bonus)


# ---------------------------------------------------------------------------
# review_pair
# ---------------------------------------------------------------------------
class TestReviewPair:
    @patch("orchestra.engine.reviewer.config.review_config", return_value={})
    def test_blocked_when_not_completed(self, _mock_cfg):
        a1 = _make_agent(alias="a1", status=AgentStatus.FAILED)
        a2 = _make_agent(alias="a2")
        decision = review_pair("round1", a1, a2)
        assert decision.status == "blocked"
        assert decision.winner == "none"
        assert decision.should_continue is False

    @patch("orchestra.engine.reviewer.config.review_config", return_value={})
    def test_basic_passed(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.85, stdout_log="a" * 1000)
        a2 = _make_agent(alias="a2", confidence=0.80, stdout_log="b" * 100)
        decision = review_pair("round1", a1, a2)
        assert decision.status == "passed"
        assert decision.should_continue is True
        assert decision.should_synthesize is True
        assert decision.winner == "a1"

    @patch("orchestra.engine.reviewer.config.review_config", return_value={})
    def test_tie_when_scores_close(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80)
        a2 = _make_agent(alias="a2", confidence=0.80)
        decision = review_pair("round1", a1, a2)
        assert decision.winner == "tie"

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "confidence_gap_threshold": 0.35,
            "low_avg_confidence_threshold": 0.45,
            "tie_margin": 0.08,
        },
    )
    def test_large_confidence_gap_concern(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.95)
        a2 = _make_agent(alias="a2", confidence=0.50)
        decision = review_pair("round1", a1, a2)
        assert "large_confidence_gap" in decision.concerns
        assert decision.status == "needs_attention"

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "confidence_gap_threshold": 0.35,
            "low_avg_confidence_threshold": 0.45,
            "tie_margin": 0.08,
        },
    )
    def test_low_avg_confidence_concern(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.30)
        a2 = _make_agent(alias="a2", confidence=0.40)
        decision = review_pair("round1", a1, a2)
        assert "low_avg_confidence" in decision.concerns
        assert decision.status == "needs_attention"

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "soft_failed_penalty": 0.35,
            "validation_failed_penalty": 0.25,
            "tie_margin": 0.08,
        },
    )
    def test_single_soft_failed(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80, soft_failed=True)
        a2 = _make_agent(alias="a2", confidence=0.80)
        decision = review_pair("round1", a1, a2)
        assert "a1_soft_failed" in decision.concerns
        assert decision.status == "needs_attention"
        assert decision.reason == "single_output_failed_review"
        assert decision.should_synthesize is False

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "soft_failed_penalty": 0.35,
            "validation_failed_penalty": 0.25,
            "tie_margin": 0.08,
        },
    )
    def test_both_soft_failed_blocked(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80, soft_failed=True)
        a2 = _make_agent(alias="a2", confidence=0.80, soft_failed=True)
        decision = review_pair("round1", a1, a2)
        assert decision.status == "blocked"
        assert decision.reason == "both_outputs_failed_review"
        assert decision.should_continue is False

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_failed_penalty": 0.25,
            "tie_margin": 0.08,
        },
    )
    def test_validation_failed_concern(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80, validation_status="failed")
        a2 = _make_agent(alias="a2", confidence=0.80)
        decision = review_pair("round1", a1, a2)
        assert "a1_validation_failed" in decision.concerns
        assert decision.status == "needs_attention"

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={
            "validation_pass_bonus": 0.20,
            "validation_partial_bonus": 0.05,
            "tie_margin": 0.08,
        },
    )
    def test_validation_partial_concern(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80, validation_status="partial")
        a2 = _make_agent(alias="a2", confidence=0.80)
        decision = review_pair("round1", a1, a2)
        assert "a1_validation_partial" in decision.concerns

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={"tie_margin": 0.08},
    )
    def test_reputation_delta_from_connection(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80)
        a2 = _make_agent(alias="a2", confidence=0.80)
        mock_conn = MagicMock()
        with patch(
            "orchestra.storage.reputation.get_reputation_delta",
            side_effect=lambda _conn, alias: 0.15 if alias == "a1" else 0.0,
        ):
            decision = review_pair("round1", a1, a2, connection=mock_conn)
            assert decision.winner == "a1"

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={"tie_margin": 0.08},
    )
    def test_reputation_delta_exception_ignored(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.80)
        a2 = _make_agent(alias="a2", confidence=0.80)
        mock_conn = MagicMock()
        with patch(
            "orchestra.storage.reputation.get_reputation_delta",
            side_effect=RuntimeError("db error"),
        ):
            decision = review_pair("round1", a1, a2, connection=mock_conn)
            assert decision.status == "passed"

    @patch(
        "orchestra.engine.reviewer.config.review_config",
        return_value={"tie_margin": 0.08},
    )
    def test_winner_by_score(self, _mock_cfg):
        a1 = _make_agent(alias="a1", confidence=0.90, stdout_log="x" * 1000)
        a2 = _make_agent(alias="a2", confidence=0.70, stdout_log="x" * 100)
        decision = review_pair("round1", a1, a2)
        assert decision.winner == "a1"
        assert decision.score_a > decision.score_b

    def test_review_decision_to_dict(self):
        decision = ReviewDecision(
            stage="round1",
            status="passed",
            winner="a1",
            reason="ok",
            should_continue=True,
            should_synthesize=True,
            concerns=[],
            score_a=0.85,
            score_b=0.80,
        )
        d = decision.to_dict()
        assert d["stage"] == "round1"
        assert d["score_a"] == 0.85
        assert d["score_b"] == 0.80
