"""Deterministic reviewer/judge for comparing agent outputs before synthesis."""
from __future__ import annotations

from dataclasses import dataclass

from orchestra import config
from orchestra.models import AgentRun, AgentStatus


@dataclass
class ReviewDecision:
    stage: str
    status: str
    winner: str
    reason: str
    should_continue: bool
    should_synthesize: bool
    concerns: list[str]
    score_a: float
    score_b: float

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "status": self.status,
            "winner": self.winner,
            "reason": self.reason,
            "should_continue": self.should_continue,
            "should_synthesize": self.should_synthesize,
            "concerns": self.concerns,
            "score_a": round(self.score_a, 3),
            "score_b": round(self.score_b, 3),
        }


def _content_score(agent: AgentRun, reputation_delta: float = 0.0) -> float:
    review = config.review_config()
    score = agent.confidence
    if agent.validation_status == "passed":
        score += float(review.get("validation_pass_bonus", 0.20))
    elif agent.validation_status == "partial":
        score += float(review.get("validation_partial_bonus", 0.05))
    elif agent.validation_status == "failed":
        score -= float(review.get("validation_failed_penalty", 0.25))
    if agent.soft_failed:
        score -= float(review.get("soft_failed_penalty", 0.35))
    length_bonus_divisor = float(review.get("length_bonus_divisor", 4000.0))
    length_bonus_cap = float(review.get("length_bonus_cap", 0.10))
    score += min(len(agent.stdout_log.strip()) / length_bonus_divisor, length_bonus_cap)
    score += reputation_delta
    return score


def review_pair(stage: str, agent_a: AgentRun, agent_b: AgentRun, *, connection=None) -> ReviewDecision:
    review = config.review_config()
    confidence_gap_threshold = float(review.get("confidence_gap_threshold", 0.35))
    low_avg_confidence_threshold = float(review.get("low_avg_confidence_threshold", 0.45))
    tie_margin = float(review.get("tie_margin", 0.08))
    concerns: list[str] = []
    if agent_a.status != AgentStatus.COMPLETED or agent_b.status != AgentStatus.COMPLETED:
        return ReviewDecision(
            stage=stage,
            status="blocked",
            winner="none",
            reason="missing_completed_outputs",
            should_continue=False,
            should_synthesize=False,
            concerns=["one_or_more_agents_incomplete"],
            score_a=0.0,
            score_b=0.0,
        )

    if agent_a.soft_failed:
        concerns.append(f"{agent_a.alias}_soft_failed")
    if agent_b.soft_failed:
        concerns.append(f"{agent_b.alias}_soft_failed")
    if agent_a.validation_status == "failed":
        concerns.append(f"{agent_a.alias}_validation_failed")
    if agent_b.validation_status == "failed":
        concerns.append(f"{agent_b.alias}_validation_failed")
    if agent_a.validation_status == "partial":
        concerns.append(f"{agent_a.alias}_validation_partial")
    if agent_b.validation_status == "partial":
        concerns.append(f"{agent_b.alias}_validation_partial")

    confidence_gap = abs(agent_a.confidence - agent_b.confidence)
    avg_confidence = (agent_a.confidence + agent_b.confidence) / 2.0
    if confidence_gap >= confidence_gap_threshold:
        concerns.append("large_confidence_gap")
    if avg_confidence < low_avg_confidence_threshold:
        concerns.append("low_avg_confidence")

    delta_a = 0.0
    delta_b = 0.0
    if connection is not None:
        try:
            from orchestra.storage.reputation import get_reputation_delta
            delta_a = get_reputation_delta(connection, agent_a.alias)
            delta_b = get_reputation_delta(connection, agent_b.alias)
        except Exception:
            pass
    score_a = _content_score(agent_a, delta_a)
    score_b = _content_score(agent_b, delta_b)
    if abs(score_a - score_b) < tie_margin:
        winner = "tie"
    else:
        winner = agent_a.alias if score_a > score_b else agent_b.alias

    hard_failures = sum(
        1
        for agent in (agent_a, agent_b)
        if agent.soft_failed or agent.validation_status == "failed"
    )
    if hard_failures == 2:
        status = "blocked"
        reason = "both_outputs_failed_review"
        should_continue = False
        should_synthesize = False
    elif hard_failures == 1:
        status = "needs_attention"
        reason = "single_output_failed_review"
        should_continue = True
        should_synthesize = False
    elif "low_avg_confidence" in concerns or "large_confidence_gap" in concerns:
        status = "needs_attention"
        reason = "confidence_signals_require_review"
        should_continue = True
        should_synthesize = True
    elif concerns:
        status = "needs_attention"
        reason = "validation_signals_require_review"
        should_continue = True
        should_synthesize = True
    else:
        status = "passed"
        reason = "outputs_consistent"
        should_continue = True
        should_synthesize = True

    return ReviewDecision(
        stage=stage,
        status=status,
        winner=winner,
        reason=reason,
        should_continue=should_continue,
        should_synthesize=should_synthesize,
        concerns=concerns,
        score_a=score_a,
        score_b=score_b,
    )
