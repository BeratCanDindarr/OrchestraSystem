"""Deterministic reviewer/judge for comparing agent outputs before synthesis."""
from __future__ import annotations

from dataclasses import dataclass

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


def _content_score(agent: AgentRun) -> float:
    score = agent.confidence
    if agent.validation_status == "passed":
        score += 0.2
    elif agent.validation_status == "partial":
        score += 0.05
    elif agent.validation_status == "failed":
        score -= 0.25
    if agent.soft_failed:
        score -= 0.35
    score += min(len(agent.stdout_log.strip()) / 4000.0, 0.1)
    return score


def review_pair(stage: str, agent_a: AgentRun, agent_b: AgentRun) -> ReviewDecision:
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
    if confidence_gap >= 0.35:
        concerns.append("large_confidence_gap")
    if avg_confidence < 0.45:
        concerns.append("low_avg_confidence")

    score_a = _content_score(agent_a)
    score_b = _content_score(agent_b)
    if abs(score_a - score_b) < 0.08:
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
