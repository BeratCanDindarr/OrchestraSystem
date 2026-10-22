"""Policy evaluation and the approval gate of a run: the decision, the cost and confidence
estimates it reads, and what a runner does when the decision stops or denies the run.
Moved out of common.py (refactor-oversized-modules.md); common imports it last and
re-exports every name, so callers and patches of common.<name> keep working."""
from __future__ import annotations

from typing import Literal

from orchestra.engine.parallel.budget import _daily_spend_usd
from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.autonomy_policy import AutonomyPolicy
from orchestra.engine.policy_engine import PolicyEngine, PolicyDecision
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.state import ApprovalState, FailureKind, FailureState
from orchestra.engine.runners.common import (  # noqa: E402
    _append_event_scrubbed,
    _now,
    logger,
)


def _evaluate_policy(run: OrchestraRun, prompt: str) -> PolicyDecision:
    """Evaluate all approval / autonomy policies and return a structured decision.

    If .orchestra/policy.yaml exists, the new declarative PolicyEngine is used.
    Otherwise the legacy TOML approval_policies / autonomy_policy path is preserved.

    Returns:
        PolicyDecision with action in ("allow", "gate", "deny").
        "deny" is produced when AutonomyPolicy.default_action == "block".
    """
    engine = PolicyEngine.from_default()
    if engine._has_policy():
        decision = engine.evaluate_run(
            mode=run.mode,
            task=prompt,
            estimated_cost_usd=_estimate_run_cost(run),
            provider_aliases=[a.alias for a in run.agents],
            daily_spend_usd=_daily_spend_usd(run.run_id),
        )
        if decision.action in ("gate", "deny"):
            logger.info(f"Approval triggered: {decision.policy_source} — {decision.reason}")
            _append_event_scrubbed(run.run_id, {
                "event": "policy_evaluated",
                "action": decision.action,
                "reason": decision.reason,
                "policy_source": decision.policy_source,
            })
        return decision

    # ── Legacy TOML path (unchanged when policy.yaml is absent) ──
    policies = config.approval_policies_config()

    if policies.get("enabled", False):
        # Policy 1: Cost-based approval gate
        if policies.get("require_approval_on_high_cost", False):
            cost_threshold = float(policies.get("cost_threshold_usd", 1.0))
            estimated_cost = _estimate_run_cost(run)
            if estimated_cost > cost_threshold:
                logger.info(f"Approval triggered: cost ${estimated_cost:.2f} > threshold ${cost_threshold:.2f}")
                return PolicyDecision("gate", f"cost ${estimated_cost:.2f} > threshold ${cost_threshold:.2f}", "legacy:cost")

        # Policy 2: Confidence-based approval gate
        if policies.get("require_approval_on_low_confidence", False):
            confidence_threshold = float(policies.get("confidence_threshold", 0.7))
            avg_confidence = _estimate_avg_confidence(run)
            if avg_confidence < confidence_threshold:
                logger.info(f"Approval triggered: confidence {avg_confidence:.2f} < threshold {confidence_threshold:.2f}")
                return PolicyDecision("gate", f"confidence {avg_confidence:.2f} < threshold {confidence_threshold:.2f}", "legacy:confidence")

        # Policy 3: Task complexity-based approval gate
        if policies.get("require_approval_on_complex_task", False):
            keywords = policies.get("complexity_keywords", [])
            if any(kw.lower() in prompt.lower() for kw in keywords):
                logger.info(f"Approval triggered: task matches complexity keywords")
                return PolicyDecision("gate", "task matches complexity keywords", "legacy:complexity")

        # Policy 4: Mode-based approval requirement
        required_modes = policies.get("approval_required_for_modes", [])
        if run.mode in required_modes:
            logger.info(f"Approval triggered: mode '{run.mode}' requires approval")
            return PolicyDecision("gate", f"mode '{run.mode}' requires approval", "legacy:mode")

    # Policy 5: AutonomyPolicy evaluation (multi-factor OR-logic)
    # Always evaluated independently so disabling legacy approvals does NOT
    # silently disable the new governance layer.
    policy = AutonomyPolicy.from_config()
    should_escalate, reason = policy.should_escalate(
        mode=run.mode,
        task=prompt,
        estimated_cost_usd=_estimate_run_cost(run),
    )
    if should_escalate:
        logger.info(f"Approval triggered: autonomy_policy — {reason}")
        action: Literal["deny", "gate"] = "deny" if policy.default_action == "block" else "gate"
        return PolicyDecision(action, reason, "autonomy_policy")

    return PolicyDecision("allow", "no policy triggered", "legacy")


def _should_require_approval(run: OrchestraRun, prompt: str) -> bool:
    """Evaluate approval policies to determine if human approval gate is required.

    Args:
        run: Current OrchestraRun with completed agents and metrics
        prompt: Original task prompt to check against complexity keywords

    Returns:
        bool: True if any approval policy triggers
    """
    policies = config.approval_policies_config()
    if not policies.get("enabled", False):
        return False

    # Policy 1: Cost-based approval gate
    if policies.get("require_approval_on_high_cost", False):
        cost_threshold = float(policies.get("cost_threshold_usd", 1.0))
        estimated_cost = _estimate_run_cost(run)
        if estimated_cost > cost_threshold:
            logger.info(f"Approval triggered: cost ${estimated_cost:.2f} > threshold ${cost_threshold:.2f}")
            return True

    # Policy 2: Confidence-based approval gate
    if policies.get("require_approval_on_low_confidence", False):
        confidence_threshold = float(policies.get("confidence_threshold", 0.7))
        avg_confidence = _estimate_avg_confidence(run)
        if avg_confidence < confidence_threshold:
            logger.info(f"Approval triggered: confidence {avg_confidence:.2f} < threshold {confidence_threshold:.2f}")
            return True

    # Policy 3: Task complexity-based approval gate
    if policies.get("require_approval_on_complex_task", False):
        keywords = policies.get("complexity_keywords", [])
        if any(kw.lower() in prompt.lower() for kw in keywords):
            logger.info(f"Approval triggered: task matches complexity keywords")
            return True

    # Policy 4: Mode-based approval requirement
    required_modes = policies.get("approval_required_for_modes", [])
    if run.mode in required_modes:
        logger.info(f"Approval triggered: mode '{run.mode}' requires approval")
        return True

    return False


def _estimate_run_cost(run: OrchestraRun) -> float:
    """What the run's agents cost so far, as engine/costing.py charged each of them.

    It used to price completion tokens from its own alias table, which knew neither Kimi,
    Ollama nor Antigravity and charged them all at gpt-5.4/medium.
    """
    return sum(float(agent.estimated_cost_usd or 0.0) for agent in run.agents)


def _estimate_avg_confidence(run: OrchestraRun) -> float:
    """Estimate average confidence across completed agents.

    Args:
        run: OrchestraRun with completed agents

    Returns:
        float: Average confidence (0-1), or 1.0 if no agents
    """
    confidences = []
    for agent in run.agents:
        if hasattr(agent, "confidence") and agent.confidence is not None:
            confidences.append(agent.confidence)
    if not confidences:
        return 1.0
    return sum(confidences) / len(confidences)


# ==============================================================================
# Block 6: PiiScrubber Integration — Scrub before EventLog append
# ==============================================================================
def approval_requested_event(stage: str, decision) -> dict:
    """The event written when a run stops for approval.

    Carries the policy's reason when a policy asked for the stop, so the log says
    why a run is waiting and not only that it is.
    """
    event = {"event": "approval_requested", "stage": stage}
    if getattr(decision, "action", None) == "gate":
        event["reason"] = decision.reason
        event["policy_source"] = decision.policy_source
    return event


def _fail_crashed(run: OrchestraRun, error: BaseException, *, finalize) -> None:
    """Give a run that raised a terminal status, so it is not left RUNNING for ever.

    The caller re-raises: whoever started the run still sees the error. What changes is what
    is on disk: a failed run with the reason, and no agent that still claims to be working.
    """
    if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.WAITING_APPROVAL):
        return
    run.status = RunStatus.FAILED
    run.failure = run.failure or FailureState(
        kind=FailureKind.UNKNOWN_RUNTIME_ERROR.value,
        message=f"{type(error).__name__}: {error}"[:500],
        retryable=True,
        source="runner",
    )
    for agent in run.agents:
        if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED, AgentStatus.RETRYING):
            agent.status = AgentStatus.FAILED
            agent.error = agent.error or "the run raised before this agent finished"
            agent.end_time = agent.end_time or _now()
            agent.pid = None
    try:
        finalize(run)
    except Exception:
        logger.warning("could not finalize crashed run %s", run.run_id, exc_info=True)


def _denied_before_start(run: OrchestraRun, aliases: list[str], *, finalize) -> bool:
    """Fail a run the policy forbids outright, before any agent is started.

    Two things can be known up front: a provider that policy.yaml denies, and a daily budget
    that is already spent. Evaluating them only after the agents had answered meant a denied
    provider was called, and paid, anyway. Gates stay where they are: they ask for approval
    of what the agents produced.
    """
    engine = PolicyEngine.from_default()
    if not engine._has_policy():
        return False
    decision = next((verdict for verdict in map(engine.evaluate_provider, aliases)
                     if verdict.action == "deny"), None)
    if decision is None:
        budget = engine.evaluate_budget(0.0, _daily_spend_usd(run.run_id))
        decision = budget if budget.action == "deny" else None
    if decision is None:
        return False
    _append_event_scrubbed(run.run_id, {"event": "policy_evaluated", "action": "deny", "stage": "before_start",
                                        "reason": decision.reason, "policy_source": decision.policy_source})
    run.status = RunStatus.FAILED
    run.failure = FailureState(kind=FailureKind.POLICY_DENIED.value, message=decision.reason,
                               retryable=False, source="policy_guard")
    finalize(run)
    return True


def _stop_at_policy(run: OrchestraRun, decision, stage: str, *, suspend, finalize) -> bool:
    """Act on the policy decision for a run whose agents have answered.

    True when the run ends or pauses here: a denial is recorded and the run finalized, a gate
    suspends it until someone approves. False when the run may go on (no decision, or allow).
    Each runner passes its own `suspend` and `finalize`.
    """
    if decision is None or decision.action not in ("deny", "gate"):
        return False
    if decision.action == "deny":
        run.status = RunStatus.FAILED
        run.failure = FailureState(
            kind=FailureKind.POLICY_DENIED.value,
            message=decision.reason,
            retryable=False,
            source="policy_guard",
        )
        finalize(run)
        return True

    from orchestra.providers.fallback import _writes
    if run.agents and not any(_writes(agent.alias) for agent in run.agents):
        # Nothing these agents did can be undone by holding their answer: they only read.
        _append_event_scrubbed(run.run_id, {"event": "policy_evaluated", "action": "allow",
                                            "reason": f"not gated, the agents only read: {decision.reason}",
                                            "policy_source": getattr(decision, "policy_source", "")})
        return False

    run.status = RunStatus.WAITING_APPROVAL
    run.approval_state = ApprovalState.PENDING
    run.updated_at = _now()
    artifacts.write_checkpoint(run, "approval_gate")
    _append_event_scrubbed(run.run_id, approval_requested_event(stage, decision))
    try:
        from orchestra.storage.db import get_db
        connection = get_db()
        suspend(connection, run, paused_by="approval_gate")
        connection.close()
    except Exception as e:
        logger.warning("Failed to persist paused state: %s", e, exc_info=True)
    return True
