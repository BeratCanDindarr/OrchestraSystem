"""Critical mode runner: multi-round with approval gates and critic."""
from __future__ import annotations

import orchestra.engine.runners as _runner

import logging
import time

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.artifacts import append_event
from orchestra.engine.parallel import run_parallel
from orchestra.engine.policy_engine import PolicyDecision
from orchestra.engine.runners.common import (
    _append_event_scrubbed,
    approval_requested_event,
    _evaluate_policy,
    _finalize,
    _get_tracer,
    _emit_span_usage,
    _now,
    _prefixed,
    _record_round1_review,
    _run_verification_loop,
    _synthesize_if_possible,
    _validate_persona_alias,
)
from orchestra.engine.state_suspension import delete_paused_run, suspend_run as suspend_run_to_db
from orchestra.engine.workspace_guard import guard_critical_files
from orchestra.monitoring.cost_monitor import _check_budget
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.state import ApprovalState, FailureKind, FailureState

logger = logging.getLogger("orchestra.engine.runner")


def _resume_critical(run: OrchestraRun, checkpoint: dict | None, **kwargs) -> OrchestraRun:
    force_continue = bool(kwargs.get("force_continue", False))
    checkpoint_label = (checkpoint or {}).get("label", "")
    checkpoint_status = (checkpoint or {}).get("status", "")

    if checkpoint_label == "approval_gate" and run.approval_state == ApprovalState.PENDING and not force_continue:
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(run.run_id, {"event": "run_resume_checkpointed", "mode": run.mode, "label": checkpoint_label})
        return run

    if force_continue or run.approval_state in {ApprovalState.APPROVED, ApprovalState.RESUMED} or checkpoint_status == RunStatus.RUNNING.value:
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.RESUMED
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(run.run_id, {"event": "run_resumed", "mode": run.mode, "label": checkpoint_label or "manifest"})

        # Load persisted state from suspension storage if available
        try:
            from orchestra.storage.db import get_db
            from orchestra.engine.state_suspension import resume_run as load_paused_state
            connection = get_db()
            paused_run = load_paused_state(connection, run.run_id)
            if paused_run:
                # Sync agent outputs and reviews from paused state
                run.agents = paused_run.agents
                run.reviews = paused_run.reviews
                run.summary = paused_run.summary
                delete_paused_run(connection, run.run_id)
                append_event(run.run_id, {"event": "paused_state_restored", "agent_count": len(run.agents)})
            connection.close()
        except Exception as e:
            logger.warning("Failed to restore paused state: %s", e, exc_info=True)

        _synthesize_if_possible(run, run.task, **kwargs)
        _finalize(run)
        return run

    if checkpoint_status == RunStatus.WAITING_APPROVAL.value:
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        return run

    return run


def run_critical(
    prompt: str,
    *,
    require_approval: bool = False,
    approval_behavior: str = "continue",
    **kwargs,
) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    run = OrchestraRun(mode="critical", task=prompt)
    run.status = RunStatus.RUNNING
    artifacts.write_manifest(run)
    artifacts.write_checkpoint(run, "started")
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "critical"})
    from orchestra.engine.replay import get_recorder
    get_recorder(run.run_id).record_run_started("critical", prompt)

    # Block 2: WorkspaceGuard
    try:
        guard_critical_files()
    except Exception:
        logger.warning("Workspace guard failed", exc_info=True)

    # Block 7: OtelTracer instrumentation
    agent_alias = "critical"
    span = _get_tracer().start_agent_span(run.run_id, agent_alias, f"critical")
    t0 = time.time()

    try:
        # Use faster agents for round1 so approval-gated flows become responsive.
        # Deep reasoning can still happen in the later synthesis / continuation stages.
        if config.local_first_enabled():
            local = config.local_first_config()
            fast_a = local.get("persona_verifier", "oll-fast")
            fast_b = local.get("persona_implementer", "oll-coder")
            if fast_a == fast_b:
                fast_a, fast_b = "oll-fast", "oll-coder"
        else:
            personas = config.personas_config()
            fast_a = _validate_persona_alias(
                personas.get("verifier"), "gmn-fast", "verifier"
            )
            fast_b = _validate_persona_alias(
                personas.get("implementer"), "cdx-fast", "implementer"
            )
            if fast_a == fast_b:
                fast_a, fast_b = "cdx-fast", "gmn-fast"
        from orchestra.engine.adversarial import ensure_diversity
        fast_a, fast_b = ensure_diversity([fast_a, fast_b])
        run_parallel(run, [(fast_a, _prefixed(fast_a, prompt, run)), (fast_b, _prefixed(fast_b, prompt, run))], **kwargs)

        # Budget enforcement after round 1
        run.total_cost_usd = round(sum(a.estimated_cost_usd for a in run.agents if a.status == AgentStatus.COMPLETED), 6)
        ok, reason = _check_budget(run)
        if not ok:
            logger.warning(f"Budget enforcement: {reason}")
            run.status = RunStatus.FAILED
            run.failure = FailureState(
                kind=FailureKind.BUDGET_EXCEEDED.value,
                message=reason,
                retryable=False,
                source="budget_guard",
            )
            for agent in run.agents:
                if agent.status == AgentStatus.QUEUED:
                    agent.status = AgentStatus.CANCELLED
                    agent.error = "budget_exceeded"
            artifacts.write_manifest(run)
            _append_event_scrubbed(run.run_id, {"event": "budget_exceeded", "reason": reason})
            _finalize(run)
            return run

        _record_round1_review(run)
        _run_verification_loop(run, prompt, **kwargs)
        artifacts.write_checkpoint(run, "round1_complete")
        if run.latest_review_status == "blocked":
            _finalize(run)
            return run

        # Block 4: Checkpoint before approval decision
        artifacts.write_checkpoint(run, "verification_complete")

        # Evaluate approval policies if not explicitly required
        if not require_approval:
            decision = _evaluate_policy(run, prompt)
        else:
            decision = PolicyDecision("allow", "approval explicitly required by caller", "legacy")
        if decision.action == "deny":
            run.status = RunStatus.FAILED
            run.failure = FailureState(
                kind=FailureKind.POLICY_DENIED.value,
                message=decision.reason,
                retryable=False,
                source="policy_guard",
            )
            _finalize(run)
            return run
        should_pause = require_approval or decision.action == "gate" or approval_behavior == "pause"
        if should_pause:
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.PENDING
            run.updated_at = _now()
            artifacts.write_checkpoint(run, "approval_gate")
            _append_event_scrubbed(run.run_id, approval_requested_event("round1", decision))

            # Persist state to database for resumption after approval
            try:
                from orchestra.storage.db import get_db
                connection = get_db()
                # Block 3: UPSERT suspend_run
                suspend_run_to_db(connection, run, paused_by="approval_gate")
                connection.close()
            except Exception as e:
                logger.warning("Failed to persist paused state: %s", e, exc_info=True)

            return run

        from orchestra.engine.adversarial import inject_red_team
        inject_red_team(run, prompt, **kwargs)
        _synthesize_if_possible(run, prompt, **kwargs)
        _critic_rerun_if_needed(run, prompt, **kwargs)
        _finalize(run)
        return run
    except Exception as e:
        # Block 7: Record error and timing
        if span:
            _get_tracer().set_error(span, str(e))
        raise
    finally:
        # Block 7: End span with timing metrics
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
            _emit_span_usage(span, run)
            span.end()


def _critic_rerun_if_needed(run: OrchestraRun, prompt: str, **kwargs) -> None:
    """CriticAgent: one-shot follow-up when dissent found + confidence below threshold.

    Only fires on critical mode, at most once per run (guarded by event check).
    """
    if not run.summary:
        return

    # Guard: only once per run
    existing_events = artifacts.read_events(run.run_id)
    if any(e.get("event") == "critic_triggered" for e in existing_events):
        return

    review = config.review_config()
    threshold = float(review.get("critic_confidence_threshold", 0.65))
    if run.avg_confidence >= threshold:
        return

    from orchestra.engine.synthesizer import parse_dissent
    dissent = parse_dissent(run.summary)
    if not dissent:
        return

    critic_alias = str(review.get("critic_alias", "cld-fast"))
    critic_prompt = (
        f"A previous analysis of the following task had this dissenting viewpoint:\n\n"
        f"DISSENT:\n{dissent}\n\n"
        f"ORIGINAL TASK:\n{prompt}\n\n"
        f"Address the dissent directly. Provide a revised, confident answer that resolves "
        f"the disagreement or explains why the original conclusion stands."
    )
    _append_event_scrubbed(run.run_id, {"event": "critic_triggered", "alias": critic_alias, "dissent_length": len(dissent)})
    from orchestra.engine.runners.ask_runner import run_ask
    critic_run = run_ask(
        critic_alias,
        critic_prompt,
        emit_console=False,
        show_live=False,
        install_signal_handlers=False,
    )
    if critic_run.agents and critic_run.agents[0].stdout_log:
        run.summary = critic_run.agents[0].stdout_log
        _append_event_scrubbed(run.run_id, {"event": "critic_completed", "alias": critic_alias})
