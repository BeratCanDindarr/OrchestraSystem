"""Dual mode runner: two-agent parallel execution with review."""
from __future__ import annotations

import orchestra.engine.runners as _runner

import logging
import time

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.parallel import run_parallel
from orchestra.engine.runners.common import show_in_status_line
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
from orchestra.engine.state_suspension import suspend_run as suspend_run_to_db
from orchestra.engine.workspace_guard import guard_critical_files
from orchestra.monitoring.cost_monitor import _check_budget
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.state import ApprovalState, FailureKind, FailureState

logger = logging.getLogger("orchestra.engine.runner")


def run_dual(prompt: str, *, agents: list[str] | None = None, **kwargs) -> OrchestraRun:
    kwargs["install_signal_handlers"] = False
    show_in_status_line()
    run = OrchestraRun(mode="dual", task=prompt, follows_run_id=kwargs.pop("follows_run_id", None)); run.status = RunStatus.RUNNING; artifacts.write_manifest(run)
    _append_event_scrubbed(run.run_id, {"event": "run_started", "mode": "dual"})
    from orchestra.engine.replay import get_recorder
    get_recorder(run.run_id).record_run_started("dual", prompt)

    # Block 2: WorkspaceGuard
    try:
        guard_critical_files()
    except Exception:
        logger.warning("Workspace guard failed", exc_info=True)

    # Block 7: OtelTracer instrumentation
    agent_alias = "dual"
    span = _get_tracer().start_agent_span(run.run_id, agent_alias, f"dual")
    t0 = time.time()

    from orchestra.engine.durable_execution import suspend_on_signal
    from orchestra.engine.state_suspension import suspend_run as _suspend_fn
    from orchestra.storage.db import get_db as _get_db
    _suspend_ctx = suspend_on_signal(run, suspend_fn=_suspend_fn, get_db_fn=_get_db)
    _suspend_ctx.__enter__()

    try:
        if config.local_first_enabled():
            local = config.local_first_config()
            default_a = local.get("persona_implementer", "oll-coder")
            default_b = local.get("persona_researcher", "oll-deep")
        else:
            personas = config.personas_config()
            default_a = _validate_persona_alias(
                personas.get("implementer"), "cdx-deep", "implementer"
            )
            default_b = _validate_persona_alias(
                personas.get("researcher"), "gmn-pro", "researcher"
            )
        _agents = agents if agents and len(agents) >= 2 else [default_a, default_b]
        from orchestra.engine.adversarial import ensure_diversity
        _agents = ensure_diversity(_agents)
        run_parallel(run, [(_agents[0], _prefixed(_agents[0], prompt, run)), (_agents[1], _prefixed(_agents[1], prompt, run))], **kwargs)

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

        # Block 4: Checkpoint after round 1
        artifacts.write_checkpoint(run, "round1_complete")

        _record_round1_review(run)
        _run_verification_loop(run, prompt, **kwargs)
        if run.latest_review_status == "blocked":
            _finalize(run); return run

        # Block 4: Checkpoint after verification
        artifacts.write_checkpoint(run, "verification_complete")

        # Evaluate approval policies (P0 #2: Approval Gates)
        decision = _evaluate_policy(run, prompt)
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
        elif decision.action == "gate":
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.PENDING
            run.updated_at = _now()
            artifacts.write_checkpoint(run, "approval_gate")
            _append_event_scrubbed(run.run_id, approval_requested_event("dual_verification", decision))

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
        _finalize(run); return run
    except Exception as e:
        # Block 7: Record error and timing
        if span:
            _get_tracer().set_error(span, str(e))
        raise
    finally:
        _suspend_ctx.__exit__(None, None, None)  # __exit__ args intentionally None: suspend_on_signal only restores signal handlers and ignores exc info.
        # Block 7: End span with timing metrics
        if span:
            _get_tracer().set_timing(span, "total_latency_ms", (time.time() - t0) * 1000)
            _emit_span_usage(span, run)
            span.end()
