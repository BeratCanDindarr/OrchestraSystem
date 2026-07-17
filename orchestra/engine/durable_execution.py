"""Durable execution for ask/dual: crash-recovery resume + signal-based suspend.

Mirrors critical_runner._resume_critical for the ask/dual modes.

Honesty boundary: if the process dies WHILE an agent subprocess is running, that
agent's partial output is lost. Resume continues from the last COMPLETED
checkpoint — identical to critical mode's guarantee.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from orchestra.engine import artifacts
from orchestra.engine.artifacts import append_event
from orchestra.models import OrchestraRun, RunStatus
from orchestra.state import ApprovalState

logger = logging.getLogger("orchestra.engine.runner")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _restore_paused_state(run, load_paused_fn, delete_paused_fn, get_db_fn) -> None:
    """Sync agents/reviews/summary from DB-persisted paused state, if present."""
    try:
        connection = get_db_fn()
        try:
            paused_run = load_paused_fn(connection, run.run_id)
            if paused_run:
                run.agents = paused_run.agents
                run.reviews = paused_run.reviews
                run.summary = paused_run.summary
                delete_paused_fn(connection, run.run_id)
                append_event(
                    run.run_id,
                    {"event": "paused_state_restored", "agent_count": len(run.agents)},
                )
        finally:
            connection.close()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to restore paused state: %s", e, exc_info=True)


def _prepare_resume(run, checkpoint, force_continue: bool) -> Optional[str]:
    """Shared gate/transition. Returns 'wait', 'continue', or None (unchanged)."""
    checkpoint_label = (checkpoint or {}).get("label", "")
    checkpoint_status = (checkpoint or {}).get("status", "")

    if (
        checkpoint_label == "approval_gate"
        and run.approval_state == ApprovalState.PENDING
        and not force_continue
    ):
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(
            run.run_id,
            {"event": "run_resume_checkpointed", "mode": run.mode, "label": checkpoint_label},
        )
        return "wait"

    if (
        force_continue
        or run.approval_state in {ApprovalState.APPROVED, ApprovalState.RESUMED}
        or checkpoint_status == RunStatus.RUNNING.value
    ):
        run.status = RunStatus.RUNNING
        run.approval_state = ApprovalState.RESUMED
        run.updated_at = _now()
        artifacts.write_manifest(run)
        append_event(
            run.run_id,
            {"event": "run_resumed", "mode": run.mode, "label": checkpoint_label or "manifest"},
        )
        return "continue"

    if checkpoint_status == RunStatus.WAITING_APPROVAL.value:
        run.status = RunStatus.WAITING_APPROVAL
        run.updated_at = _now()
        artifacts.write_manifest(run)
        return "wait"

    return None


def resume_ask(
    run,
    checkpoint,
    *,
    finalize_fn,
    load_paused_fn,
    delete_paused_fn,
    get_db_fn,
    force_continue: bool = False,
    **kwargs,
) -> OrchestraRun:
    """Resume an ask-mode run from its last checkpoint (single agent, no synthesis)."""
    if _prepare_resume(run, checkpoint, force_continue) != "continue":
        return run
    _restore_paused_state(run, load_paused_fn, delete_paused_fn, get_db_fn)
    finalize_fn(run)
    return run


def resume_dual(
    run,
    checkpoint,
    *,
    synthesize_fn,
    finalize_fn,
    load_paused_fn,
    delete_paused_fn,
    get_db_fn,
    force_continue: bool = False,
    **kwargs,
) -> OrchestraRun:
    """Resume a dual-mode run from its last checkpoint (synthesize then finalize)."""
    if _prepare_resume(run, checkpoint, force_continue) != "continue":
        return run
    _restore_paused_state(run, load_paused_fn, delete_paused_fn, get_db_fn)
    synthesize_fn(run, run.task, **kwargs)
    finalize_fn(run)
    return run
