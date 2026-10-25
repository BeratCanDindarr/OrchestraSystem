"""Durable execution for ask/dual: crash-recovery resume + signal-based suspend.

Resuming a paused ask or dual run.

Honesty boundary: if the process dies WHILE an agent subprocess is running, that
agent's partial output is lost. Resume continues from the last COMPLETED
checkpoint — identical to critical mode's guarantee.

Additionally, because suspend_on_signal's handler serializes run.agents on the
main thread WITHOUT acquiring parallel.py's worker threading.Lock, a worker
thread may be mid-update to an AgentRun at the instant of serialization. The
persisted snapshot can therefore contain a partially-updated (torn) agent record
— for example, status COMPLETED but stdout_log not yet assigned — not merely an
incomplete one. On resume, synthesis (resume_dual) and finalization (resume_ask)
must tolerate a truncated or empty stdout_log in any agent entry. Acquiring a
lock-cooperative snapshot is out of scope (spec constraint: do not touch
parallel.py).

Two-resume-path divergence
--------------------------
A signal-suspended run is persisted with status=WAITING_APPROVAL (set by the
signal handler) and a paused_runs row. When `orchestra resume` is called, it
resumes through resume_runner's **DB branch**: the row is loaded, approval_state
is set to APPROVED, an `approval_approved` checkpoint is written, and
`approval_resumed` is emitted. A HARD-crashed run (no paused_runs row) resumes
through the **checkpoint fallback**: the manifest is loaded, and resume_ask or
resume_dual is invoked directly (setting approval_state=RESUMED and emitting
`run_resumed`). Both paths converge on the same user-visible output (dual:
synthesize+finalize; ask: finalize), but the event log and approval_state differ.
Auditors, dashboards, and replay readers should expect either pattern depending
on whether a paused_runs row exists.
"""
from __future__ import annotations

import logging
import signal
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.artifacts import append_event
from orchestra.models import OrchestraRun, RunStatus
from orchestra.state import ApprovalState

logger = logging.getLogger("orchestra.engine.runner")  # reuse runner logger so resume logs group with runner logs


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
    run: OrchestraRun,
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
    run: OrchestraRun,
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


@contextmanager
def suspend_on_signal(run, *, suspend_fn, get_db_fn, enabled_fn=None):
    """Install SIGINT/SIGTERM handlers that persist the run before exiting.

    No-op (behavior unchanged) unless durable execution is enabled. On signal:
    mark the run WAITING_APPROVAL, persist it to paused_runs, restore the prior
    handlers, and re-raise as KeyboardInterrupt for a clean exit. signal.signal
    only works in the main thread; if unavailable, the guard degrades to no-op.
    """
    if enabled_fn is None:
        enabled_fn = config.durable_execution_enabled
    if not enabled_fn():
        yield
        return

    triggered = False
    previous: dict = {}

    def _handler(signum, frame):
        nonlocal triggered
        del frame
        if triggered:
            return
        triggered = True
        try:
            connection = get_db_fn()
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_state = ApprovalState.RESUMED
            run.updated_at = _now()
            suspend_fn(connection, run, paused_by="signal")
            connection.close()
            append_event(run.run_id, {"event": "durable_suspend", "signal": int(signum)})
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to suspend run on signal: %s", e, exc_info=True)
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            # Not in the main thread — cannot install; leave as no-op.
            previous.pop(sig, None)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
