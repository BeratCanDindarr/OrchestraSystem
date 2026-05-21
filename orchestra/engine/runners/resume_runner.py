"""Resume runner: load paused state and continue execution."""
from __future__ import annotations

import orchestra.engine.runners as _runner

import logging

from orchestra.engine import artifacts
from orchestra.engine.artifacts import append_event
from orchestra.engine.runners.common import (
    _finalize,
    _now,
    _run_verification_loop,
    _synthesize_if_possible,
)
from orchestra.engine.runners.critical_runner import _resume_critical
from orchestra.engine.state_suspension import delete_paused_run, resume_run as resume_run_from_db
from orchestra.models import OrchestraRun, RunStatus
from orchestra.state import ApprovalState
from orchestra.storage.db import get_db

logger = logging.getLogger("orchestra.engine.runner")


def resume_run(run_id: str, approval_decision: str = "approve", **kwargs) -> OrchestraRun:
    """Resume a paused run after approval decision (Phase 3: PENDING → APPROVED → RUNNING).

    Loads paused run from database, transitions approval state, and continues execution
    or terminates based on approval decision.

    Args:
        run_id: ID of paused run
        approval_decision: "approve" (continue) or "reject" (fail)
        **kwargs: passed to synthesis/finalization

    Returns:
        OrchestraRun with updated status and approval_state
    """
    connection = get_db()
    paused_run = resume_run_from_db(connection, run_id)

    if paused_run:
        run = paused_run
        # Guard: ensure run is in WAITING_APPROVAL state
        if run.status != RunStatus.WAITING_APPROVAL:
            raise ValueError(f"Cannot resume run {run_id}: status={run.status.value}, expected WAITING_APPROVAL")

        if approval_decision == "approve":
            run.approval_state = ApprovalState.APPROVED
            run.status = RunStatus.RUNNING
            run.updated_at = _now()
            append_event(run.run_id, {"event": "approval_resumed", "decision": "approved"})
            artifacts.write_checkpoint(run, "approval_approved")

            # Continue execution based on mode
            if run.mode == "ask":
                _finalize(run)
            elif run.mode == "dual":
                _synthesize_if_possible(run, run.task, **kwargs)
                _finalize(run)
            elif run.mode == "critical":
                _synthesize_if_possible(run, run.task, **kwargs)
                _finalize(run)
            elif run.mode == "planned":
                # Load completed node_ids from planned_checkpoints
                try:
                    completed_nodes = connection.execute(
                        "SELECT node_id FROM planned_checkpoints WHERE run_id = ? AND status = 'COMPLETED'",
                        (run_id,)
                    ).fetchall()
                    completed_node_ids = {row[0] for row in completed_nodes}
                    # Restart verification with knowledge of completed nodes
                    _run_verification_loop(run, run.task, **kwargs)
                except Exception as e:
                    logger.warning("Failed to load planned_checkpoints: %s", e, exc_info=True)
                    _run_verification_loop(run, run.task, **kwargs)
                _finalize(run)

            try:
                delete_paused_run(connection, run_id)
            except Exception:
                logger.warning("Failed to delete paused run", exc_info=True)
        else:
            run.approval_state = ApprovalState.REJECTED
            run.status = RunStatus.FAILED
            run.updated_at = _now()
            append_event(run.run_id, {"event": "approval_rejected", "decision": "rejected"})
            artifacts.write_checkpoint(run, "approval_rejected")

            try:
                delete_paused_run(connection, run_id)
            except Exception:
                logger.warning("Failed to delete paused run", exc_info=True)

        connection.close()
        return run

    connection.close()

    # Fallback: load from manifest/checkpoint (older resumption path)
    manifest = artifacts.load_manifest(run_id)
    checkpoint = artifacts.latest_checkpoint(run_id)
    if not manifest and not checkpoint:
        raise ValueError(f"Run not found: {run_id}")
    if not manifest and checkpoint:
        raise ValueError(f"Run manifest missing for {run_id}; latest checkpoint is {checkpoint.get('label', 'unknown')}")

    run = OrchestraRun.from_manifest(manifest)
    if run.mode == "critical":
        return _runner._resume_critical(run, checkpoint, **kwargs)

    return run
