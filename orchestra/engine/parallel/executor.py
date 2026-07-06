"""Run cancellation helpers for parallel execution."""
from __future__ import annotations

import asyncio
import threading

from orchestra.engine import artifacts
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.state import InterruptState

from .budget import _now


def _mark_run_cancelled(run: OrchestraRun, lock: threading.Lock, reason: str) -> None:
    with lock:
        already_cancelled = run.status == RunStatus.CANCELLED
        run.status = RunStatus.CANCELLED
        run.interrupt_state = InterruptState.CANCEL_REQUESTED
        run.updated_at = _now()
        for agent in run.agents:
            if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
                agent.status = AgentStatus.CANCELLED
                agent.end_time = agent.end_time or run.updated_at
                agent.error = reason
                agent.pid = None
        artifacts.write_manifest(run)
        if not already_cancelled:
            artifacts.append_event(run.run_id, {"event": "run_cancelled", "reason": reason})


async def _mark_run_cancelled_async(run: OrchestraRun, lock: threading.Lock, reason: str) -> None:
    with lock:
        already_cancelled = run.status == RunStatus.CANCELLED
        run.status = RunStatus.CANCELLED
        run.interrupt_state = InterruptState.CANCEL_REQUESTED
        run.updated_at = _now()
        for agent in run.agents:
            if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
                agent.status = AgentStatus.CANCELLED
                agent.end_time = agent.end_time or run.updated_at
                agent.error = reason
                agent.pid = None
        artifacts.write_manifest(run)
        if not already_cancelled:
            artifacts.append_event(run.run_id, {"event": "run_cancelled", "reason": reason})

