"""Cost estimation, run metrics, and budget enforcement for parallel runs."""
from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from orchestra import config
from orchestra.engine import artifacts
from orchestra.engine.process_group import ProcessGroupManager
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.state import FailureKind, FailureState, InterruptState


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    from orchestra.engine.context_compression import estimate_tokens
    return estimate_tokens(text)


def _estimate_cost_usd(model: str, estimated_tokens: int) -> float:
    if estimated_tokens <= 0:
        return 0.0
    return round((estimated_tokens / 1000.0) * config.price_per_1k_tokens(model), 6)


def _daily_spend_usd(exclude_run_id: str) -> float:
    from orchestra.storage.db import get_db

    today = datetime.now(timezone.utc).date().isoformat()
    connection = get_db()
    try:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(total_cost_usd), 0.0)
            FROM runs
            WHERE substr(created_at, 1, 10) = ? AND run_id != ?
            """,
            (today, exclude_run_id),
        ).fetchone()
        return float(row[0] or 0.0) if row else 0.0
    finally:
        connection.close()


def _refresh_run_metrics(run: OrchestraRun) -> None:
    completed = [agent for agent in run.agents if agent.status == AgentStatus.COMPLETED]
    run.total_cost_usd = round(sum(agent.estimated_cost_usd for agent in completed), 6)
    run.avg_confidence = round(
        sum(agent.confidence for agent in completed) / len(completed),
        3,
    ) if completed else 0.0


async def _enforce_budget_guard_async(run: OrchestraRun, lock: threading.Lock, process_manager: ProcessGroupManager, cancel_event: asyncio.Event) -> None:
    from orchestra.monitoring.cost_monitor import _check_budget
    with lock:
        ok, reason = _check_budget(run)
        max_budget = float(config.max_budget_usd())
        daily_budget = float(config.token_budget_config().get("daily_budget_usd", 0.0))
        historical_daily_spend = _daily_spend_usd(run.run_id) if daily_budget > 0 else 0.0
        exceeds_daily_budget = daily_budget > 0 and (historical_daily_spend + run.total_cost_usd) >= daily_budget
    if ok and not exceeds_daily_budget:
        return

    cancel_event.set()
    await process_manager.kill_all_async()

    with lock:
        if not ok:
            logger.warning(f"Budget enforcement: {reason}")
            message = reason
        else:
            message = f"Daily budget exceeded: ${historical_daily_spend + run.total_cost_usd:.4f} >= ${daily_budget:.4f}"
            logger.warning(f"Budget enforcement: {message}")

        run.status = RunStatus.FAILED
        run.interrupt_state = InterruptState.CANCEL_REQUESTED
        run.failure = FailureState(
            kind=FailureKind.BUDGET_EXCEEDED.value,
            message=message,
            retryable=False,
            source="budget_guard",
        )
        run.updated_at = _now()

        for agent in run.agents:
            if agent.status in (AgentStatus.QUEUED, AgentStatus.STARTED):
                agent.status = AgentStatus.CANCELLED
                agent.end_time = agent.end_time or run.updated_at
                agent.error = "budget_exceeded"
                agent.pid = None

        artifacts.append_event(
            run.run_id,
            {
                "event": "budget_exceeded",
                "reason": message,
                "total_cost_usd": run.total_cost_usd,
                "max_budget_usd": max_budget,
                "daily_budget_usd": daily_budget,
                "historical_daily_spend_usd": round(historical_daily_spend, 6),
            },
        )

