"""Graceful degradation: retry a failed run with bounded attempts, optionally stepping
down to a cheaper same-provider alias, instead of failing terminally.

Only retries failures marked retryable (``run.failure.retryable`` or a soft-fail) — budget
and policy denials are terminal. Bounded by ``retry.max_attempts`` with exponential backoff.
Config-gated (``retry.enabled``, default off). run_ask and sleep are injected so the loop is
pure and testable without real models or wall-clock delay.
"""
from __future__ import annotations

import contextvars
import logging
import time
from typing import Callable, Optional

from orchestra import config
from orchestra.models import AgentStatus, OrchestraRun, RunStatus

logger = logging.getLogger(__name__)

# Set while a degradation loop is running so nested run_ask retries do not recursively
# spawn their own degradation -> unbounded recursion.
_in_degrade: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_degrade", default=False)


def is_degrading() -> bool:
    """True while a degradation loop is running (guards against recursive retry)."""
    return _in_degrade.get()


def _run_failed(run: OrchestraRun) -> bool:
    """True if the run failed outright or any agent failed / soft-failed."""
    if run.status == RunStatus.FAILED:
        return True
    return any(a.status == AgentStatus.FAILED or a.soft_failed for a in run.agents)


def _is_retryable(run: OrchestraRun) -> bool:
    """True if the failure is worth retrying (transient), False for terminal failures.

    An explicit ``run.failure`` takes precedence (budget/policy set retryable=False). Absent
    that, a soft-fail is treated as transient and retryable.
    """
    if run.failure is not None:
        return bool(getattr(run.failure, "retryable", False))
    return any(a.soft_failed for a in run.agents)


def cheaper_alias(alias: str) -> Optional[str]:
    """Return the most capable strictly-cheaper alias from the same provider, or None."""
    from orchestra.providers.registry import AliasRegistry

    current = AliasRegistry.get(alias)
    if current is None:
        return None
    cheaper = [
        a for a in AliasRegistry.by_provider(current.provider)
        if a.cost_per_1k < current.cost_per_1k
    ]
    if not cheaper:
        return None
    # Among the cheaper siblings, pick the most capable (highest cost below current).
    return max(cheaper, key=lambda a: a.cost_per_1k).name


def degrade(
    alias: str,
    prompt: str,
    base_run: OrchestraRun,
    *,
    run_ask_fn: Callable[..., OrchestraRun],
    max_attempts: Optional[int] = None,
    degrade_to_cheaper: Optional[bool] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> OrchestraRun:
    """Retry ``base_run`` while it is failed-and-retryable, up to ``max_attempts``.

    Returns the first successful run, or the last attempt if none succeed. When
    ``degrade_to_cheaper`` is set, each retry uses a cheaper same-provider alias.
    Recursion-guarded; no-op when the base run already succeeded or is terminal.
    """
    if is_degrading() or not _run_failed(base_run) or not _is_retryable(base_run):
        return base_run

    attempts = max_attempts if max_attempts is not None else config.retry_max_attempts()
    to_cheaper = degrade_to_cheaper if degrade_to_cheaper is not None else config.retry_degrade_to_cheaper()
    backoff_ms = config.retry_backoff_ms()
    multiplier = config.retry_backoff_multiplier()

    token = _in_degrade.set(True)
    try:
        current = base_run
        current_alias = alias
        for attempt in range(max(0, attempts)):
            if not _run_failed(current):
                break                        # recovered
            if not _is_retryable(current):
                break                        # became terminal
            next_alias = current_alias
            if to_cheaper:
                cheaper = cheaper_alias(current_alias)
                if cheaper:
                    next_alias = cheaper
            delay = (backoff_ms * (multiplier ** attempt)) / 1000.0
            if delay > 0:
                sleep_fn(delay)
            logger.info("Degradation attempt %d: retrying %s as %s", attempt + 1, current_alias, next_alias)
            current = run_ask_fn(next_alias, prompt)
            current_alias = next_alias
        return current
    finally:
        _in_degrade.reset(token)
