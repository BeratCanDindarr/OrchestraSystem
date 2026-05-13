"""Phase 3 Integration hooks for outcome-weighted routing."""
import logging
from typing import Optional, List, Dict

from orchestra.router.outcome_router import OutcomeRouter
from orchestra.utils.outcome_recorder import OutcomeRecorder

logger = logging.getLogger(__name__)


def resolve_mode(task_description: str, explicit_mode: Optional[str] = None) -> str:
    """Resolve execution mode using outcome routing.

    Entry point before LLM agent chain is created (runner.py integration).

    Args:
        task_description: Natural-language task description.
        explicit_mode: If provided, bypass routing and use this mode directly.

    Returns:
        Selected execution mode: "ask", "dual", "critical", or "planned".
    """
    if explicit_mode:
        logger.info("Outcome routing bypassed: explicit mode=%s", explicit_mode)
        return explicit_mode

    try:
        history = OutcomeRecorder.load_history()
        suggested = OutcomeRouter.suggest_mode(task_description, history)
        logger.info("Outcome router suggested mode=%s for task: %s...", suggested, task_description[:50])
        return suggested
    except Exception as e:
        logger.warning("Outcome routing failed, falling back to keyword: %s", e)
        return OutcomeRouter._keyword_fallback(task_description)


def record_outcome(
    task_text: str,
    mode: str,
    confidence: float,
    total_cost_usd: float,
) -> None:
    """Record run outcome for future EV computation.

    Post-run hook called after review score + cost finalization (reviewer.py integration).

    Args:
        task_text: Original task description.
        mode: Execution mode used.
        confidence: Review score or success probability [0.0, 1.0].
        total_cost_usd: Total API cost incurred during the run.
    """
    try:
        OutcomeRecorder.record_outcome(task_text, mode, confidence, total_cost_usd)
        ev_score = confidence / (total_cost_usd + OutcomeRouter.EPSILON)
        logger.info(
            "Outcome recorded: task=%s... mode=%s confidence=%.2f cost=$%.4f ev=%.2f",
            task_text[:30],
            mode,
            confidence,
            total_cost_usd,
            ev_score,
        )
    except Exception as e:
        logger.warning("Failed to record outcome: %s", e)
