"""Cascade Routing — cheap model first, escalate on low confidence.

Strategy (not multi-armed bandit):
  1. Heuristic pick: task complexity → starting tier (light|medium|heavy)
  2. Run the cheapest alias in the escalation chain
  3. If run.avg_confidence >= threshold → done
  4. Else → escalate to next alias in chain
  5. Outcome history biases the confidence threshold (lower threshold if that
     alias historically performs well on similar tasks)
  6. Provider health check skips degraded providers (uses rate_limit circuit-breaker state)

Config section (.orchestra/config.toml):
  [cascade]
  confidence_threshold = 0.70     # escalate when confidence < this
  escalation_chain = ["cdx-fast", "cdx-deep"]   # default chain
  max_escalations = 2             # safety cap
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from orchestra import config

if TYPE_CHECKING:
    from orchestra.models import OrchestraRun

_DEFAULT_CHAIN = ["cdx-fast", "cdx-deep"]
_DEFAULT_THRESHOLD = 0.70
_DEFAULT_MAX_ESCALATIONS = 2


def _cascade_config() -> dict:
    return config.cascade_config()


def _escalation_chain() -> list[str]:
    cfg = _cascade_config()
    chain = cfg.get("escalation_chain", _DEFAULT_CHAIN)
    return list(chain) if chain else list(_DEFAULT_CHAIN)


def _confidence_threshold() -> float:
    cfg = _cascade_config()
    return float(cfg.get("confidence_threshold", _DEFAULT_THRESHOLD))


def _max_escalations() -> int:
    cfg = _cascade_config()
    return int(cfg.get("max_escalations", _DEFAULT_MAX_ESCALATIONS))


def _is_provider_healthy(alias: str) -> bool:
    """Return False if the provider backing this alias is circuit-broken (degraded)."""
    try:
        from orchestra.engine.provider_guard import _DEGRADED_UNTIL
        import time
        _PREFIX_MAP = {"cdx": "codex", "gmn": "gemini", "cld": "claude", "oll": "ollama"}
        prefix = alias[:3]
        provider_name = _PREFIX_MAP.get(prefix, alias.split("-")[0])
        degraded_until = _DEGRADED_UNTIL.get(provider_name, 0.0)
        return degraded_until <= time.time()
    except Exception:
        return True  # assume healthy if guard not available


def _outcome_adjusted_threshold(alias: str, task: str, base_threshold: float) -> float:
    """Lower the escalation threshold if alias historically performs well on similar tasks.

    A strong historical track record (avg confidence > 0.75 on similar tasks)
    gives up to -0.05 leeway on the threshold.
    """
    try:
        from orchestra.router.outcome_router import (
            _tokenize, _jaccard, _load_outcomes, _outcomes_path
        )
        outcomes = _load_outcomes(_outcomes_path())
        if not outcomes:
            return base_threshold

        task_tokens = _tokenize(task)
        similar = [
            o for o in outcomes
            if _jaccard(task_tokens, _tokenize(o.get("task", ""))) > 0.15
        ]
        if len(similar) < 5:
            return base_threshold

        avg_conf = sum(float(o.get("confidence", 0)) for o in similar) / len(similar)
        if avg_conf > 0.75:
            return max(base_threshold - 0.05, 0.50)
    except Exception:
        pass
    return base_threshold


class CascadeRouter:
    """Run cheap-first, escalate to stronger model when confidence is too low."""

    def __init__(
        self,
        chain: list[str] | None = None,
        confidence_threshold: float | None = None,
        max_escalations: int | None = None,
    ) -> None:
        self._chain = chain if chain is not None else _escalation_chain()
        self._threshold = confidence_threshold if confidence_threshold is not None else _confidence_threshold()
        self._max_escalations = max_escalations if max_escalations is not None else _max_escalations()

    def run(
        self,
        task: str,
        *,
        emit_console: bool = True,
        show_live: bool = True,
        install_signal_handlers: bool = True,
    ) -> "OrchestraRun":
        """Execute cascade: cheap first, escalate until confidence threshold met."""
        from orchestra.engine.runner import run_ask

        chain = self._chain
        escalations_done = 0
        last_run: "OrchestraRun | None" = None

        for alias in chain:
            if escalations_done > self._max_escalations:
                break

            if not _is_provider_healthy(alias):
                escalations_done += 1
                continue

            adjusted_threshold = _outcome_adjusted_threshold(alias, task, self._threshold)

            last_run = run_ask(
                alias,
                task,
                emit_console=emit_console,
                show_live=show_live,
                install_signal_handlers=install_signal_handlers,
            )

            if last_run.avg_confidence >= adjusted_threshold:
                return last_run

            escalations_done += 1

        # Exhausted chain or cap — return last result
        if last_run is not None:
            return last_run

        # Fallback: first healthy alias
        for alias in chain:
            if _is_provider_healthy(alias):
                return run_ask(
                    alias, task,
                    emit_console=emit_console,
                    show_live=show_live,
                    install_signal_handlers=install_signal_handlers,
                )

        # Last resort: first alias regardless of health
        return run_ask(
            chain[0] if chain else "cdx-fast",
            task,
            emit_console=emit_console,
            show_live=show_live,
            install_signal_handlers=install_signal_handlers,
        )
