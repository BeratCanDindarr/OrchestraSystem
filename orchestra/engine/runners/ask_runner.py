"""Ask mode runner: single-agent execution with idempotency."""
from __future__ import annotations

from orchestra import config
from orchestra.engine.runners.common import _run_ask_idempotent, _run_ask_impl
from orchestra.models import OrchestraRun
from orchestra.providers.fallback import require_known


def run_ask(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    """Public entry point: dispatches to idempotent wrapper when enabled.

    When reflection is enabled, the result is passed through an optional self-critique +
    retry loop (skipped while a reflection loop is already in flight, to avoid recursion).
    """
    require_known(alias)
    # An internal step (verifier, synthesiser) belongs to one parent run: an earlier run's
    # step must not be handed back in its place.
    if getattr(config, "idempotency_enabled", lambda: True)() and not kwargs.get("internal") \
            and not kwargs.get("follows_run_id"):
        run = _run_ask_idempotent(alias, prompt, **kwargs)
    else:
        run = _run_ask_impl(alias, prompt, **kwargs)
    # Recover a failed run first (retry/degrade), then refine a successful one.
    if config.retry_enabled():
        from orchestra.engine import degradation
        if not degradation.is_degrading():
            run = degradation.degrade(alias, prompt, run, run_ask_fn=run_ask)
    if config.reflection_enabled():
        from orchestra.engine import reflection
        if not reflection.is_reflecting():
            run = reflection.reflect(alias, prompt, run, run_ask_fn=run_ask)
    return run
