"""Ask mode runner: single-agent execution with idempotency."""
from __future__ import annotations

from orchestra import config
from orchestra.engine.runners.common import _run_ask_idempotent, _run_ask_impl
from orchestra.models import OrchestraRun


def run_ask(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    """Public entry point: dispatches to idempotent wrapper when enabled.

    When reflection is enabled, the result is passed through an optional self-critique +
    retry loop (skipped while a reflection loop is already in flight, to avoid recursion).
    """
    if getattr(config, "idempotency_enabled", lambda: True)():
        run = _run_ask_idempotent(alias, prompt, **kwargs)
    else:
        run = _run_ask_impl(alias, prompt, **kwargs)
    if config.reflection_enabled():
        from orchestra.engine import reflection
        if not reflection.is_reflecting():
            run = reflection.reflect(alias, prompt, run, run_ask_fn=run_ask)
    return run
