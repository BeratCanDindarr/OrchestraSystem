"""Ask mode runner: single-agent execution with idempotency."""
from __future__ import annotations

from orchestra import config
from orchestra.engine.runners.common import _run_ask_idempotent, _run_ask_impl
from orchestra.models import OrchestraRun


def run_ask(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    """Public entry point: dispatches to idempotent wrapper when enabled."""
    if getattr(config, "idempotency_enabled", lambda: True)():
        return _run_ask_idempotent(alias, prompt, **kwargs)
    return _run_ask_impl(alias, prompt, **kwargs)
