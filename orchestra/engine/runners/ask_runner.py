"""Ask mode runner: single-agent execution with idempotency."""
from __future__ import annotations

from orchestra import config, thinking_modes
from orchestra.engine.runners.common import _run_ask_idempotent, _run_ask_impl
from orchestra.models import OrchestraRun
from orchestra.providers.fallback import require_known


def _apply_mode(alias: str, prompt: str) -> tuple[str, str]:
    """`alias#mode` → the alias, and the task with the mode's instructions before it.

    The mode may sit before or after a node (`cdx-deep#concise@windows`,
    `cdx-deep@windows#concise`). An unknown mode is dropped and the task is left as it is.
    """
    if "#" not in alias:
        return alias, prompt
    head, _, rest = alias.partition("#")
    mode, at, target = rest.partition("@")
    bare = head + at + target
    instructions = thinking_modes.mode_prompt(mode.strip().lower(), bare)
    return bare, f"{instructions}\n\n{prompt}" if instructions else prompt


def run_ask(alias: str, prompt: str, **kwargs) -> OrchestraRun:
    """Public entry point: dispatches to idempotent wrapper when enabled.

    When reflection is enabled, the result is passed through an optional self-critique +
    retry loop (skipped while a reflection loop is already in flight, to avoid recursion).
    """
    alias, prompt = _apply_mode(alias, prompt)
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
