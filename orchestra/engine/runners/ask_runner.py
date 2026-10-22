"""Ask mode runner: single-agent execution."""
from __future__ import annotations

import functools

from orchestra import config, thinking_modes
from orchestra.engine.runners.common import _run_ask_impl
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
    """Public entry point. A stored answer to the same ask comes from the answer cache only
    (inside _run_ask_impl); the 48 h replay store that did the same was removed.

    When reflection is enabled, the result is passed through an optional self-critique +
    retry loop (skipped while a reflection loop is already in flight, to avoid recursion).
    """
    alias, prompt = _apply_mode(alias, prompt)
    require_known(alias)
    run = _run_ask_impl(alias, prompt, **kwargs)
    # Recover a failed run first (retry/degrade), then refine a successful one. A further call
    # for this task is the same kind of call (a follow-up stays a follow-up, a step stays a
    # step) and is asked again rather than answered from a store.
    # A retry is a run of its own: it must not take over this run's id.
    ask_again = functools.partial(run_ask, **{**{k: v for k, v in kwargs.items() if k != "run_id"}, "fresh": True})
    if config.retry_enabled():
        from orchestra.engine import degradation
        if not degradation.is_degrading():
            run = degradation.degrade(alias, prompt, run, run_ask_fn=ask_again)
    if config.reflection_enabled():
        from orchestra.engine import reflection
        if not reflection.is_reflecting():
            run = reflection.reflect(alias, prompt, run, run_ask_fn=ask_again)
    return run
