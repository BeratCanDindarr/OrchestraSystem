from __future__ import annotations
"""Adversarial orchestration: diversity guard, red-team injection, counter-example search.

Prevents groupthink by:
1. Enforcing ≥2 distinct model families in every multi-agent run.
2. Auto-injecting a red-team agent that argues against the emerging consensus.
3. Dispatching a counter-example search when unanimous agreement is detected.

All features are gated by config [adversarial] enabled flag.
"""

import logging
from typing import List

from orchestra import config

logger = logging.getLogger(__name__)

# Fallback aliases to try when diversity is violated, ordered by preference.
_DIVERSITY_FALLBACKS = {
    "codex": ["gmn-pro", "cld-deep", "kimi-deep", "oll-coder"],
    "gemini": ["cdx-deep", "cld-deep", "kimi-deep", "oll-coder"],
    "claude": ["kimi-deep", "cdx-deep", "gmn-pro", "oll-coder"],
    "kimi": ["cld-deep", "cdx-deep", "gmn-pro", "oll-coder"],
    "ollama": ["cld-deep", "kimi-deep", "cdx-deep", "gmn-pro"],
}


def _can_run(alias: str) -> bool:
    """Whether the alias's own provider can run: a fallback that is switched off or logged
    out would be answered by yet another model under its name, which is not diversity."""
    from orchestra.providers.fallback import ALIAS_TABLE
    entry = ALIAS_TABLE.get(alias)
    return bool(entry and entry[0].is_available())


def _provider_name(alias: str) -> str:
    """Resolve an alias to its provider name (e.g. 'codex', 'gemini')."""
    from orchestra.providers.fallback import resolve

    provider, _ = resolve(alias)
    return provider.name


def ensure_diversity(aliases: List[str]) -> List[str]:
    """Guarantee that the agent list contains ≥2 distinct provider families.

    If all aliases resolve to the same provider, the last alias is replaced
    with a fallback from a different family. The list is mutated in place
    and also returned for convenience.
    """
    if not config.adversarial_enabled() or not config.adversarial_diversity_enforced():
        return aliases
    if len(aliases) < 2:
        return aliases

    names = [_provider_name(a) for a in aliases]
    if len(set(names)) >= 2:
        return aliases

    # All same family — replace last alias with a fallback from another family.
    primary = names[0]
    fallbacks = _DIVERSITY_FALLBACKS.get(primary, ["cld-deep", "kimi-deep", "gmn-pro", "cdx-deep"])
    for fb in fallbacks:
        if _can_run(fb) and _provider_name(fb) != primary:
            logger.info("Diversity guard: replacing %s with %s", aliases[-1], fb)
            aliases[-1] = fb
            return aliases

    return aliases


def inject_red_team(run, prompt: str, **kwargs) -> None:
    """Spawn a red-team agent against the emerging consensus and append to run.agents."""
    if not config.adversarial_enabled():
        return

    alias = config.adversarial_red_team_alias()
    red_team_prompt = (
        f"The following analyses reached a consensus on this task:\n\n"
        f"{prompt}\n\n"
        f"Your sole job is to argue the opposite of the emerging consensus. "
        f"Find flaws, expose unstated assumptions, and challenge the conclusion. "
        f"Be specific and constructive."
    )

    # Lazy import to avoid circular dependency with runner.
    from orchestra.engine.runners import run_ask
    from orchestra.engine.artifacts import append_event

    rt_run = run_ask(alias, red_team_prompt, run_id=run.run_id, **kwargs)
    if rt_run.agents and rt_run.agents[0].stdout_log:
        agent = rt_run.agents[0]
        agent.alias = "red-team"
        run.agents.append(agent)
        append_event(
            run.run_id,
            {
                "event": "red_team_injected",
                "alias": alias,
                "output_chars": len(agent.stdout_log),
            },
        )
        logger.info("Red-team agent injected (%s, %d chars)", alias, len(agent.stdout_log))


def search_counter_example(run, prompt: str, **kwargs) -> None:
    """If consensus looks too uniform, spawn an agent to find a failure mode.

    When the counter-agent produces output, the run summary is downgraded
    with a challenge banner and the run is marked consensus_challenged.
    """
    if not config.adversarial_enabled():
        return
    if getattr(run, "consensus_challenged", False):
        return
    if not run.summary:
        return

    alias = config.adversarial_counter_example_alias()
    counter_prompt = (
        f"The following consensus answer was reached:\n\n"
        f"{run.summary}\n\n"
        f"Find ONE concrete failure mode, edge case, or unstated assumption "
        f"that invalidates or weakens this consensus. Be specific."
    )

    from orchestra.engine.runners import run_ask
    from orchestra.engine.artifacts import append_event

    ce_run = run_ask(alias, counter_prompt, run_id=run.run_id, **kwargs)
    if ce_run.agents and ce_run.agents[0].stdout_log:
        agent = ce_run.agents[0]
        excerpt = agent.stdout_log[:800].strip()
        run.summary = f"> ⚠️ Consensus challenged:\n{excerpt}\n\n---\n\n{run.summary}"
        run.consensus_challenged = True
        append_event(
            run.run_id,
            {
                "event": "consensus_challenged",
                "alias": alias,
                "excerpt_chars": len(excerpt),
            },
        )
        logger.info("Consensus challenged by %s (%d chars)", alias, len(excerpt))
