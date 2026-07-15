"""Thinking modes: parse an `alias#mode` model id and load the matching system prompt.

A caller addresses a model as `oll-fast#deep`; the base alias (`oll-fast`) resolves as usual
and the mode (`deep`) selects a system prompt from orchestra/prompts/modes/. The tier
(compact vs full) is chosen from the alias — small local models get the leaner `compact`
prompt, strong/cloud models get the richer `full` one."""
from __future__ import annotations

import functools
from pathlib import Path

_MODES_DIR = Path(__file__).parent / "prompts" / "modes"
KNOWN_MODES = {"fast", "deep", "research", "ideate", "plan", "critic"}
# Aliases strong enough for the richer "full" tier; everything else gets "compact"
# (compact is validated to work everywhere, so it is the safe default).
_FULL_ALIASES = {"oll-kimi", "oll-minimax", "oll-dev", "cdx-deep", "gmn-pro", "cld-deep"}


def split_mode(model: str) -> tuple[str, str | None]:
    """'oll-fast#deep' -> ('oll-fast', 'deep'); 'oll-fast' -> ('oll-fast', None).

    An unknown mode after '#' is ignored (returns None) so a typo degrades to plain."""
    if "#" not in model:
        return model, None
    base, _, mode = model.partition("#")
    mode = mode.strip().lower()
    return base, (mode if mode in KNOWN_MODES else None)


def tier_for_alias(alias: str) -> str:
    """'full' for strong/cloud aliases, else 'compact'."""
    base = alias.split("@", 1)[0].split("#", 1)[0]
    return "full" if base in _FULL_ALIASES else "compact"


@functools.lru_cache(maxsize=64)
def _read(mode: str, tier: str) -> str | None:
    try:
        return (_MODES_DIR / f"{mode}_{tier}.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def mode_prompt(mode: str, alias: str) -> str | None:
    """System-prompt text for a mode, tier chosen from the alias. None if unknown/missing."""
    if mode not in KNOWN_MODES:
        return None
    return _read(mode, tier_for_alias(alias))


def prepend_system(system, mode_text: str):
    """Prepend the mode prompt to an Anthropic `system` value (str, block list, or None)."""
    if not system:
        return mode_text
    if isinstance(system, str):
        return f"{mode_text}\n\n{system}"
    if isinstance(system, list):
        return [{"type": "text", "text": mode_text}, *system]
    return mode_text
