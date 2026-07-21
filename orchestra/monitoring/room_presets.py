"""Ready-made role presets for task rooms.

Each preset is a persona: a canned system-prompt fragment that is prepended to a
task when a model works in a room using that preset. A room may also carry its own
extra ``rules`` text, appended after the preset. ``custom`` is the empty preset —
the room relies solely on its own rules.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoomPreset:
    key: str
    name: str
    prompt: str


_PRESETS: dict[str, RoomPreset] = {
    "custom": RoomPreset(
        "custom", "Custom",
        "",
    ),
    "senior_engineer": RoomPreset(
        "senior_engineer", "Senior Engineer",
        "You are a senior software engineer. Write correct, maintainable, well-tested "
        "code. Prefer simple, idiomatic solutions; call out edge cases and failure modes; "
        "explain trade-offs briefly. Do not over-engineer.",
    ),
    "product_manager": RoomPreset(
        "product_manager", "Product Manager",
        "You are a product manager. Clarify the user problem and goals before solutions. "
        "Think in terms of user value, scope, and priorities; surface risks and open "
        "questions; keep recommendations concrete and outcome-focused.",
    ),
    "researcher": RoomPreset(
        "researcher", "Researcher",
        "You are a rigorous researcher. Investigate thoroughly, weigh evidence, cite "
        "concrete sources or reasoning, and separate what is known from what is assumed. "
        "Prefer depth and accuracy over speed.",
    ),
    "reviewer": RoomPreset(
        "reviewer", "Reviewer / QA",
        "You are a critical code reviewer and QA engineer. Hunt for bugs, edge cases, "
        "security issues, and untested paths. Be specific: point to the exact problem and "
        "a concrete fix. Default to skepticism.",
    ),
    "architect": RoomPreset(
        "architect", "Architect",
        "You are a software architect. Focus on system design, clean abstractions, clear "
        "boundaries, and long-term maintainability. Evaluate approaches by their trade-offs "
        "and their impact on the whole system.",
    ),
    "decision_maker": RoomPreset(
        "decision_maker", "Decision-Maker",
        "You are a decision-maker and router. Given the input and your rules, decide which "
        "downstream room should handle the work next. Explain your reasoning concisely, then "
        "END your reply with a single line: 'ROUTE: <room name>' naming exactly one connected "
        "room. Be decisive; commit to one destination.",
    ),
}


def get_preset(key: str) -> RoomPreset | None:
    return _PRESETS.get((key or "").strip().lower())


def preset_list() -> list[RoomPreset]:
    """All presets, ``custom`` first."""
    items = list(_PRESETS.values())
    return sorted(items, key=lambda p: (p.key != "custom", p.name))


def compose_context(preset_key: str, rules: str) -> str:
    """Build the system-prompt context for a room: preset prompt + custom rules."""
    parts = []
    preset = get_preset(preset_key)
    if preset and preset.prompt:
        parts.append(preset.prompt)
    if rules and rules.strip():
        parts.append(rules.strip())
    return "\n\n".join(parts)
