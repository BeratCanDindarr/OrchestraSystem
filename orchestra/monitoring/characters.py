from __future__ import annotations
"""The model fleet as a roster of characters.

Turns Orchestra's alias table into a cast of named personas — each model a
character with a human name, a role archetype, and a deterministic generated
avatar — so the dashboard can present the fleet as a team you dispatch work to.
Personas are auto-derived from the alias/provider/tier and overridable via
``[characters.<alias>]`` config.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional

from orchestra import config

# A pool of human first names, assigned deterministically so each model keeps a
# stable identity across restarts.
_NAME_POOL = [
    "Ada", "Bo", "Cy", "Dex", "Echo", "Fin", "Gale", "Hana", "Iris", "Jax",
    "Kit", "Lena", "Mika", "Nova", "Orin", "Pax", "Quinn", "Rex", "Sage", "Tao",
    "Uma", "Vera", "Wren", "Xan", "Yuki", "Zed", "Ari", "Bex", "Cleo", "Dane",
]

# provider -> role archetype
_ARCHETYPE = {
    "codex": "Coder",
    "openai_sdk": "Coder",
    "openai": "Coder",
    "azure_openai": "Coder",
    "gemini": "Researcher",
    "claude": "Advisor",
    "claude_sdk": "Advisor",
    "ollama": "Local Specialist",
    "deepseek": "Reasoner",
    "perplexity": "Scout",
    "remote": "Remote Agent",
}

# effort/model tokens that hint at a "deep" vs "fast" tier
_DEEP_HINTS = ("deep", "xhigh", "high", "pro", "opus", "reasoner", "reasoning", "o3", "llama3", "kimi", "minimax")
_FAST_HINTS = ("fast", "low", "flash", "mini", "lite", "sonar", "coder")


@dataclass
class Character:
    id: str          # the alias
    name: str        # human display name
    role: str        # role archetype + tier, e.g. "Deep Coder"
    provider: str    # provider name, e.g. "codex"
    model: str       # effort/model token
    available: bool  # provider.is_available()
    avatar: str      # inline SVG


def _tier(effort: str) -> str:
    e = str(effort).lower()
    if any(h in e for h in _DEEP_HINTS):
        return "Deep"
    if any(h in e for h in _FAST_HINTS):
        return "Fast"
    return ""


def _persona_for(alias: str, provider: str, effort: str) -> dict:
    """Auto-derive a persona, overridable via ``[characters.<alias>]`` config."""
    overrides = config.section("characters").get(alias, {})
    if not isinstance(overrides, dict):
        overrides = {}
    archetype = _ARCHETYPE.get(provider, "Agent")
    tier = _tier(effort)
    role = f"{tier} {archetype}".strip()
    # stable name: pick from the pool by the alias's sorted position (avoids hash collisions)
    idx = _stable_index(alias)
    name = _NAME_POOL[idx % len(_NAME_POOL)]
    return {"name": overrides.get("name", name), "role": overrides.get("role", role)}


def _stable_index(alias: str) -> int:
    """Deterministic small integer from an alias (stable across processes)."""
    return int(hashlib.sha256(alias.encode("utf-8")).hexdigest()[:6], 16)


def avatar_svg(seed: str, *, size: int = 44) -> str:
    """Deterministic geometric avatar for a character seed (no dependencies)."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    hue = int(h[:2], 16) * 360 // 255
    hue2 = (hue + 140) % 360
    eye_y = size * 0.4
    mouth_w = size * (0.25 + (int(h[2:4], 16) / 255) * 0.2)
    cx = size / 2
    return (
        f'<svg class="avatar" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cx}" r="{size/2}" fill="hsl({hue},55%,45%)"/>'
        f'<circle cx="{size*0.35:.1f}" cy="{eye_y:.1f}" r="{size*0.07:.1f}" fill="hsl({hue2},70%,88%)"/>'
        f'<circle cx="{size*0.65:.1f}" cy="{eye_y:.1f}" r="{size*0.07:.1f}" fill="hsl({hue2},70%,88%)"/>'
        f'<rect x="{cx-mouth_w/2:.1f}" y="{size*0.62:.1f}" width="{mouth_w:.1f}" height="{size*0.06:.1f}" '
        f'rx="{size*0.03:.1f}" fill="hsl({hue2},70%,88%)"/>'
        f'</svg>'
    )


def build_roster() -> list[Character]:
    """Build the full character roster from the alias table."""
    from orchestra.providers.fallback import ALIAS_TABLE
    roster: list[Character] = []
    for alias, (provider, effort) in ALIAS_TABLE.items():
        pname = getattr(provider, "name", type(provider).__name__)
        try:
            available = bool(provider.is_available())
        except Exception:
            available = False
        persona = _persona_for(alias, pname, str(effort))
        roster.append(Character(
            id=alias, name=persona["name"], role=persona["role"],
            provider=pname, model=str(effort), available=available,
            avatar=avatar_svg(alias),
        ))
    return roster
