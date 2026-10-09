from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

@dataclass
class Role:
    name: str
    description: str
    default_alias: str

ROLES: Dict[str, Role] = {
    "Architect": Role(
        "Architect",
        "Focuses on system design, clean abstractions, and long-term maintainability.",
        "gmn-pro"
    ),
    "QA": Role(
        "QA",
        "Focuses on edge cases, potential failures, and testability.",
        "cld-fast"
    ),
    "Lead": Role(
        "Lead",
        "Synthesizes discussions, resolves conflicts, and provides concrete implementations.",
        "cdx-deep"
    )
}

# Which configured persona plays each debate role. The fixed aliases above are the fallback:
# they name providers (Gemini, Codex) that may be switched off or logged out.
_ROLE_PERSONAS = {"Architect": "researcher", "QA": "reviewer", "Lead": "synthesizer"}


def alias_for_role(name: str) -> str:
    from orchestra import config

    role = ROLES.get(name)
    persona = config.personas_config().get(_ROLE_PERSONAS.get(name, ""))
    return str(persona or (role.default_alias if role else "cld-fast"))


def get_role(name: str) -> Role | None:
    for r_name, r in ROLES.items():
        if r_name.lower() == name.lower():
            return r
    return None
