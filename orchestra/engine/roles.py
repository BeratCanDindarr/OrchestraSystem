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

def get_role(name: str) -> Role | None:
    for r_name, r in ROLES.items():
        if r_name.lower() == name.lower():
            return r
    return None
