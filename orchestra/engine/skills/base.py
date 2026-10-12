"""Base abstractions for the tiered skill system."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestra.engine.skills.manifest import SkillManifest


@dataclass
class SkillContext:
    """Runtime context passed to every skill execution."""

    task: str = ""
    project_path: str = "."
    args: dict[str, Any] = field(default_factory=dict)
    # For prompt-template skills (user / marketplace tiers)
    prompt: str = ""
    alias: str = ""


@dataclass
class SkillResult:
    """Outcome of a skill execution."""

    success: bool
    output: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    """Abstract base for all skill implementations."""

    def __init__(self, manifest: "SkillManifest") -> None:
        self.manifest = manifest

    @abstractmethod
    def can_execute(self, context: SkillContext) -> bool:
        """Return True if this skill can handle the given context."""
        ...

    @abstractmethod
    def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill and return a result."""
        ...
