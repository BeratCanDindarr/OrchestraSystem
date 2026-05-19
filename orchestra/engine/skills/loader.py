"""Skill loader for core and user tiers."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from orchestra import config
from orchestra.engine.skills.base import BaseSkill, SkillContext, SkillResult
from orchestra.engine.skills.manifest import SkillManifest, load_manifest

logger = logging.getLogger(__name__)


def _core_skills_dir() -> Path:
    """Directory containing built-in core skills."""
    return Path(__file__).parent / "core"


def _user_skills_dir() -> Path:
    """Directory for dynamically taught user skills."""
    return config.orchestra_root() / "skills" / "user"


class CoreSkill(BaseSkill):
    """A static skill backed by a Python entrypoint module."""

    def __init__(self, manifest: SkillManifest, entrypoint_path: Path) -> None:
        super().__init__(manifest)
        self._entrypoint_path = entrypoint_path
        self._module: Any = None

    def _load_module(self) -> Any:
        if self._module is not None:
            return self._module
        spec = importlib.util.spec_from_file_location(
            f"orchestra.engine.skills.core.{self.manifest.name}",
            self._entrypoint_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill module {self._entrypoint_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._module = mod
        return mod

    def can_execute(self, context: SkillContext) -> bool:
        try:
            mod = self._load_module()
            if hasattr(mod, "can_execute"):
                return mod.can_execute(context)
        except Exception:
            return False
        return True

    def execute(self, context: SkillContext) -> SkillResult:
        try:
            mod = self._load_module()
            if hasattr(mod, "execute"):
                return mod.execute(context)
            return SkillResult(
                success=False,
                output=f"Skill '{self.manifest.name}' has no execute() function",
            )
        except Exception as exc:
            logger.exception("Core skill '%s' execution failed", self.manifest.name)
            return SkillResult(
                success=False,
                output=str(exc),
                metrics={"error": type(exc).__name__},
            )


class PromptTemplateSkill(BaseSkill):
    """A dynamic skill that injects a prompt template into an agent run."""

    def __init__(self, manifest: SkillManifest, template: str) -> None:
        super().__init__(manifest)
        self.template = template

    def can_execute(self, context: SkillContext) -> bool:
        return True

    def execute(self, context: SkillContext) -> SkillResult:
        """Return the wrapped prompt; actual dispatch happens in dispatch.py."""
        return SkillResult(
            success=True,
            output=self.template,
            metrics={"skill_name": self.manifest.name},
        )


def _scan_manifests(directory: Path) -> dict[str, SkillManifest]:
    """Discover and load all manifests in *directory*."""
    manifests: dict[str, SkillManifest] = {}
    if not directory.exists():
        return manifests
    for manifest_path in directory.glob("*/manifest.yaml"):
        manifest = load_manifest(manifest_path)
        if manifest is not None:
            manifests[manifest.name] = manifest
    return manifests


def load_core_skills() -> dict[str, BaseSkill]:
    """Load all valid core skills."""
    skills: dict[str, BaseSkill] = {}
    for name, manifest in _scan_manifests(_core_skills_dir()).items():
        if manifest.tier != "core":
            continue
        entry_path = _core_skills_dir() / name / manifest.entrypoint
        if not entry_path.exists():
            logger.warning("Core skill '%s' entrypoint missing", name)
            continue
        skills[name] = CoreSkill(manifest, entry_path)
    return skills


def load_user_skills() -> dict[str, BaseSkill]:
    """Load all user-taught skills (prompt templates)."""
    skills: dict[str, BaseSkill] = {}
    user_dir = _user_skills_dir()
    if not user_dir.exists():
        return skills
    for name, manifest in _scan_manifests(user_dir).items():
        if manifest.tier != "user":
            continue
        template_path = user_dir / name / "skill.py"
        template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
        skills[name] = PromptTemplateSkill(manifest, template)
    return skills


def load_marketplace_skills() -> dict[str, SkillManifest]:
    """Return manifest-only view of marketplace skills (not installed)."""
    from orchestra.engine.skills.marketplace import index as _index

    return _index.load_index()
