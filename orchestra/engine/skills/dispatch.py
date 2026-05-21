"""Skill dispatch with policy gating, audit trail, and reputation tracking."""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestra.engine.policy_engine import PolicyDecision, PolicyEngine
from orchestra.engine.skills.base import BaseSkill, SkillContext, SkillResult
from orchestra.engine.skills.loader import (
    CoreSkill,
    PromptTemplateSkill,
    load_core_skills,
    load_user_skills,
)
from orchestra.engine.skills.manifest import SkillManifest
from orchestra.events import emit
from orchestra.storage.db import get_db
from orchestra.storage.reputation import record_skill_outcome

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, BaseSkill] | None = None


def _registry() -> dict[str, BaseSkill]:
    """Lazy-loaded unified skill registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = {}
        _REGISTRY.update(load_core_skills())
        _REGISTRY.update(load_user_skills())
    return _REGISTRY


def clear_registry() -> None:
    """Clear the lazy registry (useful in tests)."""
    global _REGISTRY
    _REGISTRY = None


def list_skills(*, tier: str | None = None) -> list[SkillManifest]:
    """Return manifests for all registered skills, optionally filtered by tier.

    Live reputation scores are pulled from the database and overlaid onto each
    manifest so the CLI table shows current values.
    """
    manifests: list[SkillManifest] = []
    try:
        with get_db() as conn:
            for skill in _registry().values():
                if tier is None or skill.manifest.tier == tier:
                    from orchestra.storage.reputation import get_skill_reputation
                    skill.manifest.reputation_score = get_skill_reputation(conn, skill.manifest.name)
                    manifests.append(skill.manifest)
    except sqlite3.Error as exc:
        logger.warning("Failed to enrich skill reputation scores: %s", exc)
        for skill in _registry().values():
            if tier is None or skill.manifest.tier == tier:
                manifests.append(skill.manifest)
    return sorted(manifests, key=lambda m: (m.tier, m.name))


def _log_skill_event(
    event_type: str,
    skill_name: str,
    *,
    actor: str = "orchestra",
    details: dict[str, Any] | None = None,
) -> None:
    """Append an immutable audit event for skill operations."""
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO skill_events (event_type, skill_name, actor, details, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    skill_name,
                    actor,
                    json.dumps(details, ensure_ascii=False) if details else None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.Error as exc:
        logger.warning("Failed to log skill event %s for %s: %s", event_type, skill_name, exc)


def run_skill(name: str, context: SkillContext | None = None, **kwargs: Any) -> SkillResult:
    """Execute a skill by name after policy gating.

    For prompt-template skills (user/marketplace), this dispatches the wrapped
    prompt to the configured lightweight alias instead of executing code locally.
    """
    context = context or SkillContext()
    if context.args:
        for key, value in kwargs.items():
            context.args[key] = value

    # Policy gate
    decision = PolicyEngine.from_default().evaluate_skill(name)
    if decision.action == "deny":
        _log_skill_event("skill_run", name, details={"outcome": "denied", "reason": decision.reason})
        emit("skill", {"event": "skill_denied", "skill_name": name, "reason": decision.reason})
        return SkillResult(
            success=False,
            output=f"Skill '{name}' is denied by policy: {decision.reason}",
        )
    if decision.action == "gate":
        _log_skill_event("skill_run", name, details={"outcome": "gated", "reason": decision.reason})
        emit("skill", {"event": "skill_gated", "skill_name": name, "reason": decision.reason})
        return SkillResult(
            success=False,
            output=f"Skill '{name}' requires approval: {decision.reason}",
        )

    skill = _registry().get(name)
    if skill is None:
        _log_skill_event("skill_run", name, details={"outcome": "not_found"})
        emit("skill", {"event": "skill_not_found", "skill_name": name})
        return SkillResult(success=False, output=f"Skill '{name}' not found")

    _log_skill_event("skill_run", name, details={"tier": skill.manifest.tier})
    emit("skill", {"event": "skill_executed", "skill_name": name, "tier": skill.manifest.tier})

    result: SkillResult
    if isinstance(skill, PromptTemplateSkill):
        # Dispatch template as an agent prompt
        from orchestra.engine.runners import run_ask

        wrapped = f"[Skill: {name}]\n{skill.template}\n\n[Task]\n{context.task}"
        try:
            output, returncode = run_ask(
                context.alias or "cld-fast",
                wrapped,
                emit_console=False,
                show_live=False,
            )
            result = SkillResult(
                success=returncode == 0,
                output=output,
                metrics={"returncode": returncode},
            )
        except Exception as exc:
            logger.exception("PromptTemplateSkill dispatch failed for '%s'", name)
            result = SkillResult(success=False, output=str(exc))
    elif not skill.can_execute(context):
        result = SkillResult(
            success=False,
            output=f"Skill '{name}' cannot execute in this context",
        )
    else:
        result = skill.execute(context)

    # Audit outcome and update reputation
    _log_skill_event(
        "skill_outcome",
        name,
        details={"success": result.success, "metrics": result.metrics},
    )
    emit(
        "skill",
        {
            "event": "skill_completed" if result.success else "skill_failed",
            "skill_name": name,
            "success": result.success,
        },
 )
    try:
        with get_db() as conn:
            record_skill_outcome(conn, name, success=result.success)
    except sqlite3.Error as exc:
        logger.warning("Failed to record skill outcome for %s: %s", name, exc)

    return result


def install_marketplace_skill(name: str) -> SkillResult:
    """Copy a marketplace skill into the user skills directory."""
    from orchestra.engine.skills.loader import _user_skills_dir
    from orchestra.engine.skills.marketplace.index import load_index

    index = load_index()
    manifest = index.get(name)
    if manifest is None:
        return SkillResult(success=False, output=f"Marketplace skill '{name}' not found")

    user_dir = _user_skills_dir()
    target_dir = user_dir / name
    if target_dir.exists():
        return SkillResult(
            success=False,
            output=f"Skill '{name}' is already installed at {target_dir}",
        )

    # For MVP, marketplace skills are prompt-template only (no binary blobs)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest.tier = "user"  # type: ignore[assignment]
    from orchestra.engine.skills.manifest import save_manifest

    save_manifest(target_dir / "manifest.yaml", manifest)

    # Placeholder skill.py — user is expected to refine it
    placeholder = (
        f'"""{manifest.description}\n'
        f'Auto-generated from marketplace skill "{name}" v{manifest.version}.\n'
        f'Edit this file to customize behavior.\n"""\n'
    )
    (target_dir / "skill.py").write_text(placeholder, encoding="utf-8")

    _log_skill_event("skill_installed", name, details={"source": "marketplace", "version": manifest.version})
    emit("skill", {"event": "skill_installed", "skill_name": name, "source": "marketplace"})
    clear_registry()
    return SkillResult(
        success=True,
        output=f"Installed '{name}' v{manifest.version} to {target_dir}",
    )


def uninstall_skill(name: str) -> SkillResult:
    """Remove a user-tier skill from the user skills directory."""
    from orchestra.engine.skills.loader import _user_skills_dir

    # Prevent uninstalling core skills
    if name in load_core_skills():
        return SkillResult(
            success=False,
            output=f"Cannot uninstall core skill '{name}'",
        )

    user_dir = _user_skills_dir()
    target_dir = user_dir / name
    if not target_dir.exists():
        return SkillResult(
            success=False,
            output=f"Skill '{name}' is not installed",
        )

    import shutil
    shutil.rmtree(target_dir)
    _log_skill_event("skill_uninstalled", name)
    emit("skill", {"event": "skill_uninstalled", "skill_name": name})
    clear_registry()
    return SkillResult(
        success=True,
        output=f"Uninstalled '{name}'",
    )
