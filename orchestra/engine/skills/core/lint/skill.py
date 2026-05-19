"""Core skill: run Python linters on the project."""
from __future__ import annotations

import shutil
import subprocess

from orchestra.engine.skills.base import SkillContext, SkillResult


def can_execute(context: SkillContext) -> bool:
    return shutil.which("ruff") is not None or shutil.which("flake8") is not None


def execute(context: SkillContext) -> SkillResult:
    project_path = context.args.get("path", context.project_path)
    cmd: list[str] | None = None
    tool = ""

    if shutil.which("ruff"):
        cmd = ["ruff", "check", project_path]
        tool = "ruff"
    elif shutil.which("flake8"):
        cmd = ["flake8", project_path]
        tool = "flake8"

    if cmd is None:
        return SkillResult(
            success=False,
            output="No Python linter found (install ruff or flake8).",
        )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return SkillResult(
        success=result.returncode == 0,
        output=result.stdout or result.stderr or f"{tool} completed with no output.",
        metrics={"tool": tool, "returncode": result.returncode},
    )
