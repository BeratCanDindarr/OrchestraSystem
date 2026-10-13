"""A workspace of its own for an agent that may write.

With `[isolation] enabled = true`, a writing alias (cld-write by default) runs in a git
worktree checked out at HEAD instead of in the project. What it changed is collected as a
patch in the run's folder and the worktree is removed; the project is touched only when the
owner runs `orchestra apply <run_id>` (or never: `orchestra discard <run_id>`).

The workspace holds what is committed: uncommitted work in the project is not in it. Without
a git project there is no workspace, and the run fails rather than write into the project
after all.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from orchestra import config

logger = logging.getLogger(__name__)

PATCH_NAME = "changes.patch"
DEFAULT_ALIASES = ("cld-write",)
_GIT_TIMEOUT_SECONDS = 120


class IsolationError(RuntimeError):
    """The workspace could not be made, or its changes could not be applied."""


def applies_to(alias: str) -> bool:
    """Whether a run on *alias* gets a workspace: isolation is on and the alias writes here.

    `alias@node` runs on another machine, on that machine's files.
    """
    settings = config.isolation_config()
    if not settings["enabled"] or "@" in alias:
        return False
    return alias.split("#", 1)[0] in settings["aliases"]


def _git(directory: Path, *args: str, check: bool = True, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        done = subprocess.run(["git", "-C", str(directory), *args], input=stdin, capture_output=True,
                              text=True, timeout=_GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        raise IsolationError(f"git {args[0]} could not run: {error}") from error
    if check and done.returncode != 0:
        raise IsolationError(f"git {' '.join(args[:2])} failed: {(done.stderr or done.stdout).strip()[:400]}")
    return done


def _workspace_path(run_id: str) -> Path:
    return config.orchestra_root() / "worktrees" / run_id


def _patch_path(run_id: str) -> Path:
    return config.artifact_root() / run_id / PATCH_NAME


def open_workspace(run_id: str) -> Path:
    """A worktree of the project at HEAD for one run. Raises IsolationError when it cannot be made."""
    project = config.project_root()
    if _git(project, "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        raise IsolationError(
            f"{project} is not a git project: an isolated workspace needs one. "
            "Switch [isolation] off to let the agent write into the project directly."
        )
    path = _workspace_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(project, "worktree", "add", "--detach", str(path), "HEAD")
    return path


def close_workspace(run_id: str, workspace: Path) -> str:
    """Collect what changed in the workspace as a patch, remove the workspace, return the patch.

    The patch is written to the run's folder; nothing is written when nothing changed.
    """
    try:
        _git(workspace, "add", "-A")
        patch_text = _git(workspace, "diff", "--cached", "--binary", "HEAD").stdout
    finally:
        _remove(workspace)
    if patch_text.strip():
        target = _patch_path(run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(patch_text, encoding="utf-8")
    return patch_text


def _remove(workspace: Path) -> None:
    project = config.project_root()
    removed = _git(project, "worktree", "remove", "--force", str(workspace), check=False)
    if removed.returncode != 0 or workspace.exists():
        logger.warning("worktree %s was not removed by git: %s", workspace, removed.stderr.strip()[:200])
        shutil.rmtree(workspace, ignore_errors=True)
        _git(project, "worktree", "prune", check=False)


def pending_changes(run_id: str) -> Optional[Path]:
    """The patch a run left behind, if it has not been applied or discarded."""
    path = _patch_path(run_id)
    return path if path.is_file() else None


def apply_changes(run_id: str) -> None:
    """Apply a run's patch to the project. All of it or none of it.

    Raises IsolationError when there is no patch, or when it no longer fits the project
    (the patch is kept so it can be applied by hand).
    """
    patch_path = pending_changes(run_id)
    if patch_path is None:
        raise IsolationError(f"Run {run_id} has no changes waiting to be applied.")
    project = config.project_root()
    fits = _git(project, "apply", "--check", str(patch_path), check=False)
    if fits.returncode != 0:
        raise IsolationError(
            f"The changes of run {run_id} no longer fit the project: {fits.stderr.strip()[:400]}\n"
            f"The patch is kept at {patch_path}."
        )
    _git(project, "apply", str(patch_path))
    patch_path.unlink()


def discard_changes(run_id: str) -> None:
    """Throw a run's patch away. Raises IsolationError when there is none."""
    patch_path = pending_changes(run_id)
    if patch_path is None:
        raise IsolationError(f"Run {run_id} has no changes waiting.")
    patch_path.unlink()


def how_to_apply(run_id: str, patch_text: str) -> str:
    """The lines a run's summary gets when it left changes behind."""
    files = sorted({line.split(" b/", 1)[1] for line in patch_text.splitlines()
                    if line.startswith("diff --git ") and " b/" in line})
    listed = ", ".join(files[:8]) + (f" and {len(files) - 8} more" if len(files) > 8 else "")
    return (f"Changes to {listed or 'the project'} were made in an isolated workspace and are not in "
            f"the project yet.\nApply them: orchestra apply {run_id}    Discard them: orchestra discard {run_id}")
