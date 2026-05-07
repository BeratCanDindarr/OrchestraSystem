"""Backtracking system: capture and restore file system snapshots using Git."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Optional

def _run_git(args: list[str]) -> tuple[int, str]:
    try:
        res = subprocess.run(["git"] + args, capture_output=True, text=True)
        return res.returncode, res.stdout.strip()
    except Exception as e:
        return 1, str(e)

def create_snapshot(run_id: str, label: str) -> bool:
    """Save current working directory state to a stash."""
    msg = f"orchestra_{run_id}_{label}"
    # Check if there are changes to stash
    rc, status = _run_git(["status", "--porcelain"])
    if not status:
        return True # No changes to save

    # Create stash entry
    # We use push --include-untracked to be safe
    rc, out = _run_git(["stash", "push", "--include-untracked", "-m", msg])
    if rc == 0:
        # Immediately bring changes back so execution continues
        _run_git(["stash", "apply", "stash@{0}"])
        return True
    return False

def restore_snapshot(run_id: str, label: str) -> bool:
    """Revert working directory to a specific snapshot."""
    msg = f"orchestra_{run_id}_{label}"
    
    # 1. Discard current dirty changes
    _run_git(["restore", "."])
    _run_git(["clean", "-fd"])
    
    # 2. Find the stash index with the matching message
    rc, out = _run_git(["stash", "list"])
    for line in out.splitlines():
        if msg in line:
            index = line.split(":")[0].strip()
            # 3. Apply the stash
            rc_pop, _ = _run_git(["stash", "apply", index])
            return rc_pop == 0
            
    return False

def undo_last_run(run_id: str) -> bool:
    """Special case: Revert all changes made by a specific run."""
    # We look for the 'initial' snapshot of that run
    return restore_snapshot(run_id, "initial")

