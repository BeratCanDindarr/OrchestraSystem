"""The pre-commit hook that runs Orchestra's review gate on the staged changes.

Only a hook Orchestra wrote (it carries MARKER) is replaced or removed: one somebody else wrote
is left as it is. Git's no-verify option skips the gate, as it skips any hook.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MARKER = "# installed by `orchestra hook install`"


def _hook_path(root: Path) -> Path:
    hooks = subprocess.run(["git", "rev-parse", "--git-path", "hooks"], cwd=root, capture_output=True,
                           text=True, check=True).stdout.strip()
    return (root / hooks / "pre-commit").resolve()


def _ours(path: Path) -> bool:
    return path.exists() and MARKER in path.read_text(encoding="utf-8", errors="replace")


def install(root: Path, fail_on: str = "high") -> Path:
    """Write the hook; raises FileExistsError when another pre-commit hook is there."""
    path = _hook_path(root)
    if path.exists() and not _ours(path):
        raise FileExistsError(f"{path} exists and was not written by Orchestra: add the gate to it by hand "
                              f"(orchestra review --staged --fail-on {fail_on})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\n{MARKER}\n'
                    f'# Reviews the staged changes; a checked {fail_on}-or-worse finding stops the commit.\n'
                    f'exec "{sys.executable}" -m orchestra review --staged --fail-on {fail_on}\n', encoding="utf-8")
    path.chmod(0o755)
    return path


def uninstall(root: Path) -> bool:
    """Remove the hook if Orchestra wrote it; True when it did."""
    path = _hook_path(root)
    if not _ours(path):
        return False
    path.unlink()
    return True
