"""Workspace Guard — detects changes to critical files during execution."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

from orchestra import config

logger = logging.getLogger(__name__)


class WorkspaceGuard:
    """Records and checks the state of critical workspace files."""

    def __init__(self) -> None:
        self.enabled: bool = config.workspace_guard_enabled()
        self.critical_files: list[str] = config.workspace_guard_critical_files()
        self._snapshot: Dict[str, Tuple[float, int]] = {}
        if self.enabled:
            for rel_path in self.critical_files:
                full_path = config.project_root() / rel_path
                try:
                    stat = full_path.stat()
                    self._snapshot[str(rel_path)] = (stat.st_mtime, stat.st_size)
                except FileNotFoundError:
                    # File doesn't exist at init time — record as missing
                    self._snapshot[str(rel_path)] = (0.0, 0)
                except OSError as exc:
                    logger.warning(f"WorkspaceGuard: unable to stat {rel_path}: {exc}")
                    self._snapshot[str(rel_path)] = (0.0, 0)

    def check(self) -> bool:
        """Compare current file state to the recorded snapshot.

        Returns:
            True if all critical files are intact, False otherwise.
        """
        if not self.enabled:
            return True

        all_intact = True
        for rel_path in self.critical_files:
            recorded = self._snapshot.get(str(rel_path), (0.0, 0))
            full_path = config.project_root() / rel_path
            try:
                stat = full_path.stat()
                current = (stat.st_mtime, stat.st_size)
                if current != recorded:
                    logger.warning(
                        f"WorkspaceGuard: critical file modified: {rel_path} "
                        f"(mtime/size changed)"
                    )
                    all_intact = False
            except FileNotFoundError:
                logger.error(f"WorkspaceGuard: critical file deleted: {rel_path}")
                all_intact = False
            except OSError as exc:
                logger.warning(f"WorkspaceGuard: unable to check {rel_path}: {exc}")
                all_intact = False

        return all_intact


def guard_critical_files() -> bool:
    """Convenience wrapper: create a WorkspaceGuard and run check().

    Returns:
        True if all critical files are intact (or guard is disabled).
    """
    if not config.workspace_guard_enabled():
        return True

    guard = WorkspaceGuard()
    intact = guard.check()
    if not intact:
        # Non-fatal: log only, do not raise or stop execution
        logger.warning("WorkspaceGuard: changes detected in critical files (continuing)")
    return intact
