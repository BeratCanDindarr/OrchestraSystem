from __future__ import annotations
import os
import shutil
import subprocess
import logging
from pathlib import Path

from orchestra import config

logger = logging.getLogger(__name__)

class WorktreeManager:
    def __init__(self, repo_root: str | Path = None):
        self.repo_root = Path(repo_root or config.project_root()).resolve()
        self.worktrees_root = config.orchestra_root() / "worktrees"
        self.worktrees_root.mkdir(parents=True, exist_ok=True)

    def create(self, branch: str, base_ref: str = "HEAD") -> Path:
        """Create a new git worktree for the given branch."""
        wt_path = self.worktrees_root / branch
        
        # Cleanup if it exists but isn't valid
        if wt_path.exists():
            shutil.rmtree(wt_path)
            subprocess.run(["git", "-C", str(self.repo_root), "worktree", "prune"], capture_output=True)

        logger.info(f"Creating worktree at {wt_path} for branch {branch} from {base_ref}")
        res = subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "add", "-b", branch, str(wt_path), base_ref],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            # Maybe branch already exists?
            res = subprocess.run(
                ["git", "-C", str(self.repo_root), "worktree", "add", str(wt_path), branch],
                capture_output=True, text=True
            )
            if res.returncode != 0:
                raise RuntimeError(f"Failed to create worktree: {res.stderr}")
        return wt_path

    def cleanup(self, branch: str) -> None:
        """Remove a git worktree."""
        wt_path = self.worktrees_root / branch
        if wt_path.exists():
            subprocess.run(["git", "-C", str(self.repo_root), "worktree", "remove", "-f", str(wt_path)], capture_output=True)
            if wt_path.exists():
                shutil.rmtree(wt_path)
            subprocess.run(["git", "-C", str(self.repo_root), "worktree", "prune"], capture_output=True)
            logger.info(f"Cleaned up worktree {wt_path}")

    def list_worktrees(self) -> list[str]:
        """List active Orchestra worktrees."""
        if not self.worktrees_root.exists():
            return []
        return [d.name for d in self.worktrees_root.iterdir() if d.is_dir()]

    def prune_stale(self, max_limit: int = 5) -> None:
        """Keep only the most recent N worktrees."""
        worktrees = []
        for d in self.worktrees_root.iterdir():
            if d.is_dir():
                worktrees.append((d, d.stat().st_mtime))
                
        worktrees.sort(key=lambda x: x[1], reverse=True)
        for wt, _ in worktrees[max_limit:]:
            self.cleanup(wt.name)

    def create_pr(self, branch: str, title: str, body: str) -> str:
        """Create a PR using gh CLI."""
        wt_path = self.worktrees_root / branch
        if not wt_path.exists():
            raise FileNotFoundError(f"Worktree {wt_path} not found")

        # Ensure changes are committed
        subprocess.run(["git", "-C", str(wt_path), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(wt_path), "commit", "-m", title], capture_output=True)
        
        # Push branch
        res = subprocess.run(["git", "-C", str(wt_path), "push", "-u", "origin", branch], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to push branch: {res.stderr}")

        # Create PR
        res = subprocess.run(["gh", "pr", "create", "--title", title, "--body", body], cwd=str(wt_path), capture_output=True, text=True)
        if res.returncode == 0:
            logger.info(f"PR created: {res.stdout.strip()}")
            return res.stdout.strip()
        else:
            raise RuntimeError(f"Failed to create PR: {res.stderr}")

