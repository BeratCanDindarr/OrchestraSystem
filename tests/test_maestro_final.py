import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestra.engine.worktree_manager import WorktreeManager


def test_worktree_create(tmp_path):
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value.returncode = 0
        wtm = WorktreeManager(repo_root=tmp_path)
        wtm.worktrees_root = tmp_path / "wts"
        path = wtm.create("test-branch")
        assert "test-branch" in str(path)


