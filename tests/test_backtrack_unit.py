import pytest
from unittest.mock import MagicMock, patch
from orchestra.engine.backtrack import create_snapshot, restore_snapshot, undo_last_run

def test_create_snapshot():
    with patch("orchestra.engine.backtrack._run_git") as mock_git:
        # status (changes found), stash push, stash apply
        mock_git.side_effect = [
            (0, "M file.txt"),
            (0, ""),
            (0, "")
        ]
        assert create_snapshot("run1", "label") is True
        
def test_restore_snapshot():
    with patch("orchestra.engine.backtrack._run_git") as mock_git:
        # restore, clean, stash list, stash apply
        mock_git.side_effect = [
            (0, ""), # restore
            (0, ""), # clean
            (0, "stash@{0}: orchestra_run1_label"), # list
            (0, "")  # apply
        ]
        assert restore_snapshot("run1", "label") is True

def test_restore_snapshot_not_found():
    with patch("orchestra.engine.backtrack._run_git") as mock_git:
        mock_git.side_effect = [
            (0, ""), # restore
            (0, ""), # clean
            (0, "stash@{0}: some_other_stash") # list
        ]
        assert restore_snapshot("run1", "label") is False

def test_undo_last_run():
    with patch("orchestra.engine.backtrack.restore_snapshot") as mock_restore:
        mock_restore.return_value = True
        assert undo_last_run("run1") is True
        mock_restore.assert_called_with("run1", "initial")
