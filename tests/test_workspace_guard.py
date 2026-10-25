"""Tests for WorkspaceGuard."""
from pathlib import Path
from unittest.mock import patch

from orchestra.engine.workspace_guard import WorkspaceGuard, guard_critical_files


def test_guard_disabled_returns_true():
    with patch("orchestra.engine.workspace_guard.config.workspace_guard_enabled", return_value=False):
        assert guard_critical_files() is True


def test_guard_no_changes(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    with patch("orchestra.engine.workspace_guard.config.workspace_guard_enabled", return_value=True):
        with patch("orchestra.engine.workspace_guard.config.workspace_guard_critical_files", return_value=["test.txt"]):
            with patch("orchestra.engine.workspace_guard.config.project_root", return_value=tmp_path):
                guard = WorkspaceGuard()
                assert guard.check() is True


def test_guard_detects_modification(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    with patch("orchestra.engine.workspace_guard.config.workspace_guard_enabled", return_value=True):
        with patch("orchestra.engine.workspace_guard.config.workspace_guard_critical_files", return_value=["test.txt"]):
            with patch("orchestra.engine.workspace_guard.config.project_root", return_value=tmp_path):
                guard = WorkspaceGuard()
                file_path.write_text("modified")
                assert guard.check() is False


def test_guard_detects_deletion(tmp_path: Path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    with patch("orchestra.engine.workspace_guard.config.workspace_guard_enabled", return_value=True):
        with patch("orchestra.engine.workspace_guard.config.workspace_guard_critical_files", return_value=["test.txt"]):
            with patch("orchestra.engine.workspace_guard.config.project_root", return_value=tmp_path):
                guard = WorkspaceGuard()
                file_path.unlink()
                assert guard.check() is False


def test_a_file_that_never_existed_is_not_reported(tmp_path: Path, caplog):
    # The default critical files are Orchestra's own; run from another project they do not
    # exist, and reporting each of them as deleted on every run is noise.

    with patch("orchestra.engine.workspace_guard.config.workspace_guard_enabled", return_value=True):
        with patch("orchestra.engine.workspace_guard.config.workspace_guard_critical_files", return_value=["missing.txt"]):
            with patch("orchestra.engine.workspace_guard.config.project_root", return_value=tmp_path):
                guard = WorkspaceGuard()
                assert guard.check() is True
    assert "deleted" not in caplog.text
