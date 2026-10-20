"""Tests for worker subprocess CWD in orchestra.service."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from orchestra import service, config


def test_submit_run_spawns_worker_in_project_root(tmp_path):
    """Test that submit_run spawns job-worker with cwd == config.project_root()."""
    # Setup: mock config.project_root() to return a temp directory
    fake_project_root = tmp_path / "fake_project"
    fake_project_root.mkdir(exist_ok=True)

    # Mock subprocess.Popen to capture the cwd argument
    with patch("orchestra.service.subprocess.Popen") as mock_popen, \
         patch("orchestra.service.config.project_root", return_value=fake_project_root), \
         patch("orchestra.service.get_db") as mock_get_db, \
         patch("orchestra.service.insert_job") as mock_insert_job, \
         patch("orchestra.service.mark_job_run_started") as mock_mark_started:

        # Setup database mocks
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_insert_job.return_value = {"job_id": "test-job-123"}

        # Call submit_run
        result = service.submit_run(
            alias="test-alias",
            task="test task",
            mode="ask"
        )

        # Verify Popen was called with cwd=str(fake_project_root), NOT PACKAGE_ROOT
        assert mock_popen.called, "subprocess.Popen should be called"
        call_kwargs = mock_popen.call_args[1]
        assert "cwd" in call_kwargs, "cwd should be in Popen kwargs"
        assert call_kwargs["cwd"] == str(fake_project_root), \
            f"Expected cwd={fake_project_root}, got {call_kwargs['cwd']}"

        # Ensure it's NOT using a hardcoded path
        # (The old code would have used the PACKAGE_ROOT, which is the orchestra package directory)
        assert fake_project_root in [Path(call_kwargs["cwd"])], \
            "Worker should use config.project_root(), not a hardcoded path"


