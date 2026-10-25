"""Tests for live_status.install() wiring in cancel_run / continue_run,
and for approval_resumed / approval_rejected dispatch in LiveStatusWriter.

Run from a scratch directory:
    cd /tmp && /abs/.venv/bin/pytest /abs/tests/test_service_live_status.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(name: str, **kwargs) -> dict:
    return {"event": name, "ts": datetime.now(timezone.utc).isoformat(), **kwargs}


def _emit_sync(run_id: str, event: dict, writer) -> None:
    """Call the bus handler synchronously (bypasses thread-pool)."""
    writer._handle(run_id, event)


# ---------------------------------------------------------------------------
# Fixture: reset live_status singleton between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_live_status(tmp_path, monkeypatch):
    import orchestra.live_status as ls
    import orchestra.events.bus as bus

    monkeypatch.setattr("orchestra.config.orchestra_root", lambda: tmp_path)
    ls._writer = None
    bus.clear()
    if bus.log_subscriber not in bus._global:
        bus.subscribe_all(bus.log_subscriber)

    yield

    ls._writer = None
    bus.clear()
    if bus.log_subscriber not in bus._global:
        bus.subscribe_all(bus.log_subscriber)


# ---------------------------------------------------------------------------
# Part A: LiveStatusWriter._dispatch handles approval_resumed / approval_rejected
# ---------------------------------------------------------------------------

class TestApprovalEventHandlers:
    """Tests 1-3: writer-level dispatch for the new event names."""

    def _make_writer(self):
        from orchestra.live_status import LiveStatusWriter
        return LiveStatusWriter()

    # Test 1: approval_resumed → status "running", agents preserved
    def test_approval_resumed_sets_status_running(self, tmp_path):
        """A tracked run in waiting_approval + approval_resumed → status becomes 'running'."""
        w = self._make_writer()
        run_id = "ar-1"

        _emit_sync(run_id, _make_event("run_started", mode="critical"), w)
        _emit_sync(run_id, _make_event("agent_started", alias="ag1", model="m"), w)
        _emit_sync(run_id, _make_event("approval_requested"), w)

        # Precondition: status is waiting_approval
        import json
        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"][run_id]["status"] == "waiting_approval"

        # Now emit approval_resumed
        _emit_sync(run_id, _make_event("approval_resumed"), w)

        data2 = json.loads((tmp_path / "live_status.json").read_text())
        run = data2["runs"][run_id]
        assert run["status"] == "running", (
            f"approval_resumed must flip status to 'running', got {run['status']!r}"
        )
        # agents dict must be preserved (not wiped)
        assert "ag1" in run["agents"], "agents must be preserved after approval_resumed"

    # Test 2: approval_rejected → status "cancelled"
    def test_approval_rejected_sets_status_cancelled(self, tmp_path):
        """A tracked run in waiting_approval + approval_rejected → status becomes 'cancelled'."""
        w = self._make_writer()
        run_id = "ar-2"

        _emit_sync(run_id, _make_event("run_started", mode="critical"), w)
        _emit_sync(run_id, _make_event("approval_requested"), w)

        # Precondition: status is waiting_approval
        import json
        data = json.loads((tmp_path / "live_status.json").read_text())
        assert data["runs"][run_id]["status"] == "waiting_approval"

        # Now emit approval_rejected
        _emit_sync(run_id, _make_event("approval_rejected"), w)

        data2 = json.loads((tmp_path / "live_status.json").read_text())
        run = data2["runs"][run_id]
        assert run["status"] == "cancelled", (
            f"approval_rejected must flip status to 'cancelled', got {run['status']!r}"
        )

    # Test 3: untracked run → no run entry created, no crash
    def test_approval_events_untracked_run_no_crash_no_entry(self, tmp_path):
        """approval_resumed/rejected for untracked run must not crash and not create entry."""
        w = self._make_writer()
        untracked_id = "untracked-xyz"

        # Neither event should crash or create a run entry
        _emit_sync(untracked_id, _make_event("approval_resumed"), w)
        _emit_sync(untracked_id, _make_event("approval_rejected"), w)

        import json
        status_path = tmp_path / "live_status.json"
        if status_path.exists():
            data = json.loads(status_path.read_text())
            assert untracked_id not in data.get("runs", {}), (
                "No run entry must be created for untracked run on approval events"
            )


# ---------------------------------------------------------------------------
# Part B: install() wiring at cancel_run / continue_run entry points
# ---------------------------------------------------------------------------

class TestCancelRunInstall:
    """Test 4: cancel_run calls live_status.install()."""

    def test_cancel_run_calls_install(self, tmp_path):
        """cancel_run must call live_status.install() before any other work.

        Setup: monkeypatch artifacts.load_manifest to return a completed-status
        manifest so cancel_run hits the 'already_finished' early-return path
        without needing any real filesystem or process state.
        """
        import orchestra.service as svc

        completed_manifest = {
            "run_id": "cancel-test-1",
            "status": "completed",
            "agents": [],
        }

        with (
            patch("orchestra.live_status.install") as mock_install,
            patch.object(svc.artifacts, "load_manifest", return_value=completed_manifest),
        ):
            result = svc.cancel_run("cancel-test-1")

        mock_install.assert_called_once(), "cancel_run must call live_status.install()"
        assert result["already_finished"] is True


class TestContinueRunInstall:
    """Test 5: continue_run calls live_status.install()."""

    def test_continue_run_calls_install(self, tmp_path):
        """continue_run must call live_status.install() before delegating to resume_run.

        Setup: monkeypatch resume_run to return a fake OrchestraRun-like object
        with a to_manifest() method, and monkeypatch live_status.install.
        """
        import orchestra.service as svc

        fake_manifest = {"run_id": "cont-test-1", "status": "completed"}
        fake_run = MagicMock()
        fake_run.to_manifest.return_value = fake_manifest

        with (
            patch("orchestra.live_status.install") as mock_install,
            patch.object(svc, "resume_run", return_value=fake_run) as mock_resume,
        ):
            result = svc.continue_run("cont-test-1")

        mock_install.assert_called_once(), "continue_run must call live_status.install()"
        mock_resume.assert_called_once()
        assert result["run"]["run_id"] == "cont-test-1"
