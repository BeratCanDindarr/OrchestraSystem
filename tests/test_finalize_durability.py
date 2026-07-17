"""Tests for finalize failure isolation (M-11: no transactional/durability guarantee)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from orchestra.models import AgentRun, AgentStatus, OrchestraRun


def _run_with_completed_agent():
    run = OrchestraRun(mode="ask", task="t")
    run.agents = [
        AgentRun(alias="cdx-deep", provider="codex", model="m",
                 status=AgentStatus.COMPLETED, stdout_log="done")
    ]
    return run


def test_finalize_isolates_step_failure_and_still_records_outcome():
    from orchestra.engine.runners.common import _finalize
    run = _run_with_completed_agent()
    events: list[dict] = []
    with patch("orchestra.engine.artifacts.write_manifest", side_effect=RuntimeError("disk full")), \
         patch("orchestra.engine.artifacts.write_checkpoint"), \
         patch("orchestra.engine.replay.get_recorder"), \
         patch("orchestra.storage.memory.get_memory", return_value=MagicMock()), \
         patch("orchestra.storage.eval_tracker.EvalTracker", return_value=MagicMock()), \
         patch("orchestra.engine.runners.common.append_outcome") as mock_outcome, \
         patch("orchestra.engine.runners.common.append_event",
               side_effect=lambda rid, ev: events.append(ev)):
        _finalize(run)   # must NOT raise despite write_manifest failing

    assert mock_outcome.called                    # outcome recorded despite the earlier failure
    assert any(e.get("event") == "finalize_step_failed" and e.get("step") == "write_manifest"
               for e in events)


def test_finalize_happy_path_records_outcome_without_degraded_event():
    from orchestra.engine.runners.common import _finalize
    run = _run_with_completed_agent()
    events: list[dict] = []
    with patch("orchestra.engine.artifacts.write_manifest"), \
         patch("orchestra.engine.artifacts.write_checkpoint"), \
         patch("orchestra.engine.replay.get_recorder"), \
         patch("orchestra.storage.memory.get_memory", return_value=MagicMock()), \
         patch("orchestra.storage.eval_tracker.EvalTracker", return_value=MagicMock()), \
         patch("orchestra.engine.runners.common.append_outcome") as mock_outcome, \
         patch("orchestra.engine.runners.common.append_event",
               side_effect=lambda rid, ev: events.append(ev)):
        _finalize(run)

    assert mock_outcome.called
    assert not any(e.get("event") == "finalize_step_failed" for e in events)
