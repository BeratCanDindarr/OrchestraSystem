"""Unit tests for orchestra.engine.runners.common."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from orchestra.engine.runners.common import (
    _prefixed, 
    _finalize, 
    _is_user_facing_agent,
    _get_tracer,
    _get_pii_scrubber,
    _get_fts_cache
)
from orchestra.models import OrchestraRun, AgentRun, AgentStatus, RunStatus

def test_is_user_facing_agent():
    agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh")
    assert _is_user_facing_agent(agent) is True
    
    agent.alias = "verifier"
    assert _is_user_facing_agent(agent) is False

def test_finalize_completed():
    run = OrchestraRun(mode="ask", run_id="run-1", task="test", created_at="2026-05-21T10:00:00Z")
    agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED, stdout_log="Done")
    run.agents = [agent]
    
    with patch("orchestra.engine.runners.common._runner.artifacts.write_manifest") as mock_write:
        _finalize(run)
        assert run.status == RunStatus.COMPLETED
        assert run.summary == "Done"
        mock_write.assert_called()

def test_finalize_failed():
    run = OrchestraRun(mode="ask", run_id="run-1", task="test", created_at="2026-05-21T10:00:00Z")
    agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.FAILED, stdout_log="Error")
    run.agents = [agent]
    
    with patch("orchestra.engine.runners.common._runner.artifacts.write_manifest") as mock_write:
        _finalize(run)
        assert run.status == RunStatus.FAILED

def test_finalize_multi_agent_provenance():
    run = OrchestraRun(mode="ask", run_id="run-1", task="test", created_at="2026-05-21T10:00:00Z")
    a1 = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED, stdout_log="Output 1")
    a2 = AgentRun(alias="gmn-pro", provider="gemini", model="gemini/pro", status=AgentStatus.COMPLETED, stdout_log="Output 2")
    run.agents = [a1, a2]
    
    with patch("orchestra.engine.runners.common._runner.artifacts.write_manifest"), \
         patch("orchestra.engine.synthesizer.compute_divergence_index", return_value=0.5), \
         patch("orchestra.engine.synthesizer.format_provenance", return_value="Provenance info"):
        _finalize(run)
        assert run.status == RunStatus.COMPLETED
        assert "Output 1" in run.summary
        assert "Provenance info" in run.summary

def test_prefixed_basic():
    run = OrchestraRun(mode="ask", run_id="run-1", task="test", created_at="2026-05-21T10:00:00Z")
    prompt = "Hello"
    
    with patch("orchestra.storage.blackboard.read_notes", return_value=[]), \
         patch("orchestra.engine.context_compression.ContextCompressor.build_prompt", return_value="Prefixed Hello"):
        result = _prefixed("cdx-deep", prompt, run)
        assert result == "Prefixed Hello"

def test_singleton_getters():
    # Test lazy initialization and caching
    with patch("orchestra.engine.runners.common.OtelTracer") as mock_tracer_cls:
        tracer1 = _get_tracer()
        tracer2 = _get_tracer()
        assert tracer1 is tracer2
        
    with patch("orchestra.engine.runners.common.PiiScrubber") as mock_scrubber_cls:
        scrubber1 = _get_pii_scrubber()
        scrubber2 = _get_pii_scrubber()
        assert scrubber1 is scrubber2

    with patch("orchestra.engine.runners.common.FtsSearchCache") as mock_cache_cls:
        cache1 = _get_fts_cache()
        cache2 = _get_fts_cache()
        assert cache1 is cache2

def test_finalize_federation_summary():
    run = OrchestraRun(mode="ask", run_id="run-1", task="test", created_at="2026-05-21T10:00:00Z")
    agent = AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh", status=AgentStatus.COMPLETED, stdout_log="Done")
    run.agents = [agent]
    
    with patch("orchestra.config.federation_config", return_value={"enabled": True}), \
         patch("orchestra.engine.runners.common._runner.artifacts.write_manifest"), \
         patch("orchestra.engine.federation_summarizer.summarize_run", return_value={"summary": "fed"}), \
         patch("orchestra.engine.federation.FederationBus.enqueue_outbox") as mock_enqueue:
        _finalize(run)
        mock_enqueue.assert_called_once_with({"summary": "fed"})


# ── a verdict the runner reached is not recomputed from the agents ───────────

def _answered_run(**fields) -> OrchestraRun:
    run = OrchestraRun(mode="ask", run_id="run-verdict", task="test", created_at="2026-05-21T10:00:00Z", **fields)
    run.agents = [AgentRun(alias="cdx-deep", provider="codex", model="gpt-5.4/xhigh",
                           status=AgentStatus.COMPLETED, stdout_log="the answer")]
    return run


def test_finalize_keeps_a_policy_denial_and_withholds_the_answer():
    from orchestra.state import FailureKind, FailureState

    run = _answered_run(status=RunStatus.FAILED)
    run.failure = FailureState(kind=FailureKind.POLICY_DENIED.value, message="no", retryable=False,
                               source="policy_guard")

    _finalize(run)

    assert run.status == RunStatus.FAILED
    assert not run.summary


def test_finalize_keeps_a_budget_failure_but_reports_the_paid_answer():
    from orchestra.state import FailureKind, FailureState

    run = _answered_run(status=RunStatus.FAILED)
    run.failure = FailureState(kind=FailureKind.BUDGET_EXCEEDED.value, message="over", retryable=False,
                               source="budget_guard")

    _finalize(run)

    assert run.status == RunStatus.FAILED
    assert run.summary == "the answer"


def test_finalize_keeps_a_cancelled_run_cancelled_and_does_not_grade_it():
    run = _answered_run(status=RunStatus.CANCELLED)
    run.agents[0].status = AgentStatus.CANCELLED

    with patch("orchestra.storage.eval_tracker.EvalTracker.log_run") as log_run:
        _finalize(run)

    assert run.status == RunStatus.CANCELLED
    log_run.assert_not_called()


def test_finalize_fails_a_run_in_which_no_agent_ran():
    run = OrchestraRun(mode="planned", run_id="run-empty", task="test", created_at="2026-05-21T10:00:00Z",
                       status=RunStatus.RUNNING)

    _finalize(run)

    assert run.status == RunStatus.FAILED


# ── a run that raises still ends ────────────────────────────────────────────

def _manifest_of(task: str) -> dict:
    import json

    from orchestra import config

    for path in config.artifact_root().glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("task") == task:
            return manifest
    raise AssertionError(f"no run recorded for task {task!r}")


def _assert_ended_as_crashed(task: str, message: str) -> None:
    manifest = _manifest_of(task)
    assert manifest["status"] == "failed"
    assert message in manifest["failure"]["message"]
    assert all(agent["status"] not in ("queued", "started") for agent in manifest["agents"])


def test_an_ask_run_that_raises_is_recorded_as_failed():
    from orchestra.engine.runners import common

    with patch.object(common, "run_parallel", side_effect=RuntimeError("database is locked")):
        with pytest.raises(RuntimeError):
            common._run_ask_impl("cdx-deep", "crash probe: ask")

    _assert_ended_as_crashed("crash probe: ask", "database is locked")


def test_a_dual_run_that_raises_is_recorded_as_failed():
    from orchestra.engine.runners import dual_runner

    with patch.object(dual_runner, "run_parallel", side_effect=RuntimeError("provider exploded")):
        with pytest.raises(RuntimeError):
            dual_runner.run_dual("crash probe: dual", show_live=False, emit_console=False)

    _assert_ended_as_crashed("crash probe: dual", "provider exploded")


def test_a_planned_run_whose_planner_answers_nonsense_is_recorded_as_failed():
    from orchestra.engine.runners import planned_runner

    with patch("orchestra.engine.planner.create_graph_plan", side_effect=ValueError("no JSON in the plan")):
        with pytest.raises(ValueError):
            planned_runner.run_planned("crash probe: planned")

    _assert_ended_as_crashed("crash probe: planned", "no JSON in the plan")
