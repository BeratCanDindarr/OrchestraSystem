"""Tests for live_status wiring: entry points + per-agent stream hook (Task 21).

Run from a scratch directory to avoid CWD pollution:
    cd /tmp && /abs/path/.venv/bin/pytest /abs/path/tests/test_live_status_wiring.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

PARALLEL_DIR = Path(__file__).resolve().parent.parent / "orchestra" / "engine" / "parallel"


# ---------------------------------------------------------------------------
# Test 1: service.run_task calls live_status.install()
# ---------------------------------------------------------------------------

class TestRunTaskInstall:
    """Source-level test: run_task must call live_status.install() before routing."""

    def test_run_task_calls_install_on_ask_mode(self):
        """Call run_task with mode='ask' (heavily mocked), assert install() was invoked."""
        from orchestra.models import OrchestraRun, RunStatus
        import orchestra.service as svc

        fake_run = OrchestraRun(
            run_id="test-rt-1",
            mode="ask",
            task="hello",
            created_at="2026-01-01T00:00:00+00:00",
        )
        fake_run.status = RunStatus.COMPLETED

        with (
            patch("orchestra.live_status.install") as mock_install,
            patch.object(svc, "run_ask", return_value=fake_run),
            patch.object(svc, "route_task") as mock_route,
        ):
            mock_route.return_value = MagicMock(alias="cdx-fast", mode="ask")
            result = svc.run_task(mode="ask", task="hello", alias="cdx-fast")

        mock_install.assert_called_once()
        assert result["run"]["run_id"] == "test-rt-1"

    def test_run_task_calls_install_on_dual_mode(self):
        """run_task with mode='dual' must also call live_status.install()."""
        from orchestra.models import OrchestraRun, RunStatus
        import orchestra.service as svc

        fake_run = OrchestraRun(
            run_id="test-rt-2",
            mode="dual",
            task="hello",
            created_at="2026-01-01T00:00:00+00:00",
        )
        fake_run.status = RunStatus.COMPLETED

        with (
            patch("orchestra.live_status.install") as mock_install,
            patch.object(svc, "run_dual", return_value=fake_run),
        ):
            result = svc.run_task(mode="dual", task="hello")

        mock_install.assert_called_once()

    def test_run_task_install_called_before_routing(self):
        """install() must be called *before* the routing/runner code runs.

        Verified by injecting a side-effect that records call order.
        """
        from orchestra.models import OrchestraRun, RunStatus
        import orchestra.service as svc

        call_order: list[str] = []

        fake_run = OrchestraRun(
            run_id="test-rt-3",
            mode="ask",
            task="hello",
            created_at="2026-01-01T00:00:00+00:00",
        )
        fake_run.status = RunStatus.COMPLETED

        def record_install():
            call_order.append("install")

        def record_run_ask(*args, **kwargs):
            call_order.append("run_ask")
            return fake_run

        with (
            patch("orchestra.live_status.install", side_effect=record_install),
            patch.object(svc, "run_ask", side_effect=record_run_ask),
        ):
            svc.run_task(mode="ask", task="hello", alias="cdx-fast")

        assert "install" in call_order
        assert "run_ask" in call_order
        assert call_order.index("install") < call_order.index("run_ask"), (
            f"install() must be called before run_ask(); got order: {call_order}"
        )


# ---------------------------------------------------------------------------
# Test 2: service.execute_job calls live_status.install()
# ---------------------------------------------------------------------------

class TestExecuteJobInstall:
    """execute_job delegates to run_task which calls install(); verify the chain."""

    def test_execute_job_calls_install_via_run_task(self):
        """execute_job → run_task → live_status.install() must be reachable.

        Strategy: mock the DB fetch and mark_* functions to return a claimed job,
        then mock run_task (service-level) to capture install invocations.
        """
        import orchestra.service as svc
        from orchestra.models import OrchestraRun, RunStatus

        fake_run = OrchestraRun(
            run_id="job-run-1",
            mode="ask",
            task="job task",
            created_at="2026-01-01T00:00:00+00:00",
        )
        fake_run.status = RunStatus.COMPLETED

        fake_job = {
            "job_id": "job-1",
            "mode": "ask",
            "task": "job task",
            "alias": "cdx-fast",
            "pause_after_round1": 0,
            "run_id": "job-run-1",
        }

        fake_conn = MagicMock()
        fake_conn.__enter__ = lambda s: s
        fake_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch("orchestra.live_status.install") as mock_install,
            patch.object(svc, "get_db", return_value=fake_conn),
            patch.object(svc, "claim_job", return_value=True),
            patch.object(svc, "get_job_record", return_value=fake_job),
            patch.object(svc, "run_task", return_value={"run": fake_run.to_manifest()}) as mock_run_task,
            patch.object(svc, "mark_job_run_started"),
            patch.object(svc, "mark_job_completed"),
            patch.object(svc, "get_job", return_value={"job": fake_job}),
        ):
            result = svc.execute_job("job-1")

        mock_install.assert_called_once()
        mock_run_task.assert_called_once()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 3b: _run_one_agent_async stream hook — AST verification twin
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 3c: _run_one_agent_async (async path) stream hook calls note_stream
# ---------------------------------------------------------------------------

class TestRunOneAgentAsyncStreamHook:
    """_run_one_agent_async must also call live_status.note_stream for each chunk."""

    def _make_run(self, run_id: str = "async-hook-1"):
        from orchestra.models import OrchestraRun
        return OrchestraRun(
            run_id=run_id,
            mode="ask",
            task="stream test",
            created_at="2026-01-01T00:00:00+00:00",
        )

    def test_async_note_stream_called(self):
        """Drive _run_one_agent_async through asyncio.run and assert note_stream fires."""
        from orchestra.engine.parallel import _run_one_agent_async
        from orchestra.models import AgentStatus

        run = self._make_run()
        lock = threading.Lock()
        cancel_event = asyncio.Event()
        captured: list[tuple] = []

        async def fake_run_with_retry_async(provider, effort, prompt, policy, **kwargs):
            cb = kwargs.get("stream_callback")
            if cb:
                cb("async chunk")
            return ("async output", 0)

        mock_provider = MagicMock()
        mock_provider.model_label.return_value = "test-model"
        mock_provider.last_usage.return_value = None
        mock_provider.name = "test-provider"

        async def _noop_budget_guard(*args, **kwargs):
            return None

        async def _drive():
            with (
                patch("orchestra.engine.parallel.resolve", return_value=(mock_provider, "medium")),
                patch("orchestra.engine.parallel.run_with_retry_async", side_effect=fake_run_with_retry_async),
                patch("orchestra.engine.artifacts.append_event"),
                patch("orchestra.engine.artifacts.write_agent_log"),
                patch("orchestra.engine.artifacts.write_manifest"),
                patch("orchestra.engine.artifacts.write_normalized"),
                patch("orchestra.engine.artifacts.load_manifest", return_value=None),
                patch("orchestra.live_status.note_stream", side_effect=lambda *a: captured.append(a)),
                patch("orchestra.engine.parallel._enforce_budget_guard_async", side_effect=_noop_budget_guard),
            ):
                return await _run_one_agent_async(
                    run,
                    "cdx-fast",
                    "do something",
                    lock,
                    MagicMock(),
                    cancel_event,
                )

        agent = asyncio.run(_drive())
        assert len(captured) >= 1, (
            "_run_one_agent_async must call note_stream for streamed chunks"
        )
        assert captured[0][1] == "cdx-fast"

    def test_note_stream_failure_does_not_break_agent(self):
        """If note_stream raises, the agent still completes."""
        from orchestra.engine.parallel import _run_one_agent_async
        from orchestra.models import AgentStatus

        async def fake_run_with_retry_async(provider, effort, prompt, policy, **kwargs):
            kwargs["stream_callback"]("some chunk")
            return ("output", 0)

        mock_provider = MagicMock()
        mock_provider.model_label.return_value = "test-model"
        mock_provider.last_usage.return_value = None
        mock_provider.name = "test-provider"

        async def _drive():
            with (
                patch("orchestra.engine.parallel.resolve", return_value=(mock_provider, "medium")),
                patch("orchestra.engine.parallel.run_with_retry_async", side_effect=fake_run_with_retry_async),
                patch("orchestra.engine.artifacts.append_event"),
                patch("orchestra.engine.artifacts.write_agent_log"),
                patch("orchestra.engine.artifacts.write_manifest"),
                patch("orchestra.engine.artifacts.write_normalized"),
                patch("orchestra.engine.artifacts.load_manifest", return_value=None),
                patch("orchestra.live_status.note_stream", side_effect=RuntimeError("simulated failure")),
            ):
                return await _run_one_agent_async(self._make_run("async-hook-3"), "cdx-fast", "do something",
                                                  threading.Lock(), MagicMock(), asyncio.Event())

        assert asyncio.run(_drive()).status == AgentStatus.COMPLETED
