"""Cancelling a run stops the work: its processes, its next phases, its next models."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from unittest.mock import patch

import pytest

from orchestra.engine import artifacts, provider_guard, retries
from orchestra.engine.process_group import ProcessGroupManager, run_provider_process
from orchestra.engine.retries import RetryPolicy
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.providers import base
from orchestra.providers.base import BaseProvider


class _SleepingCli(BaseProvider):
    """A provider whose CLI runs for a long time, started through the base class as Kimi's is."""
    name = "sleeper"

    def is_available(self) -> bool:
        return True

    def model_label(self, effort_or_model: str) -> str:
        return "sleeper/1"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        return [sys.executable, "-c", "import time; time.sleep(60)"]


class _RecordingProvider(_SleepingCli):
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def run(self, prompt, effort_or_model, timeout=180, **kwargs):
        self.kwargs = kwargs
        return "ok", 0


@pytest.fixture(autouse=True)
def _clean_guard():
    for table in (provider_guard._ACTIVE_COUNTS, provider_guard._CONSECUTIVE_TIMEOUTS, provider_guard._DEGRADED_UNTIL):
        table.clear()
    yield
    for table in (provider_guard._ACTIVE_COUNTS, provider_guard._CONSECUTIVE_TIMEOUTS, provider_guard._DEGRADED_UNTIL):
        table.clear()


# ── processes ───────────────────────────────────────────────────────────────

def test_a_cli_started_through_the_base_provider_can_be_killed(run_agent_command, monkeypatch):
    monkeypatch.setattr(base, "run_command", run_agent_command)
    manager = ProcessGroupManager()
    seen_pids: list = []
    result: dict = {}

    def work():
        result["value"] = _SleepingCli().run("x", "m", timeout=60, process_manager=manager,
                                             pid_callback=seen_pids.append)

    thread = threading.Thread(target=work)
    started = time.monotonic()
    thread.start()
    while not manager._pids and time.monotonic() - started < 10:
        time.sleep(0.02)
    assert manager._pids, "the provider never registered its process"

    with patch.object(provider_guard.config, "execution_config", return_value={"kill_grace_seconds": 0.05}):
        manager.kill_all()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert time.monotonic() - started < 15                 # not the 60 s the CLI wanted
    assert result["value"][1] != 0
    assert seen_pids[0] and seen_pids[-1] is None          # reported, then cleared


def test_the_sync_path_hands_its_process_hooks_and_cwd_to_the_provider(tmp_path):
    provider, manager = _RecordingProvider(), ProcessGroupManager()

    def on_pid(pid):
        return None

    with patch("orchestra.engine.process_group.find_tool_result", return_value=None):
        run_provider_process(provider, "x", "m", timeout=5, process_manager=manager,
                             pid_callback=on_pid, cwd=tmp_path)

    assert provider.kwargs["process_manager"] is manager
    assert provider.kwargs["pid_callback"] is on_pid
    assert provider.kwargs["cwd"] == tmp_path


# ── the provider guard ──────────────────────────────────────────────────────

def test_a_good_answer_about_authentication_is_not_a_provider_failure():
    with patch.object(provider_guard.config, "rate_limit_config",
                      return_value={"timeout_circuit_breaker_threshold": 2, "default_concurrency": 2}):
        for _ in range(3):
            provider_guard.start("claude")
            provider_guard.finish("claude", returncode=0,
                                  output="The authentication middleware checks the connection timeout first.")
        assert provider_guard.can_use("claude") is True


def test_a_failed_call_still_counts_towards_the_circuit_breaker():
    with patch.object(provider_guard.config, "rate_limit_config",
                      return_value={"timeout_circuit_breaker_threshold": 2, "default_concurrency": 2}):
        for _ in range(2):
            provider_guard.start("claude")
            provider_guard.finish("claude", returncode=1, output="", stderr="connection refused")
        assert provider_guard.can_use("claude") is False


def test_a_slot_is_given_back_when_the_call_is_interrupted():
    provider = _RecordingProvider()
    policy = RetryPolicy(max_attempts=1, backoff_seconds=[0.0], retry_on_exit_codes=[])

    with patch.object(retries, "run_provider_process", side_effect=KeyboardInterrupt()):
        with pytest.raises(KeyboardInterrupt):
            retries.run_with_retry(provider, "m", "x", policy)

    assert provider_guard._ACTIVE_COUNTS.get("sleeper", 0) == 0


def test_a_provider_that_raises_is_a_failed_attempt_and_gives_its_slot_back():
    provider = _RecordingProvider()
    policy = RetryPolicy(max_attempts=1, backoff_seconds=[0.0], retry_on_exit_codes=[])

    with patch.object(retries, "run_provider_process", side_effect=RuntimeError("binary missing")):
        output, returncode = retries.run_with_retry(provider, "m", "x", policy)

    assert returncode == 1 and "binary missing" in output
    assert provider_guard._ACTIVE_COUNTS.get("sleeper", 0) == 0


@pytest.mark.asyncio
async def test_a_slot_is_given_back_when_the_task_is_cancelled():
    started = asyncio.Event()

    class _Hanging(_SleepingCli):
        async def async_run(self, prompt, effort_or_model, timeout=180, **kwargs):
            started.set()
            await asyncio.sleep(60)
            return "", 0

    policy = RetryPolicy(max_attempts=1, backoff_seconds=[0.0], retry_on_exit_codes=[])
    task = asyncio.ensure_future(retries.run_with_retry_async(_Hanging(), "m", "x", policy))
    await started.wait()
    assert provider_guard._ACTIVE_COUNTS.get("sleeper") == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider_guard._ACTIVE_COUNTS.get("sleeper", 0) == 0


# ── the order of a cancel ───────────────────────────────────────────────────

def _started_run(task: str) -> OrchestraRun:
    run = OrchestraRun(mode="dual", task=task, status=RunStatus.RUNNING)
    run.agents = [AgentRun(alias="cdx-deep", provider="codex", model="m", status=AgentStatus.STARTED)]
    artifacts.write_manifest(run)
    return run


def test_the_run_is_marked_cancelled_before_its_processes_are_killed():
    """An agent whose process dies returns empty output; the retry loop must already see the cancel."""
    from orchestra import service

    run = _started_run("cancel order probe")
    status_when_killed: list = []

    def record(self):
        status_when_killed.append(artifacts.load_manifest(run.run_id)["status"])

    with patch.object(ProcessGroupManager, "kill_all", record), patch("orchestra.live_status.install"):
        service.cancel_run(run.run_id)

    assert status_when_killed == ["cancelled"]


# ── the runners between phases ──────────────────────────────────────────────

def _cancel_on_disk(run: OrchestraRun) -> None:
    manifest = artifacts.load_manifest(run.run_id)
    manifest["status"] = "cancelled"
    artifacts.write_manifest_data(run.run_id, manifest)


def test_a_dual_run_cancelled_during_round_one_does_not_go_on_to_verification():
    from orchestra.engine.runners import dual_runner

    def round_one(run, pairs, **kwargs):
        for alias, _ in pairs:
            run.agents.append(AgentRun(alias=alias, provider="p", model="m",
                                       status=AgentStatus.COMPLETED, stdout_log="answer"))
        _cancel_on_disk(run)

    with patch.object(dual_runner, "run_parallel", round_one), \
         patch.object(dual_runner, "_run_verification_loop") as verify, \
         patch.object(dual_runner, "_synthesize_if_possible") as synthesize:
        run = dual_runner.run_dual("cancel probe: dual", agents=["cdx-deep", "kimi-deep"],
                                   show_live=False, emit_console=False)

    verify.assert_not_called()
    synthesize.assert_not_called()
    assert run.status == RunStatus.CANCELLED
    assert artifacts.load_manifest(run.run_id)["status"] == "cancelled"


def test_a_planned_run_starts_no_further_node_once_it_is_cancelled():
    from orchestra.engine.graph import OrchestraGraph
    from orchestra.engine.runners import planned_runner

    graph = OrchestraGraph("cancel-probe")
    for name in ("first", "second", "third"):
        graph.add_node(name, f"do {name}", "cdx-deep")
    graph.add_edge("first", "second")
    graph.add_edge("second", "third")
    started: list[str] = []
    parent: dict = {}

    def step(alias, prompt, **kwargs):
        started.append(prompt.split("\n")[0])
        step_run = OrchestraRun(mode="ask", task=prompt, status=RunStatus.COMPLETED,
                                parent_run_id=kwargs.get("parent_run_id"))
        step_run.agents = [AgentRun(alias=alias, provider="p", model="m",
                                    status=AgentStatus.COMPLETED, stdout_log="done")]
        step_run.summary = "done"
        manifest = artifacts.load_manifest(kwargs["parent_run_id"])
        manifest["status"] = "cancelled"                      # the user cancels during the first node
        artifacts.write_manifest_data(kwargs["parent_run_id"], manifest)
        parent["id"] = kwargs["parent_run_id"]
        return step_run

    with patch("orchestra.engine.planner.create_graph_plan", return_value=graph), \
         patch("orchestra.engine.runners.ask_runner.run_ask", step), \
         patch.object(planned_runner, "_run_verification_loop") as verify:
        run = planned_runner.run_planned("cancel probe: planned")

    assert started == ["do first"]
    verify.assert_not_called()
    assert run.status == RunStatus.CANCELLED
    assert artifacts.load_manifest(parent["id"])["status"] == "cancelled"
