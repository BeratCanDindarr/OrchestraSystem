"""run_parallel end to end with a fake provider: called with no running loop (CLI, the job worker,
an MCP call in its worker thread) and from inside a loop (the same async core in a worker thread)."""
from __future__ import annotations

import asyncio

import pytest

from orchestra.engine import parallel
from orchestra.models import AgentStatus, OrchestraRun, RunStatus
from orchestra.providers.base import BaseProvider
from orchestra.state import FailureKind


class FakeProvider(BaseProvider):
    name = "fake"

    def __init__(self, answers: dict[str, object]):
        self.answers = answers   # effort -> answer text, (text, returncode), or an exception to raise

    def is_available(self) -> bool:
        return True

    def model_label(self, effort_or_model: str) -> str:
        return f"fake/{effort_or_model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        return ["true"]

    def run(self, prompt, effort_or_model, timeout=180, **kwargs):
        answer = self.answers[effort_or_model]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, tuple):
            return answer
        self._last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return f"{answer}\n\nConfidence: high", 0


@pytest.fixture
def fake_agents(monkeypatch):
    """`good-a` / `good-b` answer; `boom` raises; `broken` exits non-zero. No fallback chain."""
    provider = FakeProvider({"good-a": "answer A", "good-b": "answer B",
                             "boom": RuntimeError("provider exploded"),
                             "broken": ("[ERROR] Error: not logged in. Run `kimi login` first.", 1)})

    def resolve(alias, attempt=0):
        return provider, alias

    monkeypatch.setattr(parallel, "resolve", resolve)
    monkeypatch.setattr("orchestra.engine.retries.resolve_with_fallback", resolve)
    monkeypatch.setattr("orchestra.providers.fallback.fallback_chain", lambda alias: [alias])
    return provider


def _run(aliases: list[str], inside_loop: bool) -> tuple[OrchestraRun, list]:
    run = OrchestraRun(mode="dual", task="t")
    run.status = RunStatus.RUNNING
    pairs = [(alias, "the task") for alias in aliases]

    def call():
        return parallel.run_parallel(run, pairs, show_live=False, install_signal_handlers=False, emit_console=False)

    if not inside_loop:
        return run, call()

    async def in_loop():
        return call()

    return run, asyncio.run(in_loop())


@pytest.mark.parametrize("inside_loop", [False, True], ids=["no-loop", "inside-a-loop"])
def test_both_agents_answer(fake_agents, inside_loop):
    run, results = _run(["good-a", "good-b"], inside_loop)

    by_alias = {agent.alias: agent for agent in run.agents}
    assert {agent.alias for agent in results} == {"good-a", "good-b"}
    assert by_alias["good-a"].status == AgentStatus.COMPLETED and "answer A" in by_alias["good-a"].stdout_log
    assert by_alias["good-b"].confidence == pytest.approx(0.85)


@pytest.mark.parametrize("inside_loop", [False, True], ids=["no-loop", "inside-a-loop"])
def test_an_agent_that_fails_does_not_take_the_other_answer_with_it(fake_agents, inside_loop):
    run, results = _run(["good-a", "broken"], inside_loop)

    by_alias = {agent.alias: agent for agent in run.agents}
    assert by_alias["good-a"].status == AgentStatus.COMPLETED
    assert by_alias["broken"].status == AgentStatus.FAILED and by_alias["broken"].error


@pytest.mark.parametrize("inside_loop", [False, True], ids=["no-loop", "inside-a-loop"])
def test_an_agent_whose_provider_raises_does_not_lose_the_other_answer(fake_agents, inside_loop):
    run, results = _run(["good-a", "boom"], inside_loop)

    by_alias = {agent.alias: agent for agent in run.agents}
    assert by_alias["good-a"].status == AgentStatus.COMPLETED and "answer A" in by_alias["good-a"].stdout_log
    assert by_alias["boom"].status == AgentStatus.FAILED


# ── the budget guard stops a run that has spent its budget ─────────────────────

def _spending_run(cost: float) -> OrchestraRun:
    from orchestra.models import AgentRun

    run = OrchestraRun(mode="dual", task="t")
    run.status = RunStatus.RUNNING
    run.agents = [AgentRun(alias="spent", provider="p", model="m", status=AgentStatus.COMPLETED, estimated_cost_usd=cost),
                  AgentRun(alias="still-running", provider="p", model="m", status=AgentStatus.STARTED)]
    run.total_cost_usd = cost
    return run


@pytest.mark.parametrize("budget", [{"max": 0.5, "daily": 0.0}, {"max": 0.0, "daily": 0.5}], ids=["run", "daily"])
def test_a_run_over_its_budget_is_stopped(monkeypatch, budget):
    import threading

    from orchestra.engine.parallel import budget as guard
    from orchestra.engine.process_group import ProcessGroupManager

    monkeypatch.setattr(guard.config, "max_budget_usd", lambda: budget["max"])
    monkeypatch.setattr(guard.config, "token_budget_config", lambda: {"daily_budget_usd": budget["daily"]})
    monkeypatch.setattr(guard, "_daily_spend_usd", lambda run_id: 0.0)
    run, cancel = _spending_run(1.0), asyncio.Event()
    asyncio.run(guard._enforce_budget_guard_async(run, threading.Lock(), ProcessGroupManager(), cancel))

    assert cancel.is_set() and run.status == RunStatus.FAILED and run.failure.kind == FailureKind.BUDGET_EXCEEDED.value
    assert {a.alias: a.status for a in run.agents} == {"spent": AgentStatus.COMPLETED, "still-running": AgentStatus.CANCELLED}


def test_a_run_within_its_budget_goes_on(monkeypatch):
    import threading

    from orchestra.engine.parallel import budget as guard
    from orchestra.engine.process_group import ProcessGroupManager

    monkeypatch.setattr(guard.config, "max_budget_usd", lambda: 5.0)
    monkeypatch.setattr(guard.config, "token_budget_config", lambda: {"daily_budget_usd": 0.0})
    run, cancel = _spending_run(1.0), asyncio.Event()
    asyncio.run(guard._enforce_budget_guard_async(run, threading.Lock(), ProcessGroupManager(), cancel))

    assert not cancel.is_set() and run.status == RunStatus.RUNNING


@pytest.mark.parametrize("inside_loop", [False, True], ids=["no-loop", "inside-a-loop"])
def test_a_failed_agent_says_why_not_just_its_exit_code(fake_agents, inside_loop):
    run, _ = _run(["broken"], inside_loop)

    error = run.agents[0].error
    assert error.startswith("exit code 1") and "not logged in" in error and "[ERROR]" not in error
    assert "not logged in" in run.failure.message
