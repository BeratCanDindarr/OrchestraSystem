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
        answer = self.answers.get(effort_or_model)
        return ["sleep", str(answer)] if isinstance(answer, float) else ["true"]

    def run(self, prompt, effort_or_model, timeout=180, **kwargs):
        answer = self.answers[effort_or_model]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, tuple):
            return answer
        if isinstance(answer, float):   # a slow agent: a real process, so a cancel can end it
            output, code = BaseProvider.run(self, prompt, effort_or_model, timeout, **kwargs)
            return ("slow answer\n\nConfidence: high", 0) if code == 0 else (output, code)
        self._last_usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return f"{answer}\n\nConfidence: high", 0


@pytest.fixture
def fake_agents(monkeypatch):
    """`good-a` / `good-b` answer; `boom` raises; `broken` exits non-zero. No fallback chain."""
    provider = FakeProvider({"good-a": "answer A", "good-b": "answer B",
                             "boom": RuntimeError("provider exploded"),
                             "broken": ("[ERROR] Error: not logged in. Run `kimi login` first.", 1),
                             "slow": 3.0})

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


def test_a_dual_agent_much_slower_than_the_first_is_cut_and_marked_late(fake_agents, monkeypatch, run_agent_command):
    monkeypatch.setattr("orchestra.providers.base.run_command", run_agent_command)   # a real `sleep`
    import time

    monkeypatch.setattr(parallel.config, "section", lambda name: {"patience_seconds": 0.5} if name == "dual" else {})
    started = time.monotonic()

    run, _ = _run(["good-a", "slow"], inside_loop=False)

    by_alias = {agent.alias: agent for agent in run.agents}
    assert time.monotonic() - started < 2.5
    assert by_alias["good-a"].status == AgentStatus.COMPLETED
    assert by_alias["slow"].status == AgentStatus.CANCELLED and by_alias["slow"].error.startswith("late")


def test_patience_zero_waits_for_every_agent(fake_agents, monkeypatch, run_agent_command):
    monkeypatch.setattr("orchestra.providers.base.run_command", run_agent_command)   # a real `sleep`
    monkeypatch.setattr(parallel.config, "section", lambda name: {"patience_seconds": 0} if name == "dual" else {})
    fake_agents.answers["slow"] = 0.8

    run, _ = _run(["good-a", "slow"], inside_loop=False)

    assert {agent.alias: agent.status for agent in run.agents} == {"good-a": AgentStatus.COMPLETED,
                                                                   "slow": AgentStatus.COMPLETED}


def test_an_attempt_that_fails_says_why_before_the_next_one_answers(monkeypatch):
    # Only the last failure of a chain was recorded: when Claude hit its usage limit and Kimi
    # answered in its place, nothing said why Claude had not.
    limited = FakeProvider({"opus": ("Error: You've hit your limit · resets 1:50pm (Europe/Istanbul)", 1)})
    stand_in = FakeProvider({"k3": "answer from the stand-in"})
    stand_in.name = "stand-in"

    def resolve(alias, attempt=0):
        return (limited, "opus") if attempt == 0 else (stand_in, "k3")

    monkeypatch.setattr(parallel, "resolve", resolve)
    monkeypatch.setattr("orchestra.engine.retries.resolve_with_fallback", resolve)
    monkeypatch.setattr(parallel, "_own_provider", lambda alias: limited)
    from orchestra.engine.retries import RetryPolicy

    monkeypatch.setattr(parallel, "RetryPolicy", lambda: RetryPolicy(max_attempts=2, backoff_seconds=[0]))

    run, _ = _run(["cld-deep"], inside_loop=False)

    retry = next(e for e in parallel.artifacts.read_events(run.run_id) if e.get("event") == "agent_retry")
    assert run.agents[0].status == AgentStatus.COMPLETED
    assert "hit your limit" in retry["after"] and retry["after"].startswith("exit code 1")


def test_a_dual_waits_for_every_agent_unless_told_otherwise(monkeypatch):
    # An agent that is still answering is not cut: patience is something to switch on.
    monkeypatch.setattr(parallel.config, "section", lambda name: {})

    assert parallel._patience_seconds(OrchestraRun(mode="dual", task="t"), 2) == 0
