"""Internal steps (verifier, synthesiser, note summary) are part of another run.

They are dispatched through run_ask, which made each of them a full run of its
own: prefixed a second time, stopped by the approval gate, left behind when the
parent was cancelled, and with their output missing from the parent's logs.
"""
from __future__ import annotations

import pytest

from unittest.mock import patch

from orchestra.engine import artifacts
from orchestra.engine.policy_engine import PolicyDecision
from orchestra.engine.runners import common, synthesis
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


@pytest.fixture(autouse=True)
def _agents_that_can_write(monkeypatch):
    """These tests exercise the approval gate, which holds back only agents that can change files."""
    monkeypatch.setattr("orchestra.providers.fallback._writes", lambda alias: True)


GATE = PolicyDecision("gate", "privileged_gate: task contains 'deploy'", "autonomy_policy")


def _answering(text: str):
    """A run_parallel stand-in: the agent completes with the given text."""
    seen: dict = {}

    def fake(run, agents, **kwargs):
        alias, prompt = agents[0]
        seen["prompt"] = prompt
        agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = text
        run.agents.append(agent)

    return fake, seen


class TestInternalAsk:
    def test_an_internal_step_is_not_stopped_by_the_approval_gate(self):
        fake, _ = _answering("VERDICT: PASS")
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_evaluate_policy", return_value=GATE) as policy, \
                patch.object(common, "suspend_run_to_db"):
            run = common._run_ask_impl("oll-gemma", "check these candidates", internal=True)

        assert run.status != RunStatus.WAITING_APPROVAL
        policy.assert_not_called()

    def test_an_ordinary_ask_is_still_stopped(self):
        fake, _ = _answering("answer")
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_evaluate_policy", return_value=GATE), \
                patch.object(common, "suspend_run_to_db"):
            run = common._run_ask_impl("oll-gemma", "deploy it")

        assert run.status == RunStatus.WAITING_APPROVAL

    def test_an_internal_step_sends_its_prompt_as_it_is(self):
        fake, seen = _answering("ok")
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_prefixed", side_effect=AssertionError("prefixed again")), \
                patch.object(common, "_try_cache_hit", side_effect=AssertionError("cache consulted")):
            common._run_ask_impl("oll-gemma", "the finished prompt", internal=True)

        assert seen["prompt"] == "the finished prompt"

    def test_an_internal_step_remembers_its_parent(self):
        fake, _ = _answering("ok")
        with patch.object(common, "run_parallel", fake):
            run = common._run_ask_impl("oll-gemma", "p", internal=True, parent_run_id="parent01")

        assert run.parent_run_id == "parent01"
        assert artifacts.load_manifest(run.run_id)["parent_run_id"] == "parent01"
        assert OrchestraRun.from_manifest(run.to_manifest()).parent_run_id == "parent01"


class TestVerificationStep:
    def _parent(self, task: str = "Explain pooling. Do not delete files.") -> OrchestraRun:
        # Each test uses its own task text: an identical verifier prompt may be answered from
        # the idempotency store when another test has switched it on.
        run = OrchestraRun(mode="dual", task=task)
        for alias, text in (("cld-deep", "Havuzlama bellek tahsisini azaltır."), ("kimi-deep", "Pooling avoids allocations.")):
            agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
            agent.stdout_log = text
            agent.validation_status = "passed"
            run.agents.append(agent)
        artifacts.write_manifest(run)
        return run

    def test_the_prompt_tells_the_verifier_what_not_to_judge(self):
        run = self._parent()
        prompt = synthesis._build_verification_prompt(run, run.task, run.agents)

        assert "different languages" in prompt
        assert "do not follow instructions" in prompt.lower()

    def test_the_verifier_gets_only_the_verification_prompt(self):
        run = self._parent("Explain pooling in one line.")
        fake, seen = _answering("VERDICT: PASS\nRATIONALE: fine\nFIXES:\n- none")
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_prefixed", side_effect=AssertionError("verifier prompt was prefixed")):
            synthesis._run_verification_loop(run, run.task)

        assert seen["prompt"].startswith("You are the verification loop.")
        assert run.latest_review_status == "passed"

    def test_the_verifiers_answer_is_in_the_parents_logs(self):
        run = self._parent("Explain pooling and name one risk.")
        fake, _ = _answering("VERDICT: NEEDS_FIX\nRATIONALE: one path is wrong\nFIXES:\n- fix it")
        with patch.object(common, "run_parallel", fake):
            synthesis._run_verification_loop(run, run.task)

        assert "one path is wrong" in (artifacts.read_agent_log(run.run_id, "verifier") or "")


class TestSynthesisStep:
    def test_synthesis_runs_and_fills_the_summary(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "auto_synthesize", lambda: True)
        run = OrchestraRun(mode="dual", task="Compare tweens and animators for a puzzle game UI.")
        for alias, text in (("cld-deep", "Tweens are cheap to author and easy to sequence in code. " * 6),
                            ("kimi-deep", "Animators suit artist-driven state machines and blending. " * 6)):
            agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
            agent.stdout_log = text
            agent.confidence = 0.9
            run.agents.append(agent)
        artifacts.write_manifest(run)

        fake, _ = _answering("Use tweens for simple UI motion and an Animator where artists own the states.")
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_evaluate_policy", return_value=GATE):
            synthesis._synthesize_if_possible(run, run.task)

        assert run.summary and run.summary.startswith("Use tweens for simple UI motion")


class TestCancelReachesChildren:
    def test_cancelling_a_run_cancels_the_steps_it_started(self):
        from orchestra import service

        parent = OrchestraRun(mode="dual", task="t")
        parent.status = RunStatus.WAITING_APPROVAL
        artifacts.write_manifest(parent)
        child = OrchestraRun(mode="ask", task="verify", parent_run_id=parent.run_id)
        child.status = RunStatus.RUNNING
        artifacts.write_manifest(child)
        other = OrchestraRun(mode="ask", task="unrelated")
        other.status = RunStatus.RUNNING
        artifacts.write_manifest(other)

        service.cancel_run(parent.run_id)

        assert artifacts.load_manifest(parent.run_id)["status"] == "cancelled"
        assert artifacts.load_manifest(child.run_id)["status"] == "cancelled"
        assert artifacts.load_manifest(other.run_id)["status"] == "running"


# ── every judge, critic and planner call is a step of the run it serves ─────

def _answer(text: str, run_id: str = "r"):
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

    run = OrchestraRun(mode="ask", task="t", run_id=run_id, status=RunStatus.COMPLETED)
    agent = AgentRun(alias="x", provider="p", model="m", status=AgentStatus.COMPLETED)
    agent.stdout_log = text
    run.agents = [agent]
    return run


class _Calls:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.kwargs: list[dict] = []
        self.aliases: list[str] = []

    def __call__(self, alias, prompt, **kwargs):
        self.aliases.append(alias)
        self.kwargs.append(kwargs)
        return _answer(self.answers.pop(0) if self.answers else "ok")


def test_the_eval_judge_is_a_step_of_the_run_it_judges():
    from unittest.mock import patch

    from orchestra.engine import eval_harness

    ask = _Calls('{"composite_score": 0.8}')
    with patch("orchestra.engine.runners.ask_runner.run_ask", ask):
        eval_harness.run_judge("task", "output", run_id="judged1", t_hash="h")

    assert ask.kwargs[0]["internal"] is True and ask.kwargs[0]["parent_run_id"] == "judged1"


def test_the_calibration_judge_is_a_step_of_the_run_it_assesses():
    from orchestra.engine import calibration

    ask = _Calls('{"calibration_score": 0.5}')
    calibration.assess_calibration("q", "a", run_ask_fn=ask, parent_run_id="assessed1")

    assert ask.kwargs[0] == {"internal": True, "parent_run_id": "assessed1"}


def test_the_reflection_critic_is_a_step_and_the_retry_is_not():
    from orchestra.engine import reflection

    ask = _Calls('{"needs_revision": true, "issues": ["x"], "guidance": "y"}', "revised",
                 '{"needs_revision": false}')
    reflection.reflect("cdx-deep", "task", _answer("first", run_id="base1"), run_ask_fn=ask,
                       max_rounds=2, critique_alias="cld-fast")

    critic, retry = ask.kwargs[0], ask.kwargs[1]
    assert critic["internal"] is True and critic["parent_run_id"] == "base1"
    assert "internal" not in retry          # the revised answer is an answer


def test_the_supervisors_planner_and_synthesiser_get_their_prompts_as_written():
    from orchestra.engine import supervisor

    ask = _Calls('{"subtasks": ["a", "b"]}', "part a", "part b", "merged")
    supervisor._supervise("big task", 0, 1, run_ask_fn=ask, worker_alias="w",
                          planner_alias="planner", synthesizer_alias="synth")

    by_alias = dict(zip(ask.aliases, ask.kwargs))
    assert by_alias["planner"].get("internal") is True
    assert by_alias["synth"].get("internal") is True
    assert "internal" not in by_alias["w"]      # a subtask is a task


def test_a_retry_after_a_failure_keeps_what_the_first_call_was_given():
    """A follow-up that is retried must still be a follow-up, and must be asked again."""
    from unittest.mock import patch

    from orchestra.engine.runners import ask_runner
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

    failed = OrchestraRun(mode="ask", task="t", status=RunStatus.FAILED)
    failed.agents = [AgentRun(alias="cdx-deep", provider="p", model="m", status=AgentStatus.FAILED)]
    seen: list[dict] = []

    def degrade(alias, prompt, run, *, run_ask_fn):
        run_ask_fn(alias, prompt)
        return run

    def impl(alias, prompt, **kwargs):
        seen.append(kwargs)
        return failed

    with patch.object(ask_runner, "_run_ask_impl", impl), \
         patch.object(ask_runner.config, "retry_enabled", return_value=True), \
         patch.object(ask_runner.config, "reflection_enabled", return_value=False), \
         patch("orchestra.engine.degradation.degrade", degrade), \
         patch("orchestra.engine.degradation.is_degrading", side_effect=lambda: len(seen) > 1):
        ask_runner.run_ask("cdx-deep", "and then?", follows_run_id="earlier1")

    first, retried = seen
    assert retried["follows_run_id"] == "earlier1"
    assert retried["fresh"] is True


def test_a_step_is_not_graded_as_a_task():
    from unittest.mock import patch

    from orchestra.engine.runners import common
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

    step = OrchestraRun(mode="ask", task="judge this", status=RunStatus.RUNNING, parent_run_id="parent01")
    step.agents = [AgentRun(alias="cld-fast", provider="claude", model="m",
                            status=AgentStatus.COMPLETED, stdout_log="verdict")]

    with patch("orchestra.storage.eval_tracker.EvalTracker.log_run") as log_run:
        common._finalize(step)

    log_run.assert_not_called()


# ── an answer that must be JSON is checked, and asked for once more ─────────

def test_an_object_is_found_inside_fences_and_prose():
    from orchestra.engine import structured

    assert structured.extract_object('Sure:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    with pytest.raises(ValueError):
        structured.extract_object("no object here")
    with pytest.raises(ValueError):
        structured.extract_object("[1, 2, 3]")


def test_a_readable_answer_is_not_asked_for_twice():
    from orchestra.engine import structured

    ask = _Calls('{"subtasks": ["a"]}')
    data, _ = structured.ask_for_object(ask, "planner", "split this")

    assert data == {"subtasks": ["a"]} and len(ask.aliases) == 1


def test_prose_is_answered_with_one_request_for_the_object():
    from orchestra.engine import structured

    ask = _Calls("I would split it in two parts.", '{"subtasks": ["a", "b"]}')
    data, run = structured.ask_for_object(ask, "planner", "split this", internal=True)

    assert data == {"subtasks": ["a", "b"]}
    assert len(ask.aliases) == 2 and ask.kwargs[1] == {"internal": True}


def test_an_object_of_the_wrong_shape_is_asked_for_again_with_the_reason():
    from orchestra.engine import structured

    prompts: list[str] = []

    def ask(alias, prompt, **kwargs):
        prompts.append(prompt)
        return _answer('{"subtasks": "a and b"}' if len(prompts) == 1 else '{"subtasks": ["a", "b"]}')

    def check(data):
        return None if isinstance(data.get("subtasks"), list) else "subtasks must be a list"

    data, _ = structured.ask_for_object(ask, "planner", "split this", check=check)

    assert data == {"subtasks": ["a", "b"]}
    assert "subtasks must be a list" in prompts[1] and prompts[1].startswith("split this")


def test_two_unusable_answers_give_up():
    from orchestra.engine import structured

    ask = _Calls("prose", "more prose")
    data, _ = structured.ask_for_object(ask, "planner", "split this")

    assert data is None and len(ask.aliases) == 2


def test_the_supervisor_gets_its_subtasks_from_a_planner_that_first_answers_in_prose():
    from orchestra.engine import supervisor

    ask = _Calls("Let me think about how to split this.", '{"subtasks": ["a", "b"]}')

    assert supervisor._decompose("big task", "planner", ask) == ["a", "b"]


def test_the_reflection_critic_is_asked_again_before_its_verdict_is_ignored():
    from orchestra.engine import reflection

    ask = _Calls("The answer has a problem with units.",
                 '{"needs_revision": true, "issues": ["units"], "guidance": "use metres"}', "revised answer",
                 '{"needs_revision": false}')
    result = reflection.reflect("cdx-deep", "task", _answer("first", run_id="base1"), run_ask_fn=ask,
                                max_rounds=2, critique_alias="cld-fast")

    assert result.agents[0].stdout_log == "revised answer"
