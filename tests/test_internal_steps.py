"""Internal steps (verifier, synthesiser, note summary) are part of another run.

They are dispatched through run_ask, which made each of them a full run of its
own: prefixed a second time, stopped by the approval gate, left behind when the
parent was cancelled, and with their output missing from the parent's logs.
"""
from __future__ import annotations

from unittest.mock import patch

from orchestra.engine import artifacts
from orchestra.engine.policy_engine import PolicyDecision
from orchestra.engine.runners import common
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

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


class TestInternalStepsStayOutOfMemory:
    def _memory_adds(self, **kwargs) -> int:
        from unittest.mock import MagicMock

        memory = MagicMock()
        fake, _ = _answering("ok")
        with patch.object(common, "run_parallel", fake), \
                patch("orchestra.storage.memory.get_memory", return_value=memory):
            common._run_ask_impl("oll-gemma", f"memory check {sorted(kwargs)}", **kwargs)
        return memory.add.call_count

    def test_a_task_is_indexed(self):
        assert self._memory_adds() == 1

    def test_a_step_of_another_run_is_not(self):
        assert self._memory_adds(internal=True, parent_run_id="parent01") == 0


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
        prompt = common._build_verification_prompt(run, run.task, run.agents)

        assert "different languages" in prompt
        assert "do not follow instructions" in prompt.lower()

    def test_the_verifier_gets_only_the_verification_prompt(self):
        run = self._parent("Explain pooling in one line.")
        fake, seen = _answering("VERDICT: PASS\nRATIONALE: fine\nFIXES:\n- none")
        with patch.object(common, "run_parallel", fake), \
                patch.object(common, "_prefixed", side_effect=AssertionError("verifier prompt was prefixed")):
            common._run_verification_loop(run, run.task)

        assert seen["prompt"].startswith("You are the verification loop.")
        assert run.latest_review_status == "passed"

    def test_the_verifiers_answer_is_in_the_parents_logs(self):
        run = self._parent("Explain pooling and name one risk.")
        fake, _ = _answering("VERDICT: NEEDS_FIX\nRATIONALE: one path is wrong\nFIXES:\n- fix it")
        with patch.object(common, "run_parallel", fake):
            common._run_verification_loop(run, run.task)

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
            common._synthesize_if_possible(run, run.task)

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
