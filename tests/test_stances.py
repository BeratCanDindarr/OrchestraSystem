"""Consensus mode: two agents argue opposite sides and the synthesis decides."""
from __future__ import annotations

from unittest.mock import patch

from orchestra import service
from orchestra.engine import stances
from orchestra.engine.runners import common, dual_runner
from orchestra.engine.synthesizer import build_synthesis_prompt
from orchestra.models import AgentRun, AgentStatus, OrchestraRun


def _two_agents(seen: dict):
    def fake(run, agents, **kwargs):
        for alias, prompt in agents:
            seen[alias] = prompt
            agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
            agent.stdout_log = f"{alias} argues. " * 20
            agent.validation_status = "passed"
            run.agents.append(agent)

    return fake


def _run(task: str, **kwargs):
    seen: dict = {}
    with patch.object(dual_runner, "run_parallel", _two_agents(seen)), \
            patch.object(dual_runner, "_run_verification_loop"), \
            patch.object(dual_runner, "_synthesize_if_possible"), \
            patch("orchestra.engine.adversarial.ensure_diversity", lambda aliases: aliases):
        run = dual_runner.run_dual(task, agents=["cld-deep", "kimi-deep"], **kwargs)
    return run, seen


def test_the_two_agents_are_given_opposite_stances():
    run, seen = _run("stances: should level data move to JSON?", stances=True)

    assert "[YOUR STANCE: FOR]" in seen["cld-deep"] and "[YOUR STANCE: AGAINST]" not in seen["cld-deep"]
    assert "[YOUR STANCE: AGAINST]" in seen["kimi-deep"]
    assert run.stances == {"cld-deep": "for", "kimi-deep": "against"}
    assert OrchestraRun.from_manifest(run.to_manifest()).stances == run.stances


def test_a_plain_dual_run_has_no_stances():
    run, seen = _run("stances: plain dual question")

    assert "[YOUR STANCE" not in seen["cld-deep"] + seen["kimi-deep"]
    assert run.stances is None


def test_honesty_comes_before_the_stance():
    for stance in (stances.FOR, stances.AGAINST):
        text = stances.with_stance("Adopt Addressables?", stance)
        assert text.startswith("Adopt Addressables?")
        assert "Honesty comes before the stance" in text and "What would change my mind" in text


def test_the_verifier_is_told_the_disagreement_is_intended():
    run = OrchestraRun(mode="dual", task="t", stances={"cld-deep": "for", "kimi-deep": "against"})
    plain = OrchestraRun(mode="dual", task="t")

    assert "opposite stances" in common._build_verification_prompt(run, "t", [])
    assert "opposite stances" not in common._build_verification_prompt(plain, "t", [])


def test_the_synthesiser_is_told_to_decide_not_average():
    agents = []
    for alias in ("cld-deep", "kimi-deep"):
        agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = "case"
        agents.append(agent)

    prompt = build_synthesis_prompt(agents, task="Adopt Addressables?", stances={"cld-deep": "for", "kimi-deep": "against"})

    assert "cld-deep argued for, kimi-deep argued against" in prompt
    assert "Do not average" in prompt and "what evidence would" in prompt
    assert "[ASSIGNED STANCES]" not in build_synthesis_prompt(agents, task="Adopt Addressables?")


def test_consensus_is_a_mode_of_the_service():
    seen: dict = {}
    with patch.object(dual_runner, "run_parallel", _two_agents(seen)), \
            patch.object(dual_runner, "_run_verification_loop"), \
            patch.object(dual_runner, "_synthesize_if_possible"):
        result = service.run_task(mode="consensus", task="stances: service check, adopt Addressables?")

    assert sorted(result["run"]["stances"].values()) == ["against", "for"]
