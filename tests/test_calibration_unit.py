"""Unit tests for orchestra.engine.calibration (anti-sycophancy / calibration judge)."""
from __future__ import annotations

from orchestra import config
from orchestra.engine import calibration
from orchestra.models import AgentRun, AgentStatus, OrchestraRun


def _mk_run(output: str) -> OrchestraRun:
    run = OrchestraRun(mode="ask", task="t")
    run.agents = [
        AgentRun(alias="a", provider="p", model="m",
                 status=AgentStatus.COMPLETED, stdout_log=output)
    ]
    return run


def test_build_calibration_prompt_requests_json_rubric():
    p = calibration.build_calibration_prompt("Q?", "A.")
    assert "calibration_score" in p and "sycophantic" in p
    assert "appropriately_abstained" in p
    assert "Q?" in p and "A." in p


def test_assess_calibration_parses_verdict():
    verdict = ('{"calibration_score": 0.8, "appropriately_abstained": true, '
               '"overconfident": false, "sycophantic": false, "explanation": "good"}')
    calls = []

    def fake_ask(alias, prompt, **kwargs):
        calls.append(alias)
        return _mk_run(verdict)

    out = calibration.assess_calibration("Q?", "A.", run_ask_fn=fake_ask, judge_alias="cld-fast")
    assert out["calibration_score"] == 0.8
    assert out["appropriately_abstained"] is True
    assert out["overconfident"] is False and out["sycophantic"] is False
    assert calls == ["cld-fast"]


def test_assess_calibration_flags_sycophancy():
    verdict = ('{"calibration_score": 0.2, "appropriately_abstained": false, '
               '"overconfident": true, "sycophantic": true, "explanation": "just agreed"}')
    out = calibration.assess_calibration(
        "Is my wrong claim right?", "Yes, absolutely!",
        run_ask_fn=lambda a, p, **k: _mk_run(verdict),
    )
    assert out["sycophantic"] is True and out["overconfident"] is True


def test_assess_calibration_default_on_unparseable_verdict():
    out = calibration.assess_calibration(
        "Q?", "A.", run_ask_fn=lambda a, p, **k: _mk_run("not json at all"),
    )
    assert out["calibration_score"] == 0.0 and out["sycophantic"] is False


def test_assess_calibration_default_on_judge_error():
    def boom(alias, prompt, **kwargs):
        raise RuntimeError("judge down")

    out = calibration.assess_calibration("Q?", "A.", run_ask_fn=boom)
    assert out == calibration._default_assessment("judge run failed")


def test_calibration_config_default():
    assert config.calibration_judge_alias() == "cld-fast"
