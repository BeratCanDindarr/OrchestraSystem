"""Tests for calibration gate/helper and finalize judge wiring (Faz 4 Block B/F)."""
from unittest.mock import patch

from orchestra import config
from orchestra.models import OrchestraRun
from orchestra.engine.runners import common


def _completed_run():
    run = OrchestraRun(mode="ask", task="what is X?")
    run.summary = "X is a thing."
    return run


def test_maybe_assess_calibration_calls_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "eval_calibration_enabled", lambda: True)
    calls = {}

    def fake_assess(question, answer, **kwargs):
        calls["args"] = (question, answer)
        return {"calibration_score": 0.7, "sycophantic": False}

    with patch("orchestra.engine.calibration.assess_calibration", side_effect=fake_assess):
        out = common._maybe_assess_calibration(_completed_run())
    assert calls["args"] == ("what is X?", "X is a thing.")
    assert out["calibration_score"] == 0.7


def test_maybe_assess_calibration_guards_against_recursion(monkeypatch):
    """The calibration run must set the judge guard so its own finalize does not
    recursively judge/calibrate itself (unbounded recursion)."""
    from orchestra.engine import eval_harness
    monkeypatch.setattr(config, "eval_calibration_enabled", lambda: True)
    seen = {}

    def fake_assess(question, answer, **kwargs):
        seen["judging"] = eval_harness.is_judging()
        return {"calibration_score": 0.5}

    with patch("orchestra.engine.calibration.assess_calibration", side_effect=fake_assess):
        common._maybe_assess_calibration(_completed_run())
    assert seen["judging"] is True          # guard active during the nested run
    assert eval_harness.is_judging() is False  # cleanly reset afterwards


def test_maybe_assess_calibration_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "eval_calibration_enabled", lambda: False)
    with patch("orchestra.engine.calibration.assess_calibration") as stub:
        out = common._maybe_assess_calibration(_completed_run())
    assert out is None
    stub.assert_not_called()


def test_config_defaults():
    # Defaults are off / full-rate (verified against a fresh accessor read).
    assert config.eval_calibration_enabled() is False
    assert config.eval_sample_rate() == 1.0
