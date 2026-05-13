"""Tests for CascadeRouter — cheap-first escalation routing."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestra.router.cascade_router import CascadeRouter, _is_provider_healthy, _outcome_adjusted_threshold


def _make_run(avg_confidence: float = 0.5, run_id: str = "run-001") -> MagicMock:
    run = MagicMock()
    run.avg_confidence = avg_confidence
    run.run_id = run_id
    return run


# ---------------------------------------------------------------------------
# _is_provider_healthy
# ---------------------------------------------------------------------------

def test_provider_healthy_when_not_degraded():
    import time
    # _DEGRADED_UNTIL has no entry for codex → healthy
    with patch("orchestra.engine.provider_guard._DEGRADED_UNTIL", {}):
        result = _is_provider_healthy("cdx-fast")
    assert result is True


def test_provider_degraded_returns_false():
    import time
    future = time.time() + 9999
    with patch("orchestra.engine.provider_guard._DEGRADED_UNTIL", {"codex": future}):
        result = _is_provider_healthy("cdx-fast")
    assert result is False


def test_provider_healthy_after_degradation_expires():
    import time
    past = time.time() - 1  # expired
    with patch("orchestra.engine.provider_guard._DEGRADED_UNTIL", {"codex": past}):
        result = _is_provider_healthy("cdx-fast")
    assert result is True


# ---------------------------------------------------------------------------
# CascadeRouter.run — happy path (first alias meets threshold)
# ---------------------------------------------------------------------------

def test_cascade_stops_at_first_confident_alias():
    router = CascadeRouter(
        chain=["cdx-fast", "cdx-deep"],
        confidence_threshold=0.65,
        max_escalations=2,
    )
    high_conf_run = _make_run(avg_confidence=0.80)
    calls_made = []

    def mock_run_ask(alias, task, **kwargs):
        calls_made.append(alias)
        return high_conf_run

    with patch("orchestra.router.cascade_router._is_provider_healthy", return_value=True):
        with patch("orchestra.engine.runner.run_ask", side_effect=mock_run_ask):
            result = router.run("Fix the payment bug")

    assert result is high_conf_run
    assert calls_made == ["cdx-fast"]  # escalation not needed


# ---------------------------------------------------------------------------
# CascadeRouter.run — escalates when confidence low
# ---------------------------------------------------------------------------

def test_cascade_escalates_on_low_confidence():
    router = CascadeRouter(
        chain=["cdx-fast", "cdx-deep"],
        confidence_threshold=0.70,
        max_escalations=2,
    )
    low_conf_run = _make_run(avg_confidence=0.40)
    high_conf_run = _make_run(avg_confidence=0.85)
    alias_sequence = []

    def mock_run_ask(alias, task, **kwargs):
        alias_sequence.append(alias)
        return high_conf_run if alias == "cdx-deep" else low_conf_run

    with patch("orchestra.router.cascade_router._is_provider_healthy", return_value=True):
        with patch("orchestra.engine.runner.run_ask", side_effect=mock_run_ask):
            result = router.run("Fix the payment bug")

    assert alias_sequence == ["cdx-fast", "cdx-deep"]
    assert result is high_conf_run


# ---------------------------------------------------------------------------
# CascadeRouter.run — respects max_escalations cap
# ---------------------------------------------------------------------------

def test_cascade_respects_max_escalations():
    router = CascadeRouter(
        chain=["cdx-fast", "cdx-deep", "cld-deep"],
        confidence_threshold=0.95,  # very high — will always want to escalate
        max_escalations=1,
    )
    low_run = _make_run(avg_confidence=0.30)
    calls = []

    def mock_run_ask(alias, task, **kwargs):
        calls.append(alias)
        return low_run

    with patch("orchestra.router.cascade_router._is_provider_healthy", return_value=True):
        with patch("orchestra.engine.runner.run_ask", side_effect=mock_run_ask):
            router.run("Analyze the architecture")

    # With max_escalations=1: escalations_done increments after each alias.
    # cdx-fast: done=0 ≤ 1 → runs → done=1
    # cdx-deep: done=1 ≤ 1 → runs → done=2
    # cld-deep: done=2 > 1 → break
    assert "cld-deep" not in calls


# ---------------------------------------------------------------------------
# CascadeRouter.run — skips degraded provider, continues to next
# ---------------------------------------------------------------------------

def test_cascade_skips_degraded_provider():
    router = CascadeRouter(
        chain=["cdx-fast", "cdx-deep"],
        confidence_threshold=0.70,
        max_escalations=2,
    )
    good_run = _make_run(avg_confidence=0.80)
    calls = []

    def mock_run_ask(alias, task, **kwargs):
        calls.append(alias)
        return good_run

    def mock_healthy(alias):
        return alias != "cdx-fast"  # cdx-fast is degraded

    with patch("orchestra.router.cascade_router._is_provider_healthy", side_effect=mock_healthy):
        with patch("orchestra.engine.runner.run_ask", side_effect=mock_run_ask):
            result = router.run("Fix the bug")

    assert "cdx-fast" not in calls
    assert "cdx-deep" in calls
    assert result is good_run


# ---------------------------------------------------------------------------
# CascadeRouter.run — all degraded → last resort runs chain[0]
# ---------------------------------------------------------------------------

def test_cascade_last_resort_when_all_degraded():
    router = CascadeRouter(
        chain=["cdx-fast", "cdx-deep"],
        confidence_threshold=0.70,
        max_escalations=2,
    )
    fallback_run = _make_run(avg_confidence=0.50)

    def mock_run_ask(alias, task, **kwargs):
        return fallback_run

    with patch("orchestra.router.cascade_router._is_provider_healthy", return_value=False):
        with patch("orchestra.engine.runner.run_ask", side_effect=mock_run_ask) as mock_ask:
            result = router.run("Fix the bug")

    assert mock_ask.called
    assert result is fallback_run


# ---------------------------------------------------------------------------
# _outcome_adjusted_threshold
# ---------------------------------------------------------------------------

def test_outcome_adjusted_threshold_lowers_when_strong_history():
    mock_outcomes = [
        {"task": "Fix payment bug", "confidence": 0.85, "mode": "ask"}
        for _ in range(6)
    ]

    with patch("orchestra.router.outcome_router._load_outcomes", return_value=mock_outcomes):
        with patch("orchestra.router.outcome_router._outcomes_path", return_value=MagicMock()):
            threshold = _outcome_adjusted_threshold("cdx-fast", "Fix payment bug", 0.70)

    assert threshold < 0.70  # lowered by up to 0.05
    assert threshold >= 0.50  # never drops below floor


def test_outcome_adjusted_threshold_unchanged_with_sparse_history():
    mock_outcomes = [
        {"task": "Fix payment bug", "confidence": 0.85, "mode": "ask"}
        for _ in range(3)  # < 5 → no adjustment
    ]

    with patch("orchestra.router.outcome_router._load_outcomes", return_value=mock_outcomes):
        with patch("orchestra.router.outcome_router._outcomes_path", return_value=MagicMock()):
            threshold = _outcome_adjusted_threshold("cdx-fast", "Fix payment bug", 0.70)

    assert threshold == 0.70


def test_outcome_adjusted_threshold_unchanged_with_weak_history():
    mock_outcomes = [
        {"task": "Fix payment bug", "confidence": 0.50, "mode": "ask"}
        for _ in range(6)  # enough count but avg_conf = 0.50 < 0.75
    ]

    with patch("orchestra.router.outcome_router._load_outcomes", return_value=mock_outcomes):
        with patch("orchestra.router.outcome_router._outcomes_path", return_value=MagicMock()):
            threshold = _outcome_adjusted_threshold("cdx-fast", "Fix payment bug", 0.70)

    assert threshold == 0.70


def test_outcome_adjusted_threshold_unchanged_on_exception():
    with patch("orchestra.router.outcome_router._load_outcomes", side_effect=RuntimeError("disk error")):
        threshold = _outcome_adjusted_threshold("cdx-fast", "Fix payment bug", 0.70)

    assert threshold == 0.70
