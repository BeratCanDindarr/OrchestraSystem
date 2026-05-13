"""Tests for Tier-Based Model Routing (Protégé Pattern)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── classify_to_tier ─────────────────────────────────────────────────────────

def test_cheap_task_maps_to_light():
    from orchestra.router.classifier import classify_to_tier
    # "summarize" is in policy.toml cheap keywords
    assert classify_to_tier("summarize this file") == "light"


def test_deep_task_maps_to_heavy():
    from orchestra.router.classifier import classify_to_tier
    assert classify_to_tier("research the architecture of this system") == "heavy"


def test_balanced_task_maps_to_medium():
    from orchestra.router.classifier import classify_to_tier
    # A plain balanced task (no cheap/deep keywords, short)
    assert classify_to_tier("update the README") == "medium"


def test_extreme_task_maps_to_heavy():
    from orchestra.router.classifier import classify_to_tier
    # A very long task triggers extreme → heavy
    long_task = "do X\n" * 25
    assert classify_to_tier(long_task) == "heavy"


# ── agents_for_tier ───────────────────────────────────────────────────────────

def test_light_tier_default_agents():
    from orchestra.router.classifier import agents_for_tier
    agents = agents_for_tier("light")
    assert agents == ["cdx-fast", "gmn-fast"]


def test_medium_tier_default_agents():
    from orchestra.router.classifier import agents_for_tier
    agents = agents_for_tier("medium")
    assert agents == ["cdx-fast", "gmn-pro"]


def test_heavy_tier_default_agents():
    from orchestra.router.classifier import agents_for_tier
    agents = agents_for_tier("heavy")
    assert agents == ["cdx-deep", "gmn-pro"]


def test_unknown_tier_falls_back_to_default():
    from orchestra.router.classifier import agents_for_tier
    agents = agents_for_tier("nonexistent")
    assert len(agents) >= 2


def test_config_tier_override(monkeypatch):
    """Tier agents come from config when present."""
    from orchestra.router import classifier
    monkeypatch.setattr(classifier.config, "tier_config", lambda: {
        "light": {"agents": ["custom-fast", "custom-flash"]}
    })
    agents = classifier.agents_for_tier("light")
    assert agents == ["custom-fast", "custom-flash"]


# ── run_dual agents parameter ─────────────────────────────────────────────────

def test_run_dual_uses_provided_agents():
    """run_dual passes provided agents to run_parallel."""
    from orchestra.engine import runner
    captured = []

    def fake_run_parallel(run, pairs, **kwargs):
        captured.extend([alias for alias, _ in pairs])

    with patch.object(runner, "run_parallel", fake_run_parallel), \
         patch.object(runner, "_finalize", lambda r: None), \
         patch.object(runner.artifacts, "write_manifest", lambda r: None), \
         patch.object(runner, "_record_round1_review", lambda r: None), \
         patch.object(runner, "_run_verification_loop", lambda r, p, **k: None), \
         patch.object(runner, "_synthesize_if_possible", lambda r, p, **k: None):
        runner.run_dual("test task", agents=["cdx-fast", "gmn-fast"],
                        show_live=False, emit_console=False)

    assert captured == ["cdx-fast", "gmn-fast"]


def test_run_dual_defaults_when_no_agents():
    """run_dual falls back to cdx-deep + gmn-pro when agents=None."""
    from orchestra.engine import runner
    captured = []

    def fake_run_parallel(run, pairs, **kwargs):
        captured.extend([alias for alias, _ in pairs])

    with patch.object(runner, "run_parallel", fake_run_parallel), \
         patch.object(runner, "_finalize", lambda r: None), \
         patch.object(runner.artifacts, "write_manifest", lambda r: None), \
         patch.object(runner, "_record_round1_review", lambda r: None), \
         patch.object(runner, "_run_verification_loop", lambda r, p, **k: None), \
         patch.object(runner, "_synthesize_if_possible", lambda r, p, **k: None):
        runner.run_dual("test task", show_live=False, emit_console=False)

    assert captured == ["cdx-deep", "gmn-pro"]


def test_run_dual_defaults_when_agents_list_too_short():
    """run_dual falls back when only 1 agent provided."""
    from orchestra.engine import runner
    captured = []

    def fake_run_parallel(run, pairs, **kwargs):
        captured.extend([alias for alias, _ in pairs])

    with patch.object(runner, "run_parallel", fake_run_parallel), \
         patch.object(runner, "_finalize", lambda r: None), \
         patch.object(runner.artifacts, "write_manifest", lambda r: None), \
         patch.object(runner, "_record_round1_review", lambda r: None), \
         patch.object(runner, "_run_verification_loop", lambda r, p, **k: None), \
         patch.object(runner, "_synthesize_if_possible", lambda r, p, **k: None):
        runner.run_dual("test task", agents=["only-one"],
                        show_live=False, emit_console=False)

    assert captured == ["cdx-deep", "gmn-pro"]


# ── service tier routing integration ─────────────────────────────────────────

def test_service_auto_dual_uses_tier_agents():
    """service.run_task auto→dual path passes tier-based agents to run_dual."""
    import orchestra.service as svc

    collected_agents = []

    def fake_run_dual(prompt, *, agents=None, **kwargs):
        collected_agents.extend(agents or [])
        m = MagicMock()
        m.to_manifest.return_value = {}
        m.latest_review_status = "pass"
        return m

    fake_decision = MagicMock()
    fake_decision.mode = "dual"
    fake_decision.alias = None
    fake_decision.require_approval = False

    with patch.object(svc, "run_dual", fake_run_dual), \
         patch.object(svc, "route_task", return_value=fake_decision), \
         patch.object(svc, "classify_to_tier", return_value="light"):
        svc.run_task(mode="auto", task="summarize this")

    # light → cdx-fast + gmn-fast
    assert collected_agents == ["cdx-fast", "gmn-fast"]
