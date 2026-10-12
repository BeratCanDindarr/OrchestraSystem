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
    from orchestra.engine.runners import dual_runner as runner
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
    from orchestra.engine.runners import dual_runner as runner
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
    from orchestra.engine.runners import dual_runner as runner
    captured = []

    def fake_run_parallel(run, pairs, **kwargs):
        captured.extend([alias for alias, _ in pairs])

    with patch.object(runner, "run_parallel", fake_run_parallel), \
         patch.object(runner, "_finalize", lambda r: None), \
         patch.object(runner.artifacts, "write_manifest", lambda r: None), \
         patch.object(runner, "_record_round1_review", lambda r: None), \
         patch.object(runner, "_run_verification_loop", lambda r, p, **k: None), \
         patch.object(runner, "_synthesize_if_possible", lambda r, p, **k: None):
        runner.run_dual("test task", agents=["cdx-deep"],
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


# ── a tier names agents; the agents that answer must be two different models ─

def _only_available(*families):
    """Patch provider availability: only the named provider families can run."""
    from contextlib import ExitStack

    from orchestra.providers import fallback

    stack = ExitStack()
    for provider in {entry[0] for entry in fallback.ALIAS_TABLE.values()}:
        # On the instance: it is the shared object the alias table holds.
        stack.enter_context(patch.object(provider, "is_available", return_value=provider.name in families))
    return stack


@pytest.fixture(autouse=True)
def _every_provider_can_run(request):
    """Which CLIs this machine has installed must not decide what these tests see."""
    if "cannot_run" in request.node.name or "can_run_is_left" in request.node.name:
        yield
        return
    with _only_available("codex", "claude", "gemini", "kimi", "ollama"):
        yield


def test_a_tier_whose_second_agent_cannot_run_does_not_compare_a_model_with_itself():
    from orchestra.providers import fallback
    from orchestra.router.classifier import agents_for_tier

    with _only_available("codex", "claude"), \
         patch("orchestra.config.tier_config", return_value={"heavy": {"agents": ["cdx-deep", "gmn-pro"]}}):
        agents = agents_for_tier("heavy")

    models = {fallback.ALIAS_TABLE[a][0].model_label(fallback.ALIAS_TABLE[a][1]) for a in agents}
    assert len(agents) == 2 and len(models) == 2
    assert agents[0] == "cdx-deep"
    assert all(fallback.ALIAS_TABLE[a][0].name in ("codex", "claude") for a in agents)


def test_a_tier_whose_agents_can_run_is_left_as_configured():
    from orchestra.router.classifier import agents_for_tier

    with _only_available("codex", "claude", "gemini"), \
         patch("orchestra.config.tier_config", return_value={"heavy": {"agents": ["cdx-deep", "gmn-pro"]}}):
        assert agents_for_tier("heavy") == ["cdx-deep", "gmn-pro"]
