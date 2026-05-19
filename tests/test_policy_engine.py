"""Unit tests for the declarative PolicyEngine."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestra.engine.policy_engine import PolicyEngine, PolicyDecision
from orchestra.engine.autonomy_policy import AutonomyPolicy


class TestPolicyEngineProviderGate:
    """Provider trust-level evaluations."""

    def test_auto_provider(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
version: "1.0"
approval:
  providers:
    claude: auto
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_provider("cld-deep")
        assert d.action == "allow"
        assert "claude" in d.reason

    def test_gated_provider(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    gemini: gated
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_provider("gmn-pro")
        assert d.action == "gate"
        assert "gemini" in d.reason

    def test_deny_provider(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    deepseek: deny
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_provider("ds-chat")
        assert d.action == "deny"
        assert "deepseek" in d.reason

    def test_sandbox_provider(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    ollama: sandbox_only
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_provider("oll-coder")
        assert d.action == "sandbox"
        assert "ollama" in d.reason

    def test_unknown_provider_defaults_allow(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: auto
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_provider("unknown-alias")
        assert d.action == "allow"


class TestPolicyEngineToolGate:
    """Tool permission-level evaluations."""

    def test_auto_tool(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
tools:
  file_read: auto
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_tool("file_read")
        assert d.action == "allow"

    def test_gated_tool(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
tools:
  git_push: gated
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_tool("git_push")
        assert d.action == "gate"

    def test_deny_tool(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
tools:
  shell_exec: deny
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_tool("shell_exec")
        assert d.action == "deny"

    def test_unknown_tool_defaults_allow(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
tools:
  file_read: auto
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_tool("unknown_tool")
        assert d.action == "allow"


class TestPolicyEngineBudgetGate:
    """Budget limit evaluations."""

    def test_within_budget(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
budget:
  max_usd_per_run: 5.0
  max_usd_per_day: 20.0
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_budget(3.0, 15.0)
        assert d.action == "allow"

    def test_exceeds_run_budget(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
budget:
  max_usd_per_run: 2.0
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_budget(2.5, 0.0)
        assert d.action == "deny"
        assert "run cost" in d.reason

    def test_exceeds_daily_budget(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
budget:
  max_usd_per_day: 10.0
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_budget(1.0, 10.0)
        assert d.action == "deny"
        assert "daily cost" in d.reason

    def test_zero_means_unlimited(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
budget:
  max_usd_per_run: 0
  max_usd_per_day: 0
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_budget(999.0, 999.0)
        assert d.action == "allow"


class TestPolicyEngineEvaluateRun:
    """Run-level orchestration of all gates."""

    def test_all_clear(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: auto
  mode_gates: []
tools:
  file_read: auto
budget:
  max_usd_per_run: 10.0
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_run(
            mode="ask",
            task="hello",
            estimated_cost_usd=1.0,
            provider_aliases=["cld-deep"],
            daily_spend_usd=0.0,
        )
        assert d.action == "allow"

    def test_provider_deny_wins_first(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    gemini: deny
  mode_gates: []
budget:
  max_usd_per_run: 0.1
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_run(
            mode="ask",
            task="hello",
            estimated_cost_usd=999.0,
            provider_aliases=["gmn-pro"],
            daily_spend_usd=0.0,
        )
        assert d.action == "deny"
        assert "provider" in d.reason

    def test_budget_deny_after_provider_allow(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: auto
  mode_gates: []
budget:
  max_usd_per_run: 1.0
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_run(
            mode="ask",
            task="hello",
            estimated_cost_usd=5.0,
            provider_aliases=["cld-deep"],
            daily_spend_usd=0.0,
        )
        assert d.action == "deny"
        assert "run cost" in d.reason

    def test_mode_gate_triggers(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: auto
  mode_gates:
    - critical
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_run(
            mode="critical",
            task="hello",
            estimated_cost_usd=0.1,
            provider_aliases=["cld-deep"],
            daily_spend_usd=0.0,
        )
        assert d.action == "gate"
        assert "mode" in d.reason

    def test_privileged_action_triggers(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: auto
  mode_gates: []
  privileged_actions:
    - git push
""")
        engine = PolicyEngine(policy_file)
        d = engine.evaluate_run(
            mode="ask",
            task="please git push to production",
            estimated_cost_usd=0.1,
            provider_aliases=["cld-deep"],
            daily_spend_usd=0.0,
        )
        assert d.action == "gate"
        assert "privileged" in d.reason


class TestPolicyEngineLegacyFallback:
    """When policy.yaml is absent, everything is allowed."""

    def test_no_policy_file(self) -> None:
        engine = PolicyEngine(Path("/nonexistent/policy.yaml"))
        assert engine.evaluate_provider("cld-deep").action == "allow"
        assert engine.evaluate_tool("shell_exec").action == "allow"
        assert engine.evaluate_budget(999.0, 999.0).action == "allow"
        assert engine.evaluate_run(
            mode="critical",
            task="git push",
            estimated_cost_usd=999.0,
            provider_aliases=["cld-deep"],
            daily_spend_usd=999.0,
        ).action == "allow"

    def test_empty_policy_file(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("")
        engine = PolicyEngine(policy_file)
        assert engine.evaluate_provider("cld-deep").action == "allow"


class TestPolicyEngineCaching:
    """File-system caching behaviour."""

    def test_cache_reloads_on_mtime_change(self, tmp_path: Path) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("""
approval:
  providers:
    claude: auto
""")
        engine = PolicyEngine(policy_file)
        assert engine.evaluate_provider("cld-deep").action == "allow"

        # Mutate file
        policy_file.write_text("""
approval:
  providers:
    claude: deny
""")
        # Force mtime change (some filesystems cache at 1s resolution)
        import time
        time.sleep(0.05)

        engine2 = PolicyEngine(policy_file)
        assert engine2.evaluate_provider("cld-deep").action == "deny"


class TestAutonomyPolicyValidation:
    """Config loading must reject malformed thresholds and invalid actions."""

    def test_nan_threshold_falls_back_to_default(self, caplog):
        with caplog.at_level("WARNING", logger="orchestra.engine.autonomy_policy"):
            policy = AutonomyPolicy.from_config()
            # Monkey-patch via object construction, not config file
            policy = AutonomyPolicy(
                cost_threshold_usd=float("nan"),
                disagreement_entropy_threshold=float("inf"),
                semantic_novelty_std_threshold=-1.0,
            )
        # Direct construction does NOT trigger from_config validation.
        # Test from_config path by mocking config.autonomy_policy_config.
        from unittest.mock import patch
        with patch("orchestra.engine.autonomy_policy.config.autonomy_policy_config") as mock_cfg:
            mock_cfg.return_value = {
                "cost_threshold_usd": "nan",
                "disagreement_entropy_threshold": "inf",
                "semantic_novelty_std_threshold": -3.0,
            }
            with caplog.at_level("WARNING", logger="orchestra.engine.autonomy_policy"):
                policy = AutonomyPolicy.from_config()
            assert policy.cost_threshold_usd == 0.50
            assert policy.disagreement_entropy_threshold == 0.60
            assert policy.semantic_novelty_std_threshold == 3.0
            assert any("cost_threshold_usd" in rec.message for rec in caplog.records)

    def test_invalid_default_action_falls_back(self, caplog):
        from unittest.mock import patch
        with patch("orchestra.engine.autonomy_policy.config.autonomy_policy_config") as mock_cfg:
            mock_cfg.return_value = {"default_action": "destroy"}
            with caplog.at_level("WARNING", logger="orchestra.engine.autonomy_policy"):
                policy = AutonomyPolicy.from_config()
            assert policy.default_action == "escalate"
            assert any("default_action" in rec.message for rec in caplog.records)

    def test_empty_config_warns(self, caplog):
        from unittest.mock import patch
        with patch("orchestra.engine.autonomy_policy.config.autonomy_policy_config") as mock_cfg:
            mock_cfg.return_value = {}
            with caplog.at_level("WARNING", logger="orchestra.engine.autonomy_policy"):
                policy = AutonomyPolicy.from_config()
            assert policy.cost_threshold_usd == 0.50
            assert any("empty dict" in rec.message for rec in caplog.records)

    def test_zero_cost_threshold_escalates_positive_cost(self):
        """Zero threshold must escalate any positive estimated cost."""
        policy = AutonomyPolicy(cost_threshold_usd=0.0)
        should_escalate, reason = policy.should_escalate(estimated_cost_usd=0.0)
        assert should_escalate is False
        should_escalate, reason = policy.should_escalate(estimated_cost_usd=0.01)
        assert should_escalate is True
        assert "zero threshold" in reason


class TestAutonomyPolicyWordBoundary:
    """Privileged-action matching must use word boundaries for single-word terms."""

    def test_prod_matches_standalone(self):
        policy = AutonomyPolicy(privileged_actions=["prod"])
        should_escalate, reason = policy.should_escalate(task="deploy to prod environment")
        assert should_escalate is True
        assert "prod" in reason

    def test_product_does_not_match_prod(self):
        policy = AutonomyPolicy(privileged_actions=["prod"])
        should_escalate, _ = policy.should_escalate(task="build a product catalog")
        assert should_escalate is False

    def test_the_is_not_privileged(self):
        policy = AutonomyPolicy(privileged_actions=["the"])
        should_escalate, _ = policy.should_escalate(task="the quick brown fox")
        assert should_escalate is False

    def test_git_push_multi_word(self):
        policy = AutonomyPolicy(privileged_actions=["git push"])
        should_escalate, reason = policy.should_escalate(task="please git push origin main")
        assert should_escalate is True
        assert "git push" in reason

    def test_git_push_does_not_match_git_pull(self):
        policy = AutonomyPolicy(privileged_actions=["git push"])
        should_escalate, _ = policy.should_escalate(task="please git pull origin main")
        assert should_escalate is False


class TestAliasToProvider:
    """Alias-to-provider heuristic mapping."""

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("cdx-deep", "codex"),
            ("gmn-pro", "gemini"),
            ("cld-fast", "claude"),
            ("oll-coder", "ollama"),
            ("ds-chat", "deepseek"),
            ("kimi-moonshot", "kimi"),
            ("custom-agent", "custom"),
        ],
    )
    def test_mappings(self, alias: str, expected: str) -> None:
        assert PolicyEngine._alias_to_provider(alias) == expected
