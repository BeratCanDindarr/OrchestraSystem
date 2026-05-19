"""Edge-case tests for Q1.1-Q1.7 changes (synthesis, autonomy, memory, personas)."""
import pytest
import sqlite3
import tempfile
import os

from unittest.mock import patch, MagicMock

from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus
from orchestra.engine.synthesizer import (
    compute_divergence_index,
    build_synthesis_prompt,
    outputs_sufficient,
    select_synthesis_alias,
    measure_synthesis_quality,
    format_provenance,
)
from orchestra.engine.autonomy_policy import AutonomyPolicy
from orchestra.engine.policy_engine import PolicyDecision
from orchestra.storage.memory import SemanticMemory


class TestSynthesisEdgeCases:
    """Edge cases for synthesis + failure taxonomy (Q1.1, Q1.2, Q1.6)."""

    def test_outputs_sufficient_all_failed(self):
        """Synthesis should be attempted even when all agents failed."""
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.FAILED, error="timeout", failure_kind="MODEL_TIMEOUT")
        a2 = AgentRun(alias="a2", provider="p", model="m", status=AgentStatus.FAILED, error="timeout", failure_kind="MODEL_TIMEOUT")
        assert outputs_sufficient([a1, a2]) is True

    def test_outputs_sufficient_single_agent(self):
        """Synthesis should NOT be attempted with a single agent."""
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.COMPLETED, stdout_log="x" * 200, confidence=0.9)
        assert outputs_sufficient([a1]) is False

    def test_divergence_index_all_failed(self):
        """All failed → divergence_index should be high (driven by failure_penalty)."""
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.FAILED, failure_kind="MODEL_TIMEOUT")
        a2 = AgentRun(alias="a2", provider="p", model="m", status=AgentStatus.FAILED, failure_kind="MODEL_TIMEOUT")
        idx = compute_divergence_index([a1, a2])
        assert idx >= 0.5, f"Expected high divergence for all-failed, got {idx}"

    def test_divergence_index_one_completed_one_failed(self):
        """1 completed + 1 failed → divergence should reflect structural failure."""
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.COMPLETED, stdout_log="output one", confidence=0.9)
        a2 = AgentRun(alias="a2", provider="p", model="m", status=AgentStatus.FAILED, failure_kind="MODEL_TIMEOUT")
        idx = compute_divergence_index([a1, a2])
        assert idx >= 0.3, f"Expected moderate-high divergence, got {idx}"

    def test_select_synthesis_alias_all_failed(self):
        """When all agents failed, synthesizer should fallback to deep reasoning."""
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.FAILED, failure_kind="MODEL_TIMEOUT")
        a2 = AgentRun(alias="a2", provider="p", model="m", status=AgentStatus.FAILED, failure_kind="MODEL_TIMEOUT")
        alias = select_synthesis_alias([a1, a2], 100)
        assert alias == "cld-deep"

    def test_build_synthesis_prompt_truncates_large_failed_output(self):
        """Failed agent stdout larger than 4000 chars should be truncated."""
        huge = "x" * 500_000
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.COMPLETED, stdout_log="ok", confidence=0.9)
        a2 = AgentRun(alias="a2", provider="p", model="m", status=AgentStatus.FAILED, stdout_log=huge, failure_kind="TOOL_EXECUTION_FAILED")
        prompt = build_synthesis_prompt([a1, a2])
        assert "...[truncated]" in prompt
        assert len(prompt) < len(huge)

    def test_build_synthesis_prompt_legacy_agent_no_failure_kind(self):
        """Legacy AgentRun objects without failure_kind should not crash."""
        a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.FAILED, stdout_log="err")
        # Simulate legacy object by deleting the attribute
        object.__delattr__(a1, "failure_kind") if hasattr(a1, "failure_kind") else None
        prompt = build_synthesis_prompt([a1])
        assert "unknown" in prompt

    def test_measure_synthesis_quality_sections(self):
        """Quality score should reflect presence of required sections."""
        sample = "## Answer\nfoo\n## Key Signals\nbar\n## Dissent\nbaz\n## Failure Report\nqux\n## Provenance\nquux"
        quality = measure_synthesis_quality(sample, 0.3)
        assert quality["score"] > 0.8
        assert quality["has_dissent"] is True
        assert quality["has_provenance"] is True

    def test_measure_synthesis_quality_empty(self):
        """Empty synthesis should score near zero."""
        quality = measure_synthesis_quality("", 0.0)
        assert quality["score"] < 0.3

    def test_measure_synthesis_quality_high_divergence_no_dissent_penalty(self):
        """High divergence without dissent should be penalized."""
        sample = "## Answer\nfoo\n## Key Signals\nbar"
        quality = measure_synthesis_quality(sample, 0.8)
        # Should be penalized because divergence > 0.5 but no dissent/provenance
        assert quality["score"] < 0.7


class TestAutonomyPolicyEdgeCases:
    """Edge cases for AutonomyPolicy (Q1.3)."""

    def test_zero_cost_threshold_escalates_everything(self):
        """cost_threshold_usd=0 means every non-zero cost run escalates."""
        policy = AutonomyPolicy(cost_threshold_usd=0.0, mode_gates=[])
        should, reason = policy.should_escalate(estimated_cost_usd=0.01)
        assert should is True
        assert "cost_gate" in reason

    def test_privileged_action_substring_match(self):
        """Privileged action keywords should trigger escalation."""
        policy = AutonomyPolicy(mode_gates=[])
        should, reason = policy.should_escalate(task="Please git push to production")
        assert should is True
        assert "privileged_gate" in reason

    def test_empty_privileged_actions_no_false_positives(self):
        """Empty privileged_actions list should never trigger on task content."""
        policy = AutonomyPolicy(privileged_actions=[], mode_gates=[])
        should, _ = policy.should_escalate(task="git push deploy delete everything")
        assert should is False

    def test_mode_gate_triggered(self):
        """Modes in mode_gates should always escalate."""
        policy = AutonomyPolicy(mode_gates=["critical"])
        should, reason = policy.should_escalate(mode="critical")
        assert should is True
        assert "mode_gate" in reason

    def test_tool_call_privileged_check(self):
        """Tool call containing privileged keyword should be blocked."""
        policy = AutonomyPolicy()
        allowed, reason = policy.check_before_tool_call("shell", {"command": "git push origin main"})
        assert allowed is False
        assert "privileged_tool" in reason


class TestMemoryDecayEdgeCases:
    """Edge cases for importance-weighted memory decay (Q1.4)."""

    def test_cleanup_preserves_pinned_items(self):
        """cleanup_expired must NEVER delete pinned items."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "mem.db")
            mem = SemanticMemory(db_path=db_path)

            # Add items and force them to be expired via direct SQL
            mem.add("pinned content", pinned=True)
            mem.add("expired content", pinned=False)
            with sqlite3.connect(db_path) as conn:
                conn.execute("UPDATE embeddings SET expires_at = datetime('now', '-1 hours')")
                conn.commit()

            deleted = mem.cleanup_expired()
            assert deleted == 1

            ledger = mem.get_memory_ledger()
            assert len(ledger) == 1
            assert ledger[0]["content"] == "pinned content"
            assert ledger[0]["pinned"] == 1

    def test_pin_unpin_cycle(self):
        """Pinning and unpinning should correctly toggle expiration."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "mem.db")
            mem = SemanticMemory(db_path=db_path)

            mem.add("content", ttl_hours=1)
            ledger = mem.get_memory_ledger()
            item_id = ledger[0]["id"]

            assert mem.pin(item_id) is True
            pinned = mem.search("content", limit=1)
            assert pinned[0]["pinned"] is True

            assert mem.unpin(item_id) is True
            unpinned = mem.search("content", limit=1)
            assert unpinned[0]["pinned"] is False

    def test_citation_boost_in_search(self):
        """Highly-cited items should get a similarity boost."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "mem.db")
            mem = SemanticMemory(db_path=db_path)

            mem.add("alpha", summary="alpha")
            mem.add("beta", summary="beta")
            ledger = mem.get_memory_ledger()
            alpha_id = next(r["id"] for r in ledger if r["content"] == "alpha")
            mem.increment_citation(alpha_id)
            mem.increment_citation(alpha_id)
            mem.increment_citation(alpha_id)

            results = mem.search("alpha", limit=2)
            # The cited item should rank first due to boost
            assert results[0]["citation_count"] >= 3

    def test_memory_ledger_contains_all_fields(self):
        """Memory ledger should expose ownership and sensitivity fields."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "mem.db")
            mem = SemanticMemory(db_path=db_path)
            mem.add("content", sensitivity_label="pii", owner="user_42", source="run_7")
            ledger = mem.get_memory_ledger()
            assert len(ledger) == 1
            assert ledger[0]["sensitivity_label"] == "pii"
            assert ledger[0]["owner"] == "user_42"
            assert ledger[0]["source"] == "run_7"


class TestPersonaEdgeCases:
    """Edge cases for persona configuration (Q1.5)."""

    def test_run_dual_with_empty_personas_config(self):
        """run_dual should fall back to hard-coded defaults when personas config is empty."""
        with patch("orchestra.config.personas_config", return_value={}):
            from orchestra.engine.runner import run_dual
            with patch("orchestra.engine.runner.run_parallel") as mock_parallel:
                def _add_agents(run, pairs, **kwargs):
                    for alias, _ in pairs:
                        a = AgentRun(alias=alias, provider="p", model="m")
                        a.status = AgentStatus.COMPLETED
                        a.stdout_log = "ok"
                        run.agents.append(a)
                mock_parallel.side_effect = _add_agents
                run = run_dual(prompt="test")
                aliases = [a.alias for a in run.agents]
                assert "cdx-deep" in aliases or "gmn-pro" in aliases

    def test_run_dual_same_alias_twice(self):
        """run_dual should handle agents=[same, same] gracefully."""
        from orchestra.engine.runner import run_dual
        with patch("orchestra.engine.runner.run_parallel") as mock_parallel, \
             patch("orchestra.engine.runner._record_round1_review") as mock_review, \
             patch("orchestra.engine.runner._run_verification_loop") as mock_verify:
            def _add_agents(run, pairs, **kwargs):
                for alias, _ in pairs:
                    a = AgentRun(alias=alias, provider="p", model="m")
                    a.status = AgentStatus.COMPLETED
                    a.stdout_log = "ok"
                    run.agents.append(a)
            mock_parallel.side_effect = _add_agents
            mock_review.return_value = None
            mock_verify.return_value = None
            run = run_dual(prompt="test", agents=["cdx-deep", "cdx-deep"])
            # run_parallel adds 2 agents; verification loop adds none when mocked
            assert len(run.agents) == 2
            assert run.agents[0].alias == run.agents[1].alias

    def test_select_synthesis_alias_empty_agents(self):
        """select_synthesis_alias with empty agents should not crash."""
        alias = select_synthesis_alias([], 100)
        assert alias == "cld-deep"


class TestLocalFirstEdgeCases:
    """Edge cases for local-first / air-gapped mode (Q1.8)."""

    def test_select_synthesis_alias_local_first(self):
        """In local-first mode, synthesis should use the configured local alias."""
        with patch("orchestra.config.local_first_enabled", return_value=True):
            with patch("orchestra.config.local_first_config", return_value={"synthesis_alias": "oll-coder"}):
                alias = select_synthesis_alias([], 100)
                assert alias == "oll-coder"

    def test_select_synthesis_alias_local_first_fallback(self):
        """In local-first mode with missing synthesis_alias, should fallback to oll-analyst."""
        with patch("orchestra.config.local_first_enabled", return_value=True):
            with patch("orchestra.config.local_first_config", return_value={}):
                alias = select_synthesis_alias([], 100)
                assert alias == "oll-analyst"

    def test_run_dual_local_first_personas(self):
        """run_dual should use local personas when local-first is enabled."""
        from orchestra.engine.runner import run_dual
        with patch("orchestra.engine.runner.config") as mock_config:
            mock_config.local_first_enabled.return_value = True
            mock_config.local_first_config.return_value = {
                "persona_implementer": "oll-coder",
                "persona_researcher": "oll-deep",
            }
            mock_config.personas_config.return_value = {}
            with patch("orchestra.engine.runner.run_parallel") as mock_parallel, \
                 patch("orchestra.engine.runner._record_round1_review") as mock_review, \
                 patch("orchestra.engine.runner._run_verification_loop") as mock_verify:
                def _add_agents(run, pairs, **kwargs):
                    for alias, _ in pairs:
                        a = AgentRun(alias=alias, provider="p", model="m")
                        a.status = AgentStatus.COMPLETED
                        a.stdout_log = "ok"
                        run.agents.append(a)
                mock_parallel.side_effect = _add_agents
                mock_review.return_value = None
                mock_verify.return_value = None
                run = run_dual(prompt="test")
                aliases = [a.alias for a in run.agents]
                assert "oll-coder" in aliases
                assert "oll-deep" in aliases

    def test_run_critical_local_first_personas(self):
        """run_critical should use local fast personas when local-first is enabled."""
        from orchestra.engine.runner import run_critical
        with patch("orchestra.engine.runner.config.local_first_enabled", return_value=True):
            with patch("orchestra.engine.runner.config.local_first_config", return_value={
                "persona_verifier": "oll-fast",
                "persona_implementer": "oll-coder",
            }):
                with patch("orchestra.engine.runner.run_parallel") as mock_parallel, \
                     patch("orchestra.engine.runner._record_round1_review") as mock_review, \
                     patch("orchestra.engine.runner._run_verification_loop") as mock_verify, \
                     patch("orchestra.engine.runner._evaluate_policy", return_value=PolicyDecision("allow", "mock", "test")):
                    def _add_agents(run, pairs, **kwargs):
                        for alias, _ in pairs:
                            a = AgentRun(alias=alias, provider="p", model="m")
                            a.status = AgentStatus.COMPLETED
                            a.stdout_log = "ok"
                            run.agents.append(a)
                    mock_parallel.side_effect = _add_agents
                    mock_review.return_value = None
                    mock_verify.return_value = None
                    run = run_critical(prompt="test")
                    aliases = [a.alias for a in run.agents]
                    assert "oll-fast" in aliases
                    assert "oll-coder" in aliases

    def test_run_dual_invalid_persona_fallback(self, caplog):
        """run_dual should fall back to defaults when personas reference unknown aliases."""
        from orchestra.engine.runner import run_dual
        with patch("orchestra.engine.runner.config.local_first_enabled", return_value=False):
            with patch("orchestra.engine.runner.config.personas_config", return_value={
                "implementer": "nonexistent-alias",
                "researcher": "also-bad",
            }):
                with patch("orchestra.engine.runner.run_parallel") as mock_parallel, \
                     patch("orchestra.engine.runner._record_round1_review") as mock_review, \
                     patch("orchestra.engine.runner._run_verification_loop") as mock_verify:
                    def _add_agents(run, pairs, **kwargs):
                        for alias, _ in pairs:
                            a = AgentRun(alias=alias, provider="p", model="m")
                            a.status = AgentStatus.COMPLETED
                            a.stdout_log = "ok"
                            run.agents.append(a)
                    mock_parallel.side_effect = _add_agents
                    mock_review.return_value = None
                    mock_verify.return_value = None
                    with caplog.at_level("WARNING", logger="orchestra.engine.runner"):
                        run = run_dual(prompt="test")
                    aliases = [a.alias for a in run.agents]
                    assert "cdx-deep" in aliases
                    assert "gmn-pro" in aliases
                    assert any("nonexistent-alias" in rec.message for rec in caplog.records)

    def test_cascade_router_local_first_chain(self):
        """Cascade router should use local-only chain in local-first mode."""
        from orchestra.router.cascade_router import _escalation_chain
        with patch("orchestra.router.cascade_router.config.local_first_enabled", return_value=True):
            chain = _escalation_chain()
            assert all(a.startswith("oll-") for a in chain), f"Expected local-only chain, got {chain}"


class TestSynthesizerEdgeCases:
    """Edge cases for synthesis and divergence index (cdx-deep review)."""

    def test_completed_agent_output_truncated(self):
        """Huge completed outputs must be truncated to protect context window."""
        from orchestra.engine.synthesizer import build_synthesis_prompt
        from orchestra.models import AgentRun, AgentStatus
        agent = AgentRun(
            alias="cdx-deep", provider="codex", model="gpt-5",
            status=AgentStatus.COMPLETED, stdout_log="x" * 50_000,
        )
        prompt = build_synthesis_prompt([agent])
        assert "...[truncated]" in prompt
        # Ensure the prompt does not contain the full 50k string
        assert len(prompt) < 60_000

    def test_divergence_index_single_completed_with_failure(self):
        """1 completed + 1 failed should yield a meaningful divergence index."""
        from orchestra.engine.synthesizer import compute_divergence_index
        from orchestra.models import AgentRun, AgentStatus
        a1 = AgentRun(
            alias="cdx-deep", provider="codex", model="gpt-5",
            status=AgentStatus.COMPLETED, stdout_log="ok",
        )
        a2 = AgentRun(
            alias="gmn-pro", provider="gemini", model="gemini-2",
            status=AgentStatus.FAILED, stdout_log="", error="timeout",
        )
        idx = compute_divergence_index([a1, a2])
        assert idx > 0.0
        assert idx <= 1.0

    def test_salvage_synthesis_when_all_fail(self):
        """All agents failed should still attempt synthesis (salvage)."""
        from orchestra.engine.synthesizer import select_synthesis_alias
        from orchestra.models import AgentRun, AgentStatus
        a1 = AgentRun(
            alias="cdx-deep", provider="codex", model="gpt-5",
            status=AgentStatus.FAILED, stdout_log="partial1", error="err",
        )
        a2 = AgentRun(
            alias="gmn-pro", provider="gemini", model="gemini-2",
            status=AgentStatus.FAILED, stdout_log="partial2", error="err",
        )
        alias = select_synthesis_alias([a1, a2], task_length=10)
        assert alias == "cld-deep"
