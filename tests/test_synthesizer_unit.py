"""Unit tests for orchestra.engine.synthesizer — deterministic, no external deps."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from orchestra.engine.synthesizer import (
    _output_similarity,
    build_synthesis_prompt,
    compute_divergence_index,
    outputs_sufficient,
    select_synthesis_alias,
)
from orchestra.models import AgentRun, AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_agent(
    alias: str = "cdx-deep",
    stdout_log: str = "",
    status: AgentStatus = AgentStatus.COMPLETED,
    confidence: float = 0.8,
    soft_failed: bool = False,
    validation_status: str = "not_run",
    error: str | None = None,
) -> AgentRun:
    return AgentRun(
        alias=alias,
        provider="codex",
        model="gpt-5.4",
        status=status,
        stdout_log=stdout_log,
        confidence=confidence,
        soft_failed=soft_failed,
        validation_status=validation_status,
        error=error,
    )


# ---------------------------------------------------------------------------
# compute_divergence_index
# ---------------------------------------------------------------------------
class TestComputeDivergenceIndex:
    def test_empty_agents_list(self):
        assert compute_divergence_index([]) == 0.0

    def test_single_agent(self):
        agent = _make_agent(stdout_log="hello world")
        assert compute_divergence_index([agent]) == 0.0

    def test_identical_outputs(self):
        a1 = _make_agent(alias="a1", stdout_log="the quick brown fox")
        a2 = _make_agent(alias="a2", stdout_log="the quick brown fox")
        idx = compute_divergence_index([a1, a2])
        assert idx == 0.0

    def test_completely_different_outputs(self):
        a1 = _make_agent(alias="a1", stdout_log="alpha beta gamma")
        a2 = _make_agent(alias="a2", stdout_log="xyz 123 qwerty")
        idx = compute_divergence_index([a1, a2])
        # Two completed, no failures: text_dissimilarity * 0.7 + failure_penalty * 0.3
        # Max text_dissimilarity = 1.0, so max idx = 0.7
        assert idx == 0.7

    def test_partial_overlap(self):
        a1 = _make_agent(alias="a1", stdout_log="hello world foo bar")
        a2 = _make_agent(alias="a2", stdout_log="hello world baz qux")
        idx = compute_divergence_index([a1, a2])
        # intersection={'hello','world'}=2, union=6, sim=2/6=0.333, dissimilarity=0.667
        # Weighted: 0.667 * 0.7 + 0.0 * 0.3 = 0.467
        assert idx == pytest.approx(0.467, abs=0.001)

    def test_all_failed_agents(self):
        a1 = _make_agent(alias="a1", status=AgentStatus.FAILED)
        a2 = _make_agent(alias="a2", status=AgentStatus.FAILED)
        idx = compute_divergence_index([a1, a2])
        assert idx == 1.0

    def test_one_failed_one_completed(self):
        a1 = _make_agent(alias="a1", status=AgentStatus.FAILED)
        a2 = _make_agent(alias="a2", stdout_log="some output here")
        idx = compute_divergence_index([a1, a2])
        # Single completed + failures: text_dissimilarity * 0.4 + failure_penalty * 0.6
        # failure_penalty = (1 + 0) / 2 = 0.5
        assert idx == round(min(1.0, 0.0 * 0.4 + 0.5 * 0.6), 3)

    def test_soft_failed_penalty(self):
        a1 = _make_agent(alias="a1", stdout_log="output one", soft_failed=True)
        a2 = _make_agent(alias="a2", stdout_log="output two", soft_failed=True)
        idx = compute_divergence_index([a1, a2])
        # Both soft-failed: no completed outputs, failure_penalty = (0 + 2*0.5)/2 = 0.5
        assert idx == 0.5

    def test_mixed_statuses_with_dissimilarity(self):
        a1 = _make_agent(alias="a1", stdout_log="foo bar baz")
        a2 = _make_agent(alias="a2", stdout_log="foo bar baz")
        a3 = _make_agent(alias="a3", status=AgentStatus.FAILED)
        idx = compute_divergence_index([a1, a2, a3])
        # text_dissimilarity = 0, failure_penalty = (1 + 0)/3 = 0.333
        assert idx == round(min(1.0, 0.0 * 0.7 + 0.333 * 0.3), 3)

    def test_three_agents_partial_divergence(self):
        a1 = _make_agent(alias="a1", stdout_log="a b c d")
        a2 = _make_agent(alias="a2", stdout_log="a b c d")
        a3 = _make_agent(alias="a3", stdout_log="e f g h")
        idx = compute_divergence_index([a1, a2, a3])
        # sim(a1,a2)=1.0, sim(a1,a3)=0.0, sim(a2,a3)=0.0 -> avg_sim = 1/3, dissimilarity = 2/3
        assert idx == round(min(1.0, (2 / 3) * 0.7 + 0.0 * 0.3), 3)


# ---------------------------------------------------------------------------
# _output_similarity
# ---------------------------------------------------------------------------
class TestOutputSimilarity:
    def test_both_empty(self):
        assert _output_similarity("", "") == 0.0

    def test_one_empty(self):
        assert _output_similarity("hello", "") == 0.0
        assert _output_similarity("", "world") == 0.0

    def test_identical(self):
        assert _output_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert _output_similarity("abc", "xyz") == 0.0

    def test_partial(self):
        # "hello world" vs "hello there" -> shared {hello}, total {hello, world, there}
        assert _output_similarity("hello world", "hello there") == pytest.approx(1 / 3)

    def test_case_insensitive(self):
        assert _output_similarity("Hello World", "hello world") == 1.0

    def test_whitespace_only(self):
        assert _output_similarity("   ", "   ") == 0.0


# ---------------------------------------------------------------------------
# outputs_sufficient
# ---------------------------------------------------------------------------
class TestOutputsSufficient:
    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={})
    def test_two_valid_completed(self, _mock_cfg):
        a1 = _make_agent(stdout_log="a" * 100)
        a2 = _make_agent(stdout_log="b" * 100)
        assert outputs_sufficient([a1, a2]) is True

    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={})
    def test_only_one_valid_completed(self, _mock_cfg):
        a1 = _make_agent(stdout_log="a" * 100)
        a2 = _make_agent(stdout_log="short")
        assert outputs_sufficient([a1, a2]) is True  # len(agents) >= 2

    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={})
    def test_both_failed_but_two_agents(self, _mock_cfg):
        a1 = _make_agent(status=AgentStatus.FAILED)
        a2 = _make_agent(status=AgentStatus.FAILED)
        assert outputs_sufficient([a1, a2]) is True

    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={})
    def test_less_than_two_agents(self, _mock_cfg):
        a1 = _make_agent(stdout_log="a" * 100)
        assert outputs_sufficient([a1]) is False

    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={"min_output_chars": 50})
    def test_respects_min_chars_from_config(self, _mock_cfg):
        a1 = _make_agent(stdout_log="a" * 60)
        a2 = _make_agent(stdout_log="b" * 60)
        assert outputs_sufficient([a1, a2]) is True

    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={"min_output_chars": 50})
    def test_below_min_chars(self, _mock_cfg):
        a1 = _make_agent(stdout_log="a" * 30)
        a2 = _make_agent(stdout_log="b" * 30)
        # Both below min, but still 2 agents exist
        assert outputs_sufficient([a1, a2]) is True

    def test_soft_failed_agent_excluded(self):
        a1 = _make_agent(stdout_log="a" * 100)
        a2 = _make_agent(stdout_log="b" * 100, soft_failed=True)
        # a2 is soft_failed so not counted as valid_completed, but len(agents) >= 2
        assert outputs_sufficient([a1, a2]) is True


# ---------------------------------------------------------------------------
# build_synthesis_prompt
# ---------------------------------------------------------------------------
class TestBuildSynthesisPrompt:
    def test_contains_both_agent_outputs(self):
        a1 = _make_agent(alias="cdx-deep", stdout_log="output from codex")
        a2 = _make_agent(alias="gmn-pro", stdout_log="output from gemini")
        prompt = build_synthesis_prompt([a1, a2])
        assert "output from codex" in prompt
        assert "output from gemini" in prompt
        assert "cdx-deep" in prompt
        assert "gmn-pro" in prompt

    def test_contains_failed_agent(self):
        a1 = _make_agent(alias="cdx-deep", stdout_log="partial", status=AgentStatus.FAILED, error="timeout")
        prompt = build_synthesis_prompt([a1])
        assert "timeout" in prompt
        assert "FAILED" in prompt

    def test_prompt_structure_sections(self):
        a1 = _make_agent(alias="a1", stdout_log="test")
        prompt = build_synthesis_prompt([a1])
        assert "## Answer" in prompt
        assert "## Key Signals" in prompt
        assert "## Dissent" in prompt
        assert "## Failure Report" in prompt
        assert "## Provenance" in prompt

    def test_truncates_long_output(self):
        long_output = "x" * 5000
        a1 = _make_agent(alias="a1", stdout_log=long_output)
        prompt = build_synthesis_prompt([a1])
        assert "...[truncated]" in prompt
        assert len(prompt) < len(long_output) + 1000

    def test_soft_failed_uses_failed_block(self):
        a1 = _make_agent(alias="a1", stdout_log="partial output", soft_failed=True, error="soft fail")
        prompt = build_synthesis_prompt([a1])
        assert "partial output" in prompt
        assert "soft fail" in prompt


# ---------------------------------------------------------------------------
# select_synthesis_alias
# ---------------------------------------------------------------------------
class TestSelectSynthesisAlias:
    def test_local_first_mode(self):
        with patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=True):
            with patch("orchestra.engine.synthesizer.config.local_first_config", return_value={"synthesis_alias": "oll-analyst"}):
                agent = _make_agent()
                assert select_synthesis_alias([agent], 100) == "oll-analyst"

    @patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=False)
    @patch("orchestra.engine.synthesizer.config.synthesis_config", return_value={})
    def test_all_failed_uses_deep(self, _mock_cfg, _mock_local):
        a1 = _make_agent(status=AgentStatus.FAILED)
        a2 = _make_agent(status=AgentStatus.FAILED)
        assert select_synthesis_alias([a1, a2], 100) == "cld-deep"

    @patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=False)
    @patch(
        "orchestra.engine.synthesizer.config.synthesis_config",
        return_value={
            "low_confidence_threshold": 0.60,
            "high_similarity_threshold": 0.70,
            "fast_confidence_threshold": 0.80,
            "fast_total_words_threshold": 150,
            "short_task_chars": 200,
            "short_task_confidence_threshold": 0.75,
            "long_task_chars": 1000,
            "long_output_words": 800,
        },
    )
    def test_low_confidence_uses_deep(self, _mock_cfg, _mock_local):
        a1 = _make_agent(stdout_log="short", confidence=0.5)
        a2 = _make_agent(stdout_log="short too", confidence=0.5)
        assert select_synthesis_alias([a1, a2], 100) == "cld-deep"

    @patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=False)
    @patch(
        "orchestra.engine.synthesizer.config.synthesis_config",
        return_value={
            "low_confidence_threshold": 0.60,
            "high_similarity_threshold": 0.70,
            "fast_confidence_threshold": 0.80,
            "fast_total_words_threshold": 150,
            "short_task_chars": 200,
            "short_task_confidence_threshold": 0.75,
            "long_task_chars": 1000,
            "long_output_words": 800,
        },
    )
    def test_high_similarity_uses_fast(self, _mock_cfg, _mock_local):
        a1 = _make_agent(stdout_log="agreed result here", confidence=0.95)
        a2 = _make_agent(stdout_log="agreed result here", confidence=0.95)
        assert select_synthesis_alias([a1, a2], 100) == "gmn-fast"

    @patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=False)
    @patch(
        "orchestra.engine.synthesizer.config.synthesis_config",
        return_value={
            "low_confidence_threshold": 0.60,
            "high_similarity_threshold": 0.70,
            "fast_confidence_threshold": 0.80,
            "fast_total_words_threshold": 150,
            "short_task_chars": 200,
            "short_task_confidence_threshold": 0.75,
            "long_task_chars": 1000,
            "long_output_words": 800,
        },
    )
    def test_long_task_uses_pro(self, _mock_cfg, _mock_local):
        # Use different + long outputs so high_similarity and fast_total_words thresholds don't trigger gmn-fast first
        out1 = " ".join([f"alpha{i}" for i in range(100)])
        out2 = " ".join([f"beta{i}" for i in range(100)])
        a1 = _make_agent(stdout_log=out1, confidence=0.85)
        a2 = _make_agent(stdout_log=out2, confidence=0.85)
        assert select_synthesis_alias([a1, a2], 2000) == "gmn-pro"

    @patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=False)
    @patch(
        "orchestra.engine.synthesizer.config.synthesis_config",
        return_value={
            "low_confidence_threshold": 0.60,
            "high_similarity_threshold": 0.70,
            "fast_confidence_threshold": 0.80,
            "fast_total_words_threshold": 150,
            "short_task_chars": 200,
            "short_task_confidence_threshold": 0.75,
            "long_task_chars": 1000,
            "long_output_words": 800,
        },
    )
    def test_failure_ratio_threshold_uses_deep(self, _mock_cfg, _mock_local):
        a1 = _make_agent(stdout_log="ok", confidence=0.95)
        a2 = _make_agent(status=AgentStatus.FAILED)
        # failure_ratio = 1/2 = 0.5 > 0.3
        assert select_synthesis_alias([a1, a2], 100) == "cld-deep"

    @patch("orchestra.engine.synthesizer.config.local_first_enabled", return_value=False)
    @patch(
        "orchestra.engine.synthesizer.config.synthesis_config",
        return_value={
            "low_confidence_threshold": 0.60,
            "high_similarity_threshold": 0.70,
            "fast_confidence_threshold": 0.80,
            "fast_total_words_threshold": 150,
            "short_task_chars": 200,
            "short_task_confidence_threshold": 0.75,
            "long_task_chars": 1000,
            "long_output_words": 800,
        },
    )
    def test_short_confident_task_uses_fast(self, _mock_cfg, _mock_local):
        a1 = _make_agent(stdout_log="hi", confidence=0.85)
        a2 = _make_agent(stdout_log="hi", confidence=0.85)
        assert select_synthesis_alias([a1, a2], 50) == "gmn-fast"


class TestConfiguredSynthesisAliases:
    """The cheap and the default synthesis model come from [synthesis] when set there."""

    def test_agreeing_outputs_use_the_configured_fast_alias(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "local_first_enabled", lambda: False)
        monkeypatch.setattr(config, "synthesis_config", lambda: {"fast_alias": "oll-flash", "default_alias": "oll-glm"})
        same = "the answer is forty two and nothing else matters here"
        a1 = _make_agent(alias="cld-deep", stdout_log=same, confidence=0.9)
        a2 = _make_agent(alias="kimi-deep", stdout_log=same, confidence=0.9)
        assert select_synthesis_alias([a1, a2], 100) == "oll-flash"

    def test_long_tasks_use_the_configured_default_alias(self, monkeypatch):
        from orchestra import config

        monkeypatch.setattr(config, "local_first_enabled", lambda: False)
        monkeypatch.setattr(config, "synthesis_config", lambda: {"fast_alias": "oll-flash", "default_alias": "oll-glm"})
        a1 = _make_agent(alias="cld-deep", stdout_log=" ".join(f"alpha{i}" for i in range(500)), confidence=0.7)
        a2 = _make_agent(alias="kimi-deep", stdout_log=" ".join(f"beta{i}" for i in range(500)), confidence=0.7)
        assert select_synthesis_alias([a1, a2], 2000) == "oll-glm"

