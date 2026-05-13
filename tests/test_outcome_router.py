"""Unit tests for outcome-weighted routing (Phase 1 + Phase 2)."""
import pytest
from orchestra.router.outcome_router import OutcomeRouter
from orchestra.utils.outcome_recorder import OutcomeRecorder


class TestTokenization:
    def test_tokenize_basic(self):
        tokens = OutcomeRecorder.tokenize("Fix the routing bug in production")
        assert "fix" in tokens
        assert "routing" in tokens
        assert "bug" in tokens
        assert "production" in tokens
        # Stop words filtered out
        assert "the" not in tokens
        assert "in" not in tokens

    def test_tokenize_empty(self):
        tokens = OutcomeRecorder.tokenize("")
        assert tokens == set()

    def test_tokenize_nonstring(self):
        with pytest.raises(ValueError):
            OutcomeRecorder.tokenize(123)


class TestJaccardSimilarity:
    def test_identical_sets(self):
        sim = OutcomeRouter.jaccard_similarity({"fix", "bug"}, {"fix", "bug"})
        assert sim == 1.0

    def test_disjoint_sets(self):
        sim = OutcomeRouter.jaccard_similarity({"fix"}, {"routing"})
        assert sim == 0.0

    def test_partial_overlap(self):
        sim = OutcomeRouter.jaccard_similarity({"fix", "bug"}, {"fix", "routing"})
        assert 0.25 < sim < 0.75

    def test_none_inputs(self):
        with pytest.raises(ValueError):
            OutcomeRouter.jaccard_similarity(None, {"fix"})


class TestFindTopSimilar:
    def test_find_top_k(self):
        history = [
            {"task_tokens": ["fix", "bug"], "mode": "ask", "confidence": 0.5, "cost_usd": 0.1},
            {"task_tokens": ["fix", "bug", "routing"], "mode": "dual", "confidence": 0.9, "cost_usd": 0.2},
            {"task_tokens": ["implement", "feature"], "mode": "planned", "confidence": 0.8, "cost_usd": 0.3},
        ]
        similar = OutcomeRouter.find_top_similar({"fix", "bug"}, history, k=2)
        assert len(similar) <= 2
        assert similar[0]["mode"] in ("ask", "dual")

    def test_invalid_tokens_type(self):
        with pytest.raises(ValueError):
            OutcomeRouter.find_top_similar("not_a_set", [], k=10)


class TestComputeEV:
    def test_ev_computation(self):
        similar = [
            {"mode": "ask", "confidence": 0.5, "cost_usd": 0.1},
            {"mode": "ask", "confidence": 0.7, "cost_usd": 0.1},
            {"mode": "dual", "confidence": 0.9, "cost_usd": 0.2},
        ]
        ev = OutcomeRouter.compute_ev(similar)
        assert ev["ask"] > 0
        assert ev["dual"] > 0
        # ask: (0.5+0.7)/2 / 0.1 = 60; dual: 0.9 / 0.2 = 4.5
        assert ev["ask"] > ev["dual"]

    def test_zero_cost_epsilon(self):
        similar = [{"mode": "ask", "confidence": 1.0, "cost_usd": 0.0}]
        ev = OutcomeRouter.compute_ev(similar)
        assert ev["ask"] > 0  # Should not raise ZeroDivisionError due to EPSILON


class TestSuggestMode:
    def test_cold_start_fallback(self):
        history = [{"task_tokens": ["test"], "mode": "ask", "confidence": 0.5, "cost_usd": 0.1}]
        mode = OutcomeRouter.suggest_mode("Fix critical production bug", history, min_history=50)
        # Cold start → keyword fallback
        assert mode == "critical"

    def test_keyword_fallback_planned(self):
        history = []
        mode = OutcomeRouter.suggest_mode("Design the routing architecture", history, min_history=1)
        assert mode == "planned"

    def test_keyword_fallback_dual(self):
        history = []
        mode = OutcomeRouter.suggest_mode("Compare two approaches", history, min_history=1)
        assert mode == "dual"

    def test_keyword_fallback_ask(self):
        history = []
        mode = OutcomeRouter.suggest_mode("Random task", history, min_history=1)
        assert mode == "ask"

    def test_invalid_task_description(self):
        with pytest.raises(ValueError):
            OutcomeRouter.suggest_mode("", [], min_history=50)

    def test_invalid_history_type(self):
        with pytest.raises(ValueError):
            OutcomeRouter.suggest_mode("Test task", "not_a_list", min_history=50)


class TestOutcomeRecorder:
    def test_record_outcome_validation(self):
        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("", "ask", 0.5, 0.1)

        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("task", "invalid_mode", 0.5, 0.1)

        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("task", "ask", 1.5, 0.1)  # confidence > 1.0

        with pytest.raises(ValueError):
            OutcomeRecorder.record_outcome("task", "ask", 0.5, -0.1)  # negative cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
