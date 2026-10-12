"""Unit tests for orchestra.router.classifier — deterministic, no external deps."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from orchestra.router.classifier import (
    RouteDecision,
    _heuristic_route,
    _load_router_policy,
    classify,
    classify_to_tier,
    route_task,
    select_alias_for_candidates,
)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------
_DEFAULT_KEYWORDS = {
    "cheap": ["hello", "selam"],
    "deep": ["research", "araştır", "plan", "analyze"],
}


class TestClassify:
    def test_hello_is_cheap(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("hello there") == "cheap"

    def test_research_is_deep(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("research quantum computing") == "deep"

    def test_plan_is_deep(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("plan the architecture") == "deep"

    def test_default_is_balanced(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("do something interesting") == "balanced"

    def test_empty_task_is_balanced(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("") == "balanced"

    def test_long_task_is_extreme(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        long_task = "x" * 1501
        assert classify(long_task) == "extreme"

    def test_many_lines_is_extreme(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        multiline = "\n" * 21
        assert classify(multiline) == "extreme"

    def test_case_insensitive_cheap(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("HELLO WORLD") == "cheap"

    def test_case_insensitive_deep(self, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify("ANALYZE the code") == "deep"


# ---------------------------------------------------------------------------
# route_task (heuristic routing)
# ---------------------------------------------------------------------------
class TestRouteTask:
    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_review_routes_to_dual(self, _mock_cfg, _mock_model):
        decision = route_task("please review this code")
        assert decision.mode == "dual"
        assert decision.reason == "comparison_or_review_request"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_a_request_to_change_code_gets_two_opinions(self, _mock_cfg, _mock_model):
        # Agents advise, Claude Code edits: there is no mode that edits behind an approval gate.
        decision = route_task("fix the bug in login")
        assert decision.mode == "dual"
        assert decision.require_approval is False
        assert decision.reason == "workspace_mutation_request"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_what_is_routes_to_ask(self, _mock_cfg, _mock_model):
        decision = route_task("what is polymorphism")
        assert decision.mode == "ask"
        assert decision.reason == "lookup_or_explanation_request"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_long_task_routes_to_planned(self, _mock_cfg, _mock_model):
        long_task = "x" * 1501
        decision = route_task(long_task)
        assert decision.mode == "planned"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_intent_hint_review(self, _mock_cfg, _mock_model):
        decision = route_task("something", intent_hint="review")
        assert decision.mode == "dual"
        assert decision.reason == "intent_hint_review"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_intent_hint_plan(self, _mock_cfg, _mock_model):
        decision = route_task("something", intent_hint="plan")
        assert decision.mode == "planned"
        assert decision.reason == "intent_hint_plan"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_preferred_alias_passed_through(self, _mock_cfg, _mock_model):
        decision = route_task("what is this", preferred_alias="gmn-fast")
        assert decision.mode == "ask"
        assert decision.alias == "gmn-fast"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_default_routes_to_ask(self, _mock_cfg, _mock_model):
        decision = route_task("random nonsense here")
        assert decision.mode == "ask"
        assert decision.classifier == "heuristic"

    @patch("orchestra.router.classifier._model_route", return_value=None)
    @patch("orchestra.router.classifier.config.routing_config", return_value={})
    def test_implement_routes_to_dual(self, _mock_cfg, _mock_model):
        decision = route_task("implement new feature")
        assert decision.mode == "dual"
        assert decision.require_approval is False


# ---------------------------------------------------------------------------
# select_alias_for_candidates
# ---------------------------------------------------------------------------
class TestSelectAliasForCandidates:
    def test_empty_aliases_returns_the_reviewer_persona(self):
        with patch("orchestra.router.classifier.config.personas_config", return_value={"reviewer": "kimi-code"}):
            assert select_alias_for_candidates("task", []) == "kimi-code"

    def test_single_alias_returns_it(self):
        assert select_alias_for_candidates("task", ["gmn-pro"]) == "gmn-pro"

    @patch(
        "orchestra.router.classifier.suggest_best_alias",
        return_value="gmn-fast",
    )
    def test_semantic_best_used(self, _mock_semantic):
        assert select_alias_for_candidates("task", ["cdx-fast", "gmn-fast"]) == "gmn-fast"

    @patch(
        "orchestra.router.classifier.suggest_best_alias",
        side_effect=Exception("model down"),
    )
    @patch(
        "orchestra.router.classifier.get_db",
        side_effect=Exception("db down"),
    )
    def test_falls_back_to_first_alias(self, _mock_db, _mock_semantic):
        assert select_alias_for_candidates("task", ["cld-deep", "gmn-pro"]) == "cld-deep"

    @patch(
        "orchestra.router.classifier.suggest_best_alias",
        return_value=None,
    )
    @patch(
        "orchestra.router.classifier.choose_alias_by_reputation",
        return_value="gmn-pro",
    )
    @patch("orchestra.router.classifier.get_db")
    def test_reputation_fallback(self, _mock_db, _mock_rep, _mock_semantic):
        assert select_alias_for_candidates("task", ["cdx-fast", "gmn-pro"]) == "gmn-pro"

    @patch(
        "orchestra.router.classifier.suggest_best_alias",
        return_value=None,
    )
    @patch(
        "orchestra.router.classifier.choose_alias_by_reputation",
        return_value=None,
    )
    @patch("orchestra.router.classifier.get_db")
    def test_reputation_none_falls_to_first(self, _mock_db, _mock_rep, _mock_semantic):
        assert select_alias_for_candidates("task", ["cdx-fast", "gmn-pro"]) == "cdx-fast"


# ---------------------------------------------------------------------------
# classify_to_tier
# ---------------------------------------------------------------------------
class TestClassifyToTier:
    @patch("orchestra.router.classifier.config.tier_routing_config", return_value={})
    def test_cheap_to_light(self, _mock_cfg, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify_to_tier("hello") == "light"

    @patch("orchestra.router.classifier.config.tier_routing_config", return_value={})
    def test_balanced_to_medium(self, _mock_cfg, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify_to_tier("do something") == "medium"

    @patch("orchestra.router.classifier.config.tier_routing_config", return_value={})
    def test_deep_to_heavy(self, _mock_cfg, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify_to_tier("research") == "heavy"

    @patch("orchestra.router.classifier.config.tier_routing_config", return_value={})
    def test_extreme_to_heavy(self, _mock_cfg, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify_to_tier("x" * 1501) == "heavy"

    @patch(
        "orchestra.router.classifier.config.tier_routing_config",
        return_value={"cheap": "custom_light"},
    )
    def test_custom_tier_routing(self, _mock_cfg, monkeypatch):
        monkeypatch.setattr("orchestra.router.classifier._ROUTER_POLICY", {"keywords": dict(_DEFAULT_KEYWORDS)})
        assert classify_to_tier("hello") == "custom_light"


class TestKeywordsMatchWords:
    def _route(self, task):
        with patch("orchestra.router.classifier.select_alias_for_candidates", return_value="cld-fast"):
            return _heuristic_route(task)

    def test_a_keyword_inside_another_word_does_not_count(self):
        assert self._route("explain the prefix of this path").mode == "ask"

    def test_a_keyword_at_the_start_of_a_word_counts(self):
        assert self._route("fixing the timer label").reason == "workspace_mutation_request"
        assert self._route("timer etiketini düzeltir misin").reason == "workspace_mutation_request"


# ── a term matches a word, not the letters a longer word starts with ────────

class TestWholeWords:
    @pytest.mark.parametrize("task", [
        "what is a const in javascript?",
        "what does the console print here?",
        "explain the editor layout code",
        "where is the movement speed defined?",
        "show me the changelog",
        "which scenery assets are loaded first?",
    ])
    def test_a_plain_question_is_asked_not_debated(self, task):
        from orchestra.router.classifier import _heuristic_route

        with patch("orchestra.router.classifier.select_alias_for_candidates", return_value="cdx-fast"):
            assert _heuristic_route(task).mode == "ask"

    @pytest.mark.parametrize("task", [
        "fix the timer",                       # the term itself
        "this fixes nothing, refactoring needed",   # English inflections
        "pros and cons of pooling",
        "şunu düzeltir misin",                 # a Turkish stem takes suffixes
        "iki yaklaşımı karşılaştırır mısın",
    ])
    def test_a_request_to_change_or_compare_still_gets_two_agents(self, task):
        from orchestra.router.classifier import _heuristic_route

        assert _heuristic_route(task).mode == "dual"

    def test_classify_reads_words_not_substrings(self):
        from orchestra.router.classifier import classify

        assert classify("list the redesigned icons") == "balanced"     # "design" is inside "redesigned"
        assert classify("why is the build slow") == "deep"
