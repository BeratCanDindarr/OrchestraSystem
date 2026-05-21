"""Extended unit tests for orchestra.engine.runners.common and monitoring."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from orchestra.engine.runners.common import (
    _load_retrieval_policy, 
    _load_agent_policy, 
    _compress_blackboard_context, 
    _estimate_avg_confidence, 
    _alias_to_model_label, 
    _try_cache_hit,
    _store_in_cache,
    _estimate_run_cost
)
from orchestra.models import OrchestraRun, AgentRun, AgentStatus

def test_load_retrieval_policy(tmp_path):
    policy_file = tmp_path / "orchestra-retrieval-policy.md"
    policy_file.write_text("My Retrieval Policy", encoding="utf-8")
    
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        res = _load_retrieval_policy()
        assert "My Retrieval Policy" in res
        assert "[HIDDEN RETRIEVAL POLICY]" in res

def test_load_agent_policy(tmp_path):
    policy_file = tmp_path / "orchestra-agent-policy.md"
    policy_file.write_text("My Agent Policy", encoding="utf-8")
    
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        res = _load_agent_policy()
        assert "My Agent Policy" in res
        assert "[HIDDEN AGENT POLICY]" in res

def test_compress_blackboard_context():
    text = "line1\nline2\nline3"
    # Small limit should trigger compression
    res = _compress_blackboard_context(text, 10)
    assert "[GLOBAL BLACKBOARD (Compressed)]" in res
    assert "line3" in res
    assert "line1" not in res

def test_estimate_avg_confidence():
    run = MagicMock()
    a1 = MagicMock()
    a1.confidence = 0.8
    a2 = MagicMock()
    a2.confidence = 0.6
    run.agents = [a1, a2]
    assert _estimate_avg_confidence(run) == 0.7
    
    run.agents = []
    assert _estimate_avg_confidence(run) == 1.0

def test_alias_to_model_label():
    assert _alias_to_model_label("cdx-deep") == "gpt-5.4/xhigh"
    assert _alias_to_model_label("gmn-pro") == "gemini/pro"
    assert _alias_to_model_label("unknown") == "gpt-5.4/medium"

def test_estimate_run_cost():
    run = OrchestraRun(mode="ask", task="t", run_id="r1")
    a1 = AgentRun(alias="cdx-deep", provider="p", model="m")
    a1.estimated_completion_tokens = 1000
    run.agents = [a1]
    
    with patch("orchestra.config.price_per_1k_tokens", return_value=0.03):
        cost = _estimate_run_cost(run)
        assert cost == 0.03

def test_try_cache_hit():
    with patch("orchestra.engine.runners.common._get_fts_cache") as mock_cache_getter:
        mock_cache = MagicMock()
        mock_cache_getter.return_value = mock_cache
        
        # No results
        mock_cache.search.return_value = []
        assert _try_cache_hit("p", "r1") is None
        
        # Hit below threshold
        hit = MagicMock()
        hit.similarity = 0.5
        mock_cache.search.return_value = [hit]
        assert _try_cache_hit("p", "r1") is None
        
        # Hit above threshold
        hit.similarity = 0.9
        hit.cost_saved_usd = 0.01
        assert _try_cache_hit("p", "r1") is hit

def test_store_in_cache():
    with patch("orchestra.engine.runners.common._get_fts_cache") as mock_cache_getter:
        mock_cache = MagicMock()
        mock_cache_getter.return_value = mock_cache
        _store_in_cache("p", "r", "run-1", 0.01)
        mock_cache.store.assert_called_once_with("run-1", "p", "r", cost_usd=0.01)

# 8. Monitoring
from orchestra.monitoring.cost_monitor import _check_budget, extract_cost_report, format_cost_report_for_broadcast

def test_check_budget():
    run = OrchestraRun(mode="ask", task="t", run_id="r1")
    a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.COMPLETED)
    a1.estimated_cost_usd = 0.1
    run.agents = [a1]
    
    with patch("orchestra.config.execution_config", return_value={"max_budget_usd": 0.05}):
        ok, reason = _check_budget(run)
        assert ok is False
        assert "Budget exceeded" in reason

def test_extract_cost_report():
    run = OrchestraRun(mode="ask", task="t", run_id="r1")
    run.total_cost_usd = 0.1
    a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.COMPLETED)
    a1.estimated_completion_tokens = 1000
    a1.estimated_cost_usd = 0.1
    run.agents = [a1]
    
    report = extract_cost_report(run)
    assert report.total_cost_usd == 0.1
    assert report.agent_costs[0].alias == "a1"

def test_format_cost_report():
    run = OrchestraRun(mode="ask", task="t", run_id="r1")
    run.total_cost_usd = 0.1
    a1 = AgentRun(alias="a1", provider="p", model="m", status=AgentStatus.COMPLETED)
    a1.estimated_completion_tokens = 1000
    a1.estimated_cost_usd = 0.1
    run.agents = [a1]
    
    report = extract_cost_report(run)
    payload = format_cost_report_for_broadcast(report)
    assert payload["type"] == "cost_report"
    assert payload["total_cost_usd"] == 0.1
