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


def test_try_cache_hit():
    with patch("orchestra.engine.runners.common._get_fts_cache") as mock_cache_getter:
        mock_cache = MagicMock()
        mock_cache_getter.return_value = mock_cache

        # Nothing stored for this prompt and alias
        mock_cache.lookup.return_value = None
        assert _try_cache_hit("p", "r1", alias="oll-gemma") is None

        # The same prompt from the same alias is stored
        hit = MagicMock()
        hit.cost_saved_usd = 0.01
        mock_cache.lookup.return_value = hit
        assert _try_cache_hit("p", "r1", alias="oll-gemma") is hit
        assert mock_cache.lookup.call_args.args[0] == "p"
        assert mock_cache.lookup.call_args.args[1].startswith("oll-gemma@")   # scoped to the project state
        mock_cache.search.assert_not_called()

def test_store_in_cache():
    with patch("orchestra.engine.runners.common._get_fts_cache") as mock_cache_getter:
        mock_cache = MagicMock()
        mock_cache_getter.return_value = mock_cache
        _store_in_cache("p", "r", "run-1", 0.01, alias="oll-gemma")
        mock_cache.store.assert_called_once()
        assert mock_cache.store.call_args.kwargs["alias"].startswith("oll-gemma@")

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


def _git_project(path, text):
    import subprocess
    path.mkdir()
    (path / "a.py").write_text(text)
    for cmd in (["git", "init", "-q"], ["git", "add", "."],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=path, check=True)
    return path


def test_a_stored_answer_holds_only_for_the_project_state_it_was_given_in(tmp_path):
    from orchestra.engine.runners import common
    from orchestra.storage.fts_cache import FtsSearchCache

    one, two = _git_project(tmp_path / "one", "x = 1\n"), _git_project(tmp_path / "two", "x = 1\n")
    cache = FtsSearchCache(db_path=str(tmp_path / "cache.db"))
    with patch.object(common, "_get_fts_cache", return_value=cache), \
         patch.object(common.config, "project_root", return_value=one):
        common._store_in_cache("what is x?", "x is 1", "run-1", alias="cld-fast")
        assert common._try_cache_hit("what is x?", "r", alias="cld-fast") is not None
        (one / "a.py").write_text("x = 2\n")   # an edit nobody committed
        assert common._try_cache_hit("what is x?", "r", alias="cld-fast") is None
    with patch.object(common, "_get_fts_cache", return_value=cache), \
         patch.object(common.config, "project_root", return_value=two):
        assert common._try_cache_hit("what is x?", "r", alias="cld-fast") is None


def test_a_cached_answer_keeps_the_confidence_its_agent_stated():
    from orchestra.engine.runners.common import _cached_confidence

    assert _cached_confidence("x is 1\n\nConfidence: medium") == 0.60
    assert _cached_confidence("x is 1") == 0.5   # none stated: the neutral default, not 1.0


def test_the_replay_store_key_holds_the_project_state_too(tmp_path):
    from orchestra.engine.runners import common

    one, two = _git_project(tmp_path / "one", "x = 1\n"), _git_project(tmp_path / "two", "x = 1\n")
    with patch.object(common.config, "project_root", return_value=one):
        key_one = common._replay_key("cld-fast", "what is x?")
    with patch.object(common.config, "project_root", return_value=two):
        key_two = common._replay_key("cld-fast", "what is x?")

    assert key_one != key_two and "what is x?" in key_one
