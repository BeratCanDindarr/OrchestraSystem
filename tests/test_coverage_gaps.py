"""Tests to cover remaining 0% coverage files."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone

# 1. Observability
from orchestra.engine.observability import (
    format_duration, format_cost, parse_event_time, 
    resolve_run_id, load_manifest, load_events, 
    load_agent_logs, enrich_timeline_events, 
    compare_field_values, agent_from_manifest_dict, load_cached_agents
)
from orchestra.models import OrchestraRun, AgentRun, AgentStatus, RunStatus

def test_observability_formatting():
    assert format_duration(0.0005) == "500µs"
    assert format_duration(0.5) == "500ms"
    assert format_duration(1.5) == "1.50s"
    assert format_duration(65) == "1m5s"
    assert format_duration(3700) == "1h1m"
    
    assert format_cost(0.0001) == "$0.0001"
    assert format_cost(1.5) == "$1.50"

def test_parse_event_time():
    ts = "2026-05-21T10:00:00Z"
    epoch = parse_event_time({"ts": ts})
    assert isinstance(epoch, float)
    with pytest.raises(ValueError):
        parse_event_time({})
    with pytest.raises(ValueError):
        parse_event_time({"ts": "invalid"})

def test_resolve_run_id(tmp_path):
    with patch("orchestra.config.artifact_root", return_value=tmp_path):
        (tmp_path / "run_12345678").mkdir()
        assert resolve_run_id("run_1234") == "run_12345678"
        assert resolve_run_id("missing") is None

def test_load_manifest():
    with patch("orchestra.engine.artifacts.load_manifest", return_value={"mode": "ask", "run_id": "r1", "task": "t"}):
        run = load_manifest("r1")
        assert run.run_id == "r1"

def test_load_events():
    mock_events = [{"ts": "2026-05-21T10:00:00Z", "event": "start"}]
    with patch("orchestra.engine.artifacts.read_events", return_value=mock_events):
        events = load_events("r1")
        assert len(events) == 1
        assert "ts_epoch" in events[0]

def test_load_agent_logs(tmp_path):
    run_dir = tmp_path / "r1"
    agents_dir = run_dir / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "agent1.stdout.log").write_text("logs", encoding="utf-8")
    
    with patch("orchestra.config.artifact_root", return_value=tmp_path):
        logs = load_agent_logs("r1")
        assert logs["agent1"] == "logs"

def test_enrich_timeline_events():
    events = [{"event": "run_start"}, {"event": "agent_start", "alias": "a1"}]
    enriched = enrich_timeline_events(events)
    assert enriched[0]["event_type"] == "run"
    assert enriched[1]["event_type"] == "agent"
    assert enriched[1]["hierarchy"] == (None, None, "a1")

def test_compare_field_values():
    assert compare_field_values(10, 10) == "✓"
    assert compare_field_values(10, 20) == "different"
    assert compare_field_values("cost $10", "cost $20") == "A cheaper"

def test_agent_from_manifest_dict():
    d = {"alias": "a1", "status": "completed"}
    res = agent_from_manifest_dict(d)
    assert res["alias"] == "a1"
    assert res["status"] == "completed"

# 2. Backtrack
from orchestra.engine.backtrack import create_snapshot, restore_snapshot, undo_last_run

def test_backtrack_create():
    with patch("orchestra.engine.backtrack._run_git") as mock_git:
        mock_git.side_effect = [(0, ""), (0, "M file.txt"), (0, "")]
        assert create_snapshot("r1", "label") is True

def test_backtrack_restore():
    with patch("orchestra.engine.backtrack._run_git") as mock_git:
        mock_git.side_effect = [(0, "stash@{0}: orchestra_r1_label"), (0, ""), (0, ""), (0, "")]
        assert restore_snapshot("r1", "label") is True

# 3. Patterns
from orchestra.engine.patterns import execute_pattern

def test_patterns():
    with patch("orchestra.engine.patterns.run_ask_pattern") as mock_ask:
        execute_pattern("ask", "task")
        mock_ask.assert_called_once()

# 4. Planner
from orchestra.engine.planner import create_graph_plan

def test_planner_success():
    mock_run = MagicMock()
    mock_run.task = "task"
    mock_run.run_id = "r1"
    mock_agent_run = MagicMock()
    mock_agent_run.status = MagicMock(value="completed")
    from orchestra.models import AgentStatus
    mock_agent_run.status = AgentStatus.COMPLETED
    mock_agent_run.stdout_log = '```json\n{"nodes": [], "edges": []}\n```'
    
    with patch("orchestra.engine.planner.run_parallel", return_value=[mock_agent_run]), \
         patch("orchestra.engine.planner.build_graph_from_plan") as mock_build:
        create_graph_plan(mock_run)
        mock_build.assert_called_once()

# 5. Toolmaker
from orchestra.engine.toolmaker import ToolMaker

def test_toolmaker(tmp_path):
    tm = ToolMaker(tools_dir=str(tmp_path))
    path = tm.propose_new_tool("test", "print('hi')", "desc")
    assert Path(path).exists()
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        ok, msg = tm.test_tool_syntax(path)
        assert ok is True
        
    ok, msg = tm.promote_to_live(path)
    assert ok is True
    assert (tmp_path / "test.py").exists()

# 6. Continuous Learning
from orchestra.adapters.continuous_learning import is_available, suggest_mode_adjustment

def test_continuous_learning_not_available():
    with patch("orchestra.adapters.continuous_learning.CONFIG_PATH") as mock_path:
        mock_path.exists.return_value = False
        assert is_available() is False
        assert suggest_mode_adjustment("task", "ask") is None

# 7. Integration Hooks
from orchestra.router.integration_hooks import resolve_mode, record_outcome

def test_integration_hooks():
    with patch("orchestra.utils.outcome_recorder.OutcomeRecorder.load_history", return_value=[]), \
         patch("orchestra.router.outcome_router.OutcomeRouter.suggest_mode", return_value="dual"):
        assert resolve_mode("task") == "dual"
    
    with patch("orchestra.utils.outcome_recorder.OutcomeRecorder.record_outcome") as mock_record:
        record_outcome("task", "dual", 0.9, 0.05)
        mock_record.assert_called_once()
