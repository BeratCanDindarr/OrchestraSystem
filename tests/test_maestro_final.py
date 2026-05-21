import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestra.engine.cue import CueEngine
from orchestra.engine.cue_scheduler import CueScheduler
from orchestra.engine.session_discovery import SessionDiscovery
from orchestra.engine.worktree_manager import WorktreeManager
from orchestra.engine.moderator import ModeratorAgent

def test_cue_chaining():
    engine = CueEngine(cue_file="/tmp/nonexistent.yaml")
    from orchestra.engine.cue import CueRule
    r1 = CueRule("rule1", "start", "run_agent", {"task": "Task 1"})
    # Use different event to avoid recursion
    r2 = CueRule("rule2", "other.event", "run_agent", {"task": "Task 2"})
    engine.rules = [r1, r2]
    
    with patch("orchestra.engine.cue.run_ask") as mock_run:
        engine.trigger("start", {})
        assert mock_run.call_count == 1
        assert mock_run.call_args_list[0].kwargs["prompt"] == "Task 1"

def test_session_discovery_claude(tmp_path):
    sd = SessionDiscovery(project_root=tmp_path)
    claude_dir = tmp_path / ".claude" / "session-state"
    claude_dir.mkdir(parents=True)
    (claude_dir / "session1.json").write_text('{"summary": "Test Title", "messages": [{"content": [{"text": "Hello"}]}]}', encoding="utf-8")
    
    sessions = sd.discover_all()
    assert len(sessions) == 1
    assert sessions[0].title == "Test Title"

def test_moderator_debate():
    with patch("orchestra.engine.moderator.run_ask") as mock_ask:
        mock_run = MagicMock()
        mock_run.agents = [MagicMock(stdout_log="CONSENSUS")]
        mock_ask.return_value = mock_run
        
        mod = ModeratorAgent("Topic")
        mod.run_debate(max_rounds=1)
        assert "CONSENSUS" in mod.state.get_transcript()

def test_cue_yaml_load(tmp_path):
    cue_file = tmp_path / "cue.yaml"
    cue_file.write_text("rules:\n  test:\n    'on': file.changed\n    do: run_agent\n    with:\n      task: hi", encoding="utf-8")
    engine = CueEngine(cue_file=cue_file)
    assert len(engine.rules) == 1
    assert engine.rules[0].event_type == "file.changed"

def test_worktree_create(tmp_path):
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value.returncode = 0
        wtm = WorktreeManager(repo_root=tmp_path)
        wtm.worktrees_root = tmp_path / "wts"
        path = wtm.create("test-branch")
        assert "test-branch" in str(path)
