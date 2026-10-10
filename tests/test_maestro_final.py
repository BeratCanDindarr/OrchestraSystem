import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestra.engine.cue import CueEngine
from orchestra.engine.cue_scheduler import CueScheduler
from orchestra.engine.session_discovery import SessionDiscovery
from orchestra.engine.worktree_manager import WorktreeManager

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


def test_claude_code_transcripts_are_discovered(tmp_path, monkeypatch):
    import json as _json

    home = tmp_path / "home"
    project = tmp_path / "my_game"
    project.mkdir()
    import re as _re
    folder = home / ".claude" / "projects" / _re.sub(r"[^A-Za-z0-9]", "-", str(project.resolve()))
    folder.mkdir(parents=True)
    lines = [
        {"type": "mode", "mode": "normal"},
        {"type": "user", "message": {"content": "<system-reminder>ignore</system-reminder>"}},
        {"type": "user", "message": {"content": [{"type": "text", "text": "add a hint button\nto the HUD"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
    ]
    (folder / "abc123.jsonl").write_text("\n".join(_json.dumps(line) for line in lines), encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)

    sd = SessionDiscovery(project_root=project)
    sessions = sd.discover_all()

    assert [(s.id, s.provider, s.title) for s in sessions] == [("abc123", "claude-code", "add a hint button to the HUD")]
    assert [s.id for s in sd.search("hint button")] == ["abc123"]
    assert sd.search("no such text") == []


def test_replay_synthesises_a_finished_run_again(monkeypatch):
    from orchestra import service
    from orchestra.engine import artifacts
    from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus

    run = OrchestraRun(mode="dual", task="Compare tweens and animators.")
    for alias, text in (("cld-deep", "Tweens are cheap. " * 20), ("kimi-deep", "Animators blend. " * 20)):
        agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = text
        run.agents.append(agent)
        artifacts.write_agent_log(run.run_id, alias, text)
    run.status = RunStatus.COMPLETED
    artifacts.write_manifest(run)
    prompts = []

    def fake_ask(alias, prompt, **kwargs):
        prompts.append((alias, prompt))
        out = OrchestraRun(mode="ask", task=prompt)
        agent = AgentRun(alias=alias, provider="test", model="test/model", status=AgentStatus.COMPLETED)
        agent.stdout_log = "Use tweens for UI."
        out.agents.append(agent)
        return out

    monkeypatch.setattr("orchestra.engine.runner.run_ask", fake_ask)

    result = service.replay_run_output(run.run_id, synthesizer="oll-flash")

    assert result["synthesizer"] == "oll-flash"
    assert prompts[0][0] == "oll-flash" and "Compare tweens and animators." in prompts[0][1]
