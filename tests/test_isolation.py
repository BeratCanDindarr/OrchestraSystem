"""An agent that may write works on a copy; its changes reach the project when the owner says so."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from orchestra import config
from orchestra.engine import isolation
from orchestra.models import AgentRun, AgentStatus, OrchestraRun, RunStatus


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A git project with one committed file, and Orchestra's state inside it."""
    repo = tmp_path / "project"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "timer.py").write_text("interval = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    state = repo / ".orchestra"
    runs = state / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr("orchestra.config.project_root", lambda: repo)
    monkeypatch.setattr("orchestra.config.orchestra_root", lambda: state)
    monkeypatch.setattr("orchestra.config.artifact_root", lambda: runs)
    monkeypatch.setattr("orchestra.config.db_path", lambda: state / "orchestra.db")
    return repo


def _enable(monkeypatch, **settings):
    monkeypatch.setitem(config._cfg, "isolation", {"enabled": True, **settings})


# ── the workspace ───────────────────────────────────────────────────────────

def test_a_workspace_is_the_project_as_committed(project):
    workspace = isolation.open_workspace("run-1")

    assert (workspace / "timer.py").read_text() == "interval = 1\n"
    assert workspace != project and project not in workspace.parents[:0]
    isolation.close_workspace("run-1", workspace)


def test_what_the_agent_changed_becomes_a_patch_and_the_project_is_untouched(project):
    workspace = isolation.open_workspace("run-2")
    (workspace / "timer.py").write_text("interval = 5\n")
    (workspace / "new.py").write_text("created = True\n")

    patch_text = isolation.close_workspace("run-2", workspace)

    assert "interval = 5" in patch_text and "new.py" in patch_text
    assert isolation.pending_changes("run-2").read_text() == patch_text
    assert (project / "timer.py").read_text() == "interval = 1\n"
    assert not (project / "new.py").exists()
    assert not workspace.exists()
    assert "run-2" not in _git(project, "worktree", "list")


def test_an_agent_that_changed_nothing_leaves_no_patch(project):
    workspace = isolation.open_workspace("run-3")

    assert isolation.close_workspace("run-3", workspace) == ""
    assert isolation.pending_changes("run-3") is None


def test_applying_puts_the_changes_into_the_project(project):
    workspace = isolation.open_workspace("run-4")
    (workspace / "timer.py").write_text("interval = 5\n")
    (workspace / "new.py").write_text("created = True\n")
    isolation.close_workspace("run-4", workspace)

    isolation.apply_changes("run-4")

    assert (project / "timer.py").read_text() == "interval = 5\n"
    assert (project / "new.py").read_text() == "created = True\n"
    assert isolation.pending_changes("run-4") is None


def test_changes_that_no_longer_fit_are_refused_and_kept(project):
    workspace = isolation.open_workspace("run-5")
    (workspace / "timer.py").write_text("interval = 5\n")
    isolation.close_workspace("run-5", workspace)
    (project / "timer.py").write_text("interval = 99  # the owner edited this meanwhile\n")

    with pytest.raises(isolation.IsolationError):
        isolation.apply_changes("run-5")

    assert "99" in (project / "timer.py").read_text()
    assert isolation.pending_changes("run-5") is not None


def test_discarding_throws_the_changes_away(project):
    workspace = isolation.open_workspace("run-6")
    (workspace / "timer.py").write_text("interval = 5\n")
    isolation.close_workspace("run-6", workspace)

    isolation.discard_changes("run-6")

    assert isolation.pending_changes("run-6") is None
    assert (project / "timer.py").read_text() == "interval = 1\n"


def test_outside_a_git_project_there_is_no_workspace_and_no_silent_fallback(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    (plain / ".orchestra" / "runs").mkdir(parents=True)
    monkeypatch.setattr("orchestra.config.project_root", lambda: plain)
    monkeypatch.setattr("orchestra.config.orchestra_root", lambda: plain / ".orchestra")
    monkeypatch.setattr("orchestra.config.artifact_root", lambda: plain / ".orchestra" / "runs")

    with pytest.raises(isolation.IsolationError):
        isolation.open_workspace("run-7")


# ── who gets one ────────────────────────────────────────────────────────────

def test_only_a_writing_alias_is_isolated_and_only_when_switched_on(monkeypatch):
    monkeypatch.setitem(config._cfg, "isolation", {})
    assert isolation.applies_to("cld-write") is False

    _enable(monkeypatch)
    assert isolation.applies_to("cld-write") is True
    assert isolation.applies_to("cld-write#concise") is True
    assert isolation.applies_to("cld-deep") is False
    assert isolation.applies_to("cld-write@windows") is False       # another machine's files

    _enable(monkeypatch, aliases=["cdx-deep"])
    assert isolation.applies_to("cdx-deep") is True and isolation.applies_to("cld-write") is False


# ── a run ───────────────────────────────────────────────────────────────────

def _writing_agent(seen: dict):
    def run_parallel(run, pairs, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        if kwargs.get("cwd"):
            (Path(kwargs["cwd"]) / "timer.py").write_text("interval = 5\n")
        agent = AgentRun(alias=pairs[0][0], provider="claude", model="claude/opus", status=AgentStatus.COMPLETED)
        agent.stdout_log = "I changed the interval."
        run.agents.append(agent)
    return run_parallel


def test_a_writing_run_works_in_its_workspace_and_leaves_a_patch(project, monkeypatch):
    from orchestra.engine.runners import common

    _enable(monkeypatch)
    seen: dict = {}
    with patch.object(common, "run_parallel", _writing_agent(seen)), \
         patch.object(common, "_try_cache_hit", return_value=None), \
         patch.object(common, "_store_in_cache"):
        run = common._run_ask_impl("cld-write", "set the interval to 5")

    assert seen["cwd"] and Path(seen["cwd"]) != project
    assert run.status == RunStatus.COMPLETED
    assert (project / "timer.py").read_text() == "interval = 1\n"
    assert "interval = 5" in isolation.pending_changes(run.run_id).read_text()
    assert "orchestra apply" in run.summary


def test_a_reading_run_is_given_no_workspace(project, monkeypatch):
    from orchestra.engine.runners import common

    _enable(monkeypatch)
    seen: dict = {}
    with patch.object(common, "run_parallel", _writing_agent(seen)), \
         patch.object(common, "_try_cache_hit", return_value=None), \
         patch.object(common, "_store_in_cache"):
        common._run_ask_impl("cld-deep", "what is the interval?")

    assert seen["cwd"] is None


def test_the_working_directory_reaches_the_agents_process():
    from orchestra.engine.parallel import _run_one_agent_async

    provider = MagicMock()
    provider.model_label.return_value = "p/m"
    provider.name = "p"
    provider.last_usage.return_value = None
    run = OrchestraRun(run_id="run-cwd", mode="ask", task="t")

    with patch("orchestra.engine.parallel.resolve", return_value=(provider, "m")), \
         patch("orchestra.engine.parallel.run_with_retry_async", new=AsyncMock(return_value=("ok", 0))) as retry_loop, \
         patch("orchestra.engine.artifacts.append_event"), \
         patch("orchestra.engine.artifacts.write_agent_log"), \
         patch("orchestra.engine.artifacts.write_normalized"), \
         patch("orchestra.engine.artifacts.write_manifest"):
        asyncio.run(_run_one_agent_async(run, "cld-write", "prompt", threading.Lock(), MagicMock(), asyncio.Event(),
                                         cwd="/somewhere/else"))

    assert retry_loop.call_args.kwargs["cwd"] == "/somewhere/else"


# ── the commands ────────────────────────────────────────────────────────────

def test_apply_and_discard_commands(project):
    from orchestra import cli

    for run_id, command in (("run-8", "apply"), ("run-9", "discard")):
        workspace = isolation.open_workspace(run_id)
        (workspace / f"{run_id}.txt").write_text("x\n")
        isolation.close_workspace(run_id, workspace)
        result = CliRunner().invoke(cli.app, [command, run_id])
        assert result.exit_code == 0, result.output

    assert (project / "run-8.txt").exists() and not (project / "run-9.txt").exists()
    assert CliRunner().invoke(cli.app, ["apply", "run-8"]).exit_code == 1      # nothing left to apply
