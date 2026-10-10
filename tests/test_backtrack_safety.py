"""Undoing a run must never cost work when there is nothing to restore."""
from __future__ import annotations

import subprocess

from orchestra.engine import backtrack


def _repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], check=True)
    (tmp_path / "tracked.txt").write_text("committed\n")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], check=True)
    (tmp_path / "tracked.txt").write_text("my uncommitted work\n")
    (tmp_path / "new_file.txt").write_text("my untracked work\n")
    return tmp_path


def test_without_a_snapshot_nothing_is_touched(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)

    assert backtrack.undo_last_run("norun000") is False
    assert (repo / "tracked.txt").read_text() == "my uncommitted work\n"
    assert (repo / "new_file.txt").read_text() == "my untracked work\n"


def test_a_snapshot_is_found_by_its_run(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)

    assert backtrack.find_snapshot("run12345") is None
    assert backtrack.create_snapshot("run12345", "initial") is True
    assert backtrack.find_snapshot("run12345") == "stash@{0}"
