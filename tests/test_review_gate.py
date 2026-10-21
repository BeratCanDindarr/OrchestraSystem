"""The review gate: `orchestra review --staged --fail-on` and the pre-commit hook that runs it."""
from __future__ import annotations

import os
import subprocess
from unittest.mock import patch

from typer.testing import CliRunner

from orchestra.cli import app

runner = CliRunner()


def _repo(tmp_path):
    for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"], ["git", "config", "user.name", "t"]):
        subprocess.run(cmd, cwd=tmp_path, check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def test_only_staged_changes_are_collected(tmp_path):
    from orchestra.engine.code_review import collect_diff

    repo = _repo(tmp_path)
    (repo / "a.py").write_text("x = 2\n")
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 3\n")   # an unstaged edit on top

    diff = collect_diff(staged=True, root=repo)

    assert "+x = 2" in diff and "x = 3" not in diff


def _review(findings, reviewed_by=("kimi-deep", "cld-deep")):
    return {"run": {"run_id": "rv1", "summary": "rendered"}, "findings": findings, "discarded": 0,
            "reviewed_by": list(reviewed_by), "not_reviewed_by": []}


def _finding(severity):
    return {"severity": severity, "file": "a.py", "line": 1, "quote": "x = 2", "problem": "p", "fix": "f",
            "found_by": ["cld-deep"]}


def test_the_gate_fails_on_a_checked_finding_at_or_above_the_level():
    with patch("orchestra.service.review_changes", return_value=_review([_finding("high")])) as review:
        result = runner.invoke(app, ["review", "--staged", "--fail-on", "high"])

    assert result.exit_code == 1 and review.call_args.kwargs["staged"] is True


def test_the_gate_lets_lesser_findings_through():
    with patch("orchestra.service.review_changes", return_value=_review([_finding("low"), _finding("medium")])):
        result = runner.invoke(app, ["review", "--staged", "--fail-on", "high"])

    assert result.exit_code == 0


def test_nothing_staged_lets_the_commit_through():
    with patch("orchestra.service.review_changes", side_effect=ValueError("Nothing to review: no changes")):
        result = runner.invoke(app, ["review", "--staged", "--fail-on", "high"])

    assert result.exit_code == 0


def test_no_reviewer_answering_does_not_block_the_commit():
    with patch("orchestra.service.review_changes", return_value=_review([], reviewed_by=())):
        result = runner.invoke(app, ["review", "--staged", "--fail-on", "high"])

    assert result.exit_code == 0 and "not reviewed" in result.output.lower()


def test_the_hook_is_installed_runs_the_gate_and_is_removed(tmp_path):
    from orchestra.utils import git_hook

    repo = _repo(tmp_path)

    path = git_hook.install(repo, fail_on="high")

    text = path.read_text()
    assert os.access(path, os.X_OK) and "review --staged --fail-on high" in text and git_hook.MARKER in text
    assert git_hook.uninstall(repo) is True and not path.exists()


def test_a_hook_someone_else_wrote_is_left_alone(tmp_path):
    import pytest

    from orchestra.utils import git_hook

    repo = _repo(tmp_path)
    theirs = repo / ".git" / "hooks" / "pre-commit"
    theirs.write_text("#!/bin/sh\nmake lint\n")

    with pytest.raises(FileExistsError):
        git_hook.install(repo, fail_on="high")
    assert git_hook.uninstall(repo) is False and theirs.read_text() == "#!/bin/sh\nmake lint\n"


def test_findings_are_posted_to_a_pull_request_only_when_asked():
    calls = []
    with patch("orchestra.service.review_changes", return_value=_review([_finding("high")])), \
         patch("orchestra.cli.subprocess.run", side_effect=lambda cmd, **kw: calls.append(cmd) or
               subprocess.CompletedProcess(cmd, 0, "", "")):
        runner.invoke(app, ["review", "--base", "main"])
        assert calls == []
        runner.invoke(app, ["review", "--base", "main", "--comment-pr", "42"])

    assert calls and calls[0][:4] == ["gh", "pr", "comment", "42"]
