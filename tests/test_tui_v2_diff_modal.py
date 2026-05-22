"""Focused tests for the git diff modal screen."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from orchestra.tui.screens.diff_modal import DiffModalResult, DiffModalScreen


class DiffModalHarness(App[None]):
    """Minimal host app used to mount the diff modal under test."""

    def __init__(self, screen: DiffModalScreen) -> None:
        super().__init__()
        self.screen_under_test = screen
        self.result: DiffModalResult | None = None

    def compose(self) -> ComposeResult:
        yield Static("host")

    def capture_result(self, result: DiffModalResult | None) -> None:
        self.result = result


def _content_text(widget: Static) -> str:
    content = widget.content
    return content.plain if hasattr(content, "plain") else str(content)


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _init_git_repo(tmp_path: Path) -> None:
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.name", "Orchestra Tests")
    _run_git(tmp_path, "config", "user.email", "tests@example.com")


def _commit_file(repo_root: Path, relative_path: str, contents: str) -> Path:
    file_path = repo_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(contents, encoding="utf-8")
    assert _run_git(repo_root, "add", "--", relative_path).returncode == 0
    assert _run_git(repo_root, "commit", "-m", "seed").returncode == 0
    return file_path


@pytest.mark.asyncio
async def test_diff_modal_renders_and_closes_with_escape(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    file_path = _commit_file(tmp_path, "notes/example.py", "print('before')\n")
    file_path.write_text("print('after')\nprint('second')\n", encoding="utf-8")

    app = DiffModalHarness(DiffModalScreen(tmp_path, file_path))
    async with app.run_test() as pilot:
        app.push_screen(app.screen_under_test, app.capture_result)
        await pilot.pause()

        unified = _content_text(app.screen_under_test.query_one("#diff-unified-body", Static))
        assert "print('after')" in unified
        assert "-print('before')" in unified

        await pilot.press("tab")
        await pilot.pause()
        assert app.screen_under_test.query_one("#diff-side-by-side").display is True

        left = _content_text(app.screen_under_test.query_one("#diff-left-body", Static))
        right = _content_text(app.screen_under_test.query_one("#diff-right-body", Static))
        assert "print('before')" in left
        assert "print('after')" in right
        assert "print('second')" in right

        await pilot.press("escape")
        await pilot.pause()

    assert app.result == DiffModalResult("close", file_path.resolve(), "side_by_side")
    assert _run_git(tmp_path, "status", "--short").stdout.startswith(" M notes/example.py")


@pytest.mark.asyncio
async def test_diff_modal_accept_stages_file_with_y(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    file_path = _commit_file(tmp_path, "notes/stage_me.py", "alpha = 1\n")
    file_path.write_text("alpha = 2\n", encoding="utf-8")

    app = DiffModalHarness(DiffModalScreen(tmp_path, file_path))
    async with app.run_test() as pilot:
        app.push_screen(app.screen_under_test, app.capture_result)
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()

    assert app.result == DiffModalResult("accept", file_path.resolve(), "unified")
    cached_names = _run_git(tmp_path, "diff", "--cached", "--name-only").stdout.splitlines()
    assert cached_names == ["notes/stage_me.py"]


@pytest.mark.asyncio
async def test_diff_modal_reject_discards_file_with_n(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    file_path = _commit_file(tmp_path, "notes/discard_me.py", "value = 'base'\n")
    file_path.write_text("value = 'changed'\n", encoding="utf-8")

    app = DiffModalHarness(DiffModalScreen(tmp_path, file_path))
    async with app.run_test() as pilot:
        app.push_screen(app.screen_under_test, app.capture_result)
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()

    assert app.result == DiffModalResult("reject", file_path.resolve(), "unified")
    assert file_path.read_text(encoding="utf-8") == "value = 'base'\n"
    assert _run_git(tmp_path, "status", "--short").stdout.strip() == ""


@pytest.mark.asyncio
async def test_diff_modal_reject_discards_staged_new_file(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    tracked_file = _commit_file(tmp_path, "notes/base.py", "base = True\n")

    new_file = tmp_path / "notes" / "new_file.py"
    new_file.write_text("created = True\n", encoding="utf-8")
    assert _run_git(tmp_path, "add", "--", "notes/new_file.py").returncode == 0

    app = DiffModalHarness(DiffModalScreen(tmp_path, new_file))
    async with app.run_test() as pilot:
        app.push_screen(app.screen_under_test, app.capture_result)
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()

    assert tracked_file.exists()
    assert app.result == DiffModalResult("reject", new_file.resolve(), "unified")
    assert not new_file.exists()
    assert _run_git(tmp_path, "status", "--short").stdout.strip() == ""
