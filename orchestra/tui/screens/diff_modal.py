"""Modal screen for reviewing and applying a single-file git diff."""
from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import shutil
import subprocess
from typing import Literal

from rich.text import Text

try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.screen import ModalScreen
    from textual.widgets import Button, Static

    TEXTUAL_AVAILABLE = True
except ImportError:
    ComposeResult = object  # type: ignore[assignment, misc]
    Binding = object  # type: ignore[assignment, misc]
    Horizontal = Vertical = VerticalScroll = object  # type: ignore[assignment, misc]
    ModalScreen = object  # type: ignore[assignment, misc]
    Button = Static = object  # type: ignore[assignment, misc]
    TEXTUAL_AVAILABLE = False


DiffViewMode = Literal["unified", "side_by_side"]
DiffDecision = Literal["accept", "reject", "close"]
DiffComparison = Literal["clean", "staged", "untracked", "worktree"]


@dataclass(frozen=True)
class DiffModalResult:
    """Decision returned when the modal closes."""

    decision: DiffDecision
    file_path: Path
    view_mode: DiffViewMode


@dataclass(frozen=True)
class DiffSnapshot:
    """Resolved diff payload for a single file."""

    repo_root: Path
    file_path: Path
    relative_path: str
    status_code: str
    comparison: DiffComparison
    unified_diff: str
    left_text: str
    right_text: str
    error: str | None = None


_DIFF_COMPARISON_LABELS: dict[DiffComparison, tuple[str, str, str]] = {
    "clean": ("Current", "Current", "No pending diff"),
    "staged": ("HEAD", "Index", "Staged diff"),
    "untracked": ("Empty", "Working Tree", "Untracked file"),
    "worktree": ("Index", "Working Tree", "Working tree diff"),
}

_STATUS_LEGEND = {
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "U": "unmerged",
}

_UNIFIED_HEADER_STYLES: tuple[tuple[str, str], ...] = (
    ("diff --git ", "bold #cccccc"),
    ("index ", "dim"),
    ("--- ", "bold #f44747"),
    ("+++ ", "bold #4ec9b0"),
    ("@@", "bold #007acc"),
)

_BUTTON_ACTIVE_CLASS = "-active"

if TEXTUAL_AVAILABLE:
    _DIFF_BINDINGS = [
        Binding("y", "accept", "Stage"),
        Binding("n", "reject", "Discard"),
        Binding("tab", "toggle_view", "Toggle View"),
        Binding("escape", "close_modal", "Close"),
    ]
else:
    _DIFF_BINDINGS = []


class DiffModal(ModalScreen):
    """Compact Maestro-style modal for a single-file git diff review."""

    BINDINGS = _DIFF_BINDINGS

    def __init__(
        self,
        repo_root: Path | str,
        file_path: Path | str,
        *,
        view_mode: DiffViewMode = "unified",
    ) -> None:
        super().__init__()
        self._repo_root = Path(repo_root).resolve()
        self._file_path = _resolve_file_path(self._repo_root, file_path)
        self._view_mode: DiffViewMode = "side_by_side" if view_mode == "side_by_side" else "unified"
        self._snapshot = _load_diff_snapshot(self._repo_root, self._file_path)
        self._status_text = "y stage  n discard  tab toggle view  esc close"

    def compose(self) -> ComposeResult:
        left_title, right_title, comparison_title = _DIFF_COMPARISON_LABELS[self._snapshot.comparison]
        with Vertical(id="diff-container"):
            with Horizontal(id="diff-header"):
                yield Static(f"{self._snapshot.relative_path}  {comparison_title}", id="diff-title")
                yield Static("Git Diff Review", id="diff-badge")
            with Horizontal(id="diff-toolbar"):
                yield Button("Unified", id="diff-mode-unified", classes="mode-button")
                yield Button("Side By Side", id="diff-mode-side-by-side", classes="mode-button")
                yield Static(_format_status_chip(self._snapshot), id="diff-status-chip")
            yield Static(_format_summary(self._snapshot), id="diff-summary")
            with Vertical(id="diff-body-shell"):
                with VerticalScroll(id="diff-unified-view"):
                    yield Static(_render_unified_diff(self._snapshot), id="diff-unified-body", classes="diff-body")
                with Horizontal(id="diff-side-by-side"):
                    with Vertical(classes="diff-pane"):
                        yield Static(left_title, classes="diff-pane-title")
                        with VerticalScroll(classes="diff-scroll"):
                            yield Static(_render_side_column(self._snapshot.left_text, self._snapshot.right_text, side="left"), id="diff-left-body", classes="diff-body")
                    with Vertical(classes="diff-pane"):
                        yield Static(right_title, classes="diff-pane-title")
                        with VerticalScroll(classes="diff-scroll"):
                            yield Static(_render_side_column(self._snapshot.left_text, self._snapshot.right_text, side="right"), id="diff-right-body", classes="diff-body")
            yield Static(self._status_text, id="diff-footer")

    def on_mount(self) -> None:
        self._sync_view_mode()

    def on_button_pressed(self, event) -> None:
        if event.button.id == "diff-mode-unified":
            self._set_view_mode("unified")
            return
        if event.button.id == "diff-mode-side-by-side":
            self._set_view_mode("side_by_side")

    def action_toggle_view(self) -> None:
        next_mode: DiffViewMode = "side_by_side" if self._view_mode == "unified" else "unified"
        self._set_view_mode(next_mode)

    def action_close_modal(self) -> None:
        self.dismiss(DiffModalResult("close", self._file_path, self._view_mode))

    def action_accept(self) -> None:
        if self._snapshot.error:
            self._set_status(self._snapshot.error)
            return
        result = _run_git(self._repo_root, "add", "--", self._snapshot.relative_path)
        if result.returncode != 0:
            self._set_status(_git_failure_message("stage", result))
            return
        self.dismiss(DiffModalResult("accept", self._file_path, self._view_mode))

    def action_reject(self) -> None:
        if self._snapshot.error:
            self._set_status(self._snapshot.error)
            return

        if self._snapshot.status_code == "??":
            result = _run_git(self._repo_root, "clean", "-f", "--", self._snapshot.relative_path)
        elif not _path_exists_in_head(self._repo_root, self._snapshot.relative_path):
            result = _discard_new_path(self._repo_root, self._file_path, self._snapshot.relative_path)
        else:
            result = _run_git(
                self._repo_root,
                "restore",
                "--source=HEAD",
                "--staged",
                "--worktree",
                "--",
                self._snapshot.relative_path,
            )
        if result.returncode != 0:
            self._set_status(_git_failure_message("discard", result))
            return
        self.dismiss(DiffModalResult("reject", self._file_path, self._view_mode))

    def _set_status(self, message: str) -> None:
        self._status_text = message
        if self.is_mounted:
            self.query_one("#diff-footer", Static).update(message)

    def _set_view_mode(self, view_mode: DiffViewMode) -> None:
        if view_mode == self._view_mode:
            return
        self._view_mode = view_mode
        if self.is_mounted:
            self._sync_view_mode()

    def _sync_view_mode(self) -> None:
        self.query_one("#diff-unified-view").display = self._view_mode == "unified"
        self.query_one("#diff-side-by-side").display = self._view_mode == "side_by_side"

        unified_button = self.query_one("#diff-mode-unified", Button)
        side_button = self.query_one("#diff-mode-side-by-side", Button)
        _apply_button_state(unified_button, self._view_mode == "unified")
        _apply_button_state(side_button, self._view_mode == "side_by_side")


def _resolve_file_path(repo_root: Path, file_path: Path | str) -> Path:
    candidate = Path(file_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def _load_diff_snapshot(repo_root: Path, file_path: Path) -> DiffSnapshot:
    try:
        relative_path = file_path.relative_to(repo_root).as_posix()
    except ValueError:
        return DiffSnapshot(
            repo_root=repo_root,
            file_path=file_path,
            relative_path=file_path.name,
            status_code="",
            comparison="clean",
            unified_diff="",
            left_text="",
            right_text="",
            error=f"{file_path} is outside the repository root.",
        )

    status_code = _git_status(repo_root, relative_path)
    has_worktree_diff = status_code == "??" or (len(status_code) == 2 and status_code[1] != " ")
    has_staged_diff = status_code not in {"", "??"} and len(status_code) == 2 and status_code[0] != " "

    if status_code == "??":
        comparison: DiffComparison = "untracked"
        left_text = ""
        right_text = _read_worktree_text(file_path)
    elif has_worktree_diff:
        comparison = "worktree"
        left_text = _read_git_blob(repo_root, f":{relative_path}")
        right_text = _read_worktree_text(file_path)
    elif has_staged_diff:
        comparison = "staged"
        left_text = _read_git_blob(repo_root, f"HEAD:{relative_path}")
        right_text = _read_git_blob(repo_root, f":{relative_path}")
    else:
        comparison = "clean"
        current_text = _read_worktree_text(file_path)
        left_text = current_text
        right_text = current_text

    unified_diff = _load_unified_diff(
        repo_root=repo_root,
        file_path=file_path,
        relative_path=relative_path,
        comparison=comparison,
        left_text=left_text,
        right_text=right_text,
    )

    return DiffSnapshot(
        repo_root=repo_root,
        file_path=file_path,
        relative_path=relative_path,
        status_code=status_code,
        comparison=comparison,
        unified_diff=unified_diff,
        left_text=left_text,
        right_text=right_text,
    )


def _load_unified_diff(
    *,
    repo_root: Path,
    file_path: Path,
    relative_path: str,
    comparison: DiffComparison,
    left_text: str,
    right_text: str,
) -> str:
    if comparison == "untracked":
        result = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--no-color", "--no-index", "--", "/dev/null", str(file_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.stdout.strip():
            return result.stdout.rstrip()
        return _build_fallback_unified_diff(relative_path, left_text, right_text)

    if comparison == "worktree":
        result = _run_git(repo_root, "diff", "--no-color", "--unified=3", "--", relative_path)
    elif comparison == "staged":
        result = _run_git(repo_root, "diff", "--cached", "--no-color", "--unified=3", "--", relative_path)
    else:
        result = _run_git(repo_root, "diff", "--no-color", "--unified=3", "--", relative_path)

    if result.stdout.strip():
        return result.stdout.rstrip()
    if comparison == "clean":
        return "No changes for this file."
    return _build_fallback_unified_diff(relative_path, left_text, right_text)


def _build_fallback_unified_diff(relative_path: str, left_text: str, right_text: str) -> str:
    diff_lines = difflib.unified_diff(
        left_text.splitlines(),
        right_text.splitlines(),
        fromfile=f"a/{relative_path}",
        tofile=f"b/{relative_path}",
        lineterm="",
    )
    return "\n".join(diff_lines) or "No changes for this file."


def _git_status(repo_root: Path, relative_path: str) -> str:
    result = _run_git(repo_root, "status", "--porcelain=v1", "--", relative_path)
    line = next((entry for entry in result.stdout.splitlines() if entry.strip()), "")
    return line[:2] if line else ""


def _path_exists_in_head(repo_root: Path, relative_path: str) -> bool:
    result = _run_git(repo_root, "cat-file", "-e", f"HEAD:{relative_path}")
    return result.returncode == 0


def _read_git_blob(repo_root: Path, object_name: str) -> str:
    result = _run_git(repo_root, "show", object_name)
    if result.returncode != 0:
        return ""
    return result.stdout


def _read_worktree_text(file_path: Path) -> str:
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8", errors="replace")


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _discard_new_path(
    repo_root: Path,
    file_path: Path,
    relative_path: str,
) -> subprocess.CompletedProcess[str]:
    result = _run_git(repo_root, "rm", "--cached", "-f", "--", relative_path)
    if result.returncode != 0:
        return result

    try:
        if file_path.is_dir():
            shutil.rmtree(file_path)
        elif file_path.exists():
            file_path.unlink()
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=["git", "-C", str(repo_root), "rm", "--cached", "-f", "--", relative_path],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )
    return result


def _git_failure_message(action: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout).strip() or "unknown git error"
    return f"Unable to {action}: {detail}"


def _format_summary(snapshot: DiffSnapshot) -> str:
    if snapshot.error:
        return snapshot.error
    left_title, right_title, comparison_title = _DIFF_COMPARISON_LABELS[snapshot.comparison]
    return f"{comparison_title}  {left_title} -> {right_title}"


def _format_status_chip(snapshot: DiffSnapshot) -> str:
    if snapshot.error:
        return "invalid path"
    if snapshot.status_code == "??":
        return "untracked"
    if snapshot.status_code == "":
        return "clean"

    index_code, worktree_code = snapshot.status_code
    labels: list[str] = []
    if index_code != " ":
        labels.append(f"index {_STATUS_LEGEND.get(index_code, index_code.lower())}")
    if worktree_code != " ":
        labels.append(f"worktree {_STATUS_LEGEND.get(worktree_code, worktree_code.lower())}")
    return "  ".join(labels) if labels else "clean"


def _render_unified_diff(snapshot: DiffSnapshot) -> Text:
    if snapshot.error:
        return Text(snapshot.error, style="bold #f44747", no_wrap=True)

    renderable = Text(no_wrap=True)
    for raw_line in snapshot.unified_diff.splitlines() or ["No changes for this file."]:
        style = _style_unified_line(raw_line)
        renderable.append(raw_line, style=style)
        renderable.append("\n")
    return renderable


def _style_unified_line(line: str) -> str:
    if line.startswith("+++ "):
        return "bold #4ec9b0"
    if line.startswith("--- "):
        return "bold #f44747"
    if line.startswith("@@"):
        return "bold #007acc"
    if line.startswith("+") and not line.startswith("+++"):
        return "black on #4ec9b0"
    if line.startswith("-") and not line.startswith("---"):
        return "white on #7f2f2f"
    for prefix, style in _UNIFIED_HEADER_STYLES:
        if line.startswith(prefix):
            return style
    return "#cccccc"


def _render_side_column(left_text: str, right_text: str, *, side: Literal["left", "right"]) -> Text:
    left_lines = left_text.splitlines()
    right_lines = right_text.splitlines()
    renderable = Text(no_wrap=True)
    line_pairs = list(_iter_diff_line_pairs(left_lines, right_lines))

    if not line_pairs:
        renderable.append("No changes for this file.\n", style="dim")
        return renderable

    for left_line, right_line, left_number, right_number, change_tag in line_pairs:
        if side == "left":
            content = left_line
            number = left_number
            marker = "-" if change_tag in {"delete", "replace"} and left_line is not None else " "
            style = _side_line_style(change_tag, side="left", has_content=left_line is not None)
        else:
            content = right_line
            number = right_number
            marker = "+" if change_tag in {"insert", "replace"} and right_line is not None else " "
            style = _side_line_style(change_tag, side="right", has_content=right_line is not None)

        renderable.append(f"{number:>4}" if number is not None else "    ", style="dim")
        renderable.append(f" {marker} ", style="bold #6b6b6b")
        renderable.append((content or "").expandtabs(4), style=style)
        renderable.append("\n")
    return renderable


def _iter_diff_line_pairs(
    left_lines: list[str],
    right_lines: list[str],
):
    matcher = difflib.SequenceMatcher(a=left_lines, b=right_lines, autojunk=False)
    left_number = 1
    right_number = 1

    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        chunk_left = left_lines[left_start:left_end]
        chunk_right = right_lines[right_start:right_end]
        row_count = max(len(chunk_left), len(chunk_right))

        for offset in range(row_count):
            left_line = chunk_left[offset] if offset < len(chunk_left) else None
            right_line = chunk_right[offset] if offset < len(chunk_right) else None
            current_left = left_number if left_line is not None else None
            current_right = right_number if right_line is not None else None
            yield left_line, right_line, current_left, current_right, tag
            if left_line is not None:
                left_number += 1
            if right_line is not None:
                right_number += 1


def _side_line_style(change_tag: str, *, side: Literal["left", "right"], has_content: bool) -> str:
    if not has_content:
        return "dim"
    if change_tag == "equal":
        return "#cccccc"
    if side == "left":
        return "white on #7f2f2f"
    return "black on #4ec9b0"


def _apply_button_state(button: Button, is_active: bool) -> None:
    if is_active:
        button.add_class(_BUTTON_ACTIVE_CLASS)
        return
    button.remove_class(_BUTTON_ACTIVE_CLASS)
