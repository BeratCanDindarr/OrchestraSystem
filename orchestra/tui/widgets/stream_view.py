"""Stream view widget for Orchestra TUI right pane."""
from __future__ import annotations

import os
from pathlib import Path

from rich.text import Text
from textual.widgets import RichLog


_CATEGORY_RULES: list[tuple[tuple[str, ...], str, str]] = [
    (("[thinking]", "thinking:", "=== thinking"), "thinking", "cyan dim"),
    (("[tool]", "tool:", "shell ", "bash ", "[input]"), "tool", "yellow"),
    (("[output]", "output:", "## ", "### "), "output", "white"),
]


def _classify(line: str) -> tuple[str, str]:
    lower = line.lower()
    for prefixes, category, color in _CATEGORY_RULES:
        if any(lower.startswith(p.lower()) for p in prefixes):
            return category, color
    return "output", "#aaaaaa"


class StreamView(RichLog):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, markup=False, auto_scroll=True)
        self._log_path: Path | None = None
        self._file_pos: int = 0
        self._buffer: list[tuple[str, str, str]] = []  # (line, category, color)
        self._filters: dict[str, bool] = {
            "thinking": True,
            "tool": True,
            "output": True,
        }

    def set_log_path(self, path: Path) -> None:
        if self._log_path != path:
            self._log_path = path
            self._file_pos = 0
            self._buffer.clear()
            self.clear()
            self.poll()

    def set_filters(self, thinking: bool, tool: bool, output: bool) -> None:
        self._filters = {"thinking": thinking, "tool": tool, "output": output}
        self._redraw()

    def poll(self) -> None:
        if not self._log_path or not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._file_pos)
                new_lines = f.readlines()
                self._file_pos = f.tell()

            for raw in new_lines:
                line = raw.rstrip("\n")
                category, color = _classify(line)
                self._buffer.append((line, category, color))
                if self._filters.get(category, True):
                    self.write(Text(line, style=color))
        except OSError:
            pass

    def _redraw(self) -> None:
        self.clear()
        for line, category, color in self._buffer:
            if self._filters.get(category, True):
                self.write(Text(line, style=color))
