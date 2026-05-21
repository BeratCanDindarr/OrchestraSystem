"""Agent stream widget for Orchestra TUI v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.text import Text
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import RichLog


_TOOL_HINTS = (
    "tool",
    "function",
    "exec_command",
    "apply_patch",
    "web.run",
    "mcp__",
)


class AgentStreamWidget(RichLog):
    """Scrollable agent stdout panel with ANSI rendering and tool-use markers."""

    DEFAULT_CSS = """
    AgentStreamWidget {
        height: 100%;
        border: tall $primary;
    }
    """

    agent_alias: reactive[str | None] = reactive(None)
    log_path: reactive[Path | None] = reactive(None)
    line_count: reactive[int] = reactive(0)

    class ToolUseDetected(Message):
        def __init__(self, agent_alias: str | None, line: str) -> None:
            super().__init__()
            self.agent_alias = agent_alias
            self.line = line

    class StreamLineAppended(Message):
        def __init__(self, agent_alias: str | None, line: str, line_count: int) -> None:
            super().__init__()
            self.agent_alias = agent_alias
            self.line = line
            self.line_count = line_count

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(markup=False, auto_scroll=True, wrap=True, max_lines=4000, **kwargs)
        self._file_pos = 0
        self._pending_fragment = ""
        self._buffer: list[str] = []

    def set_agent(self, alias: str | None, log_path: Path | str | None = None) -> None:
        self.agent_alias = alias
        self.set_log_path(Path(log_path) if log_path is not None else None)

    def set_log_path(self, path: Path | str | None) -> None:
        next_path = Path(path) if path is not None else None
        if self.log_path == next_path:
            return
        self.log_path = next_path
        self._reset_stream_state()
        if self.is_mounted:
            self.clear()
            if self.agent_alias:
                self.write(Text.assemble(("agent ", "dim"), (self.agent_alias, "bold cyan")))
        self.poll()

    def append_stdout(self, content: str) -> None:
        """Append raw stdout content, preserving ANSI color codes."""
        self._ingest_chunk(content)

    def poll(self) -> None:
        """Read newly appended log lines from the current agent log file."""
        path = self.log_path
        if path is None or not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._file_pos)
                chunk = handle.read()
                self._file_pos = handle.tell()
        except OSError:
            return
        if chunk:
            self._ingest_chunk(chunk)

    def _reset_stream_state(self) -> None:
        self._file_pos = 0
        self._pending_fragment = ""
        self._buffer.clear()
        self.line_count = 0

    def _ingest_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        text = self._pending_fragment + chunk
        parts = text.splitlines(keepends=True)

        if parts and not parts[-1].endswith(("\n", "\r")):
            self._pending_fragment = parts.pop()
        else:
            self._pending_fragment = ""

        for part in parts:
            self._append_line(part.rstrip("\r\n"))

    def _append_line(self, line: str) -> None:
        text = Text.from_ansi(line)
        if _looks_like_tool_use(line):
            text = Text.assemble(("⚙ ", "yellow"), text)
            self.post_message(self.ToolUseDetected(self.agent_alias, line))

        self._buffer.append(line)
        self.line_count += 1
        self.write(text)
        self.post_message(self.StreamLineAppended(self.agent_alias, line, self.line_count))


def _looks_like_tool_use(line: str) -> bool:
    lower = line.lower()
    return any(hint in lower for hint in _TOOL_HINTS)
