"""Agent stream widget for Orchestra TUI v2."""
from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual import work
from textual.message import Message
from textual.reactive import reactive
from textual.worker import get_current_worker
from textual.widgets import RichLog


_TOOL_HINTS = (
    "tool",
    "function",
    "exec_command",
    "apply_patch",
    "web.run",
    "mcp__",
)


@dataclass(frozen=True)
class StreamPollResult:
    next_position: int
    chunk: bytes
    reset: bool = False
    has_more: bool = False


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

    def __init__(self, *, max_read_bytes: int = 65536, **kwargs: Any) -> None:
        super().__init__(markup=False, auto_scroll=True, wrap=True, max_lines=4000, **kwargs)
        self.max_read_bytes = max(4096, max_read_bytes)
        self._file_pos = 0
        self._pending_fragment = ""
        self._buffer: list[str] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._poll_inflight = False
        self._poll_queued = False

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
            self._write_header()
            self.poll()

    def append_stdout(self, content: str) -> None:
        """Append raw stdout content, preserving ANSI color codes."""
        self._ingest_chunk(content)

    @work(exclusive=True, thread=True)
    def poll(self) -> None:
        """Queue a bounded background read for newly appended log content."""
        path = self.log_path
        if path is None or not path.exists():
            return
        if self._poll_inflight:
            self._poll_queued = True
            return
        self._poll_inflight = True
        self._poll_worker(path, self._file_pos)

    @work(thread=True, group="agent-stream-poll", exit_on_error=False)
    def _poll_worker(self, path: Path, start_position: int) -> None:
        result = _read_log_chunk(path, start_position, self.max_read_bytes)
        worker = get_current_worker()
        if worker.is_cancelled:
            return
        self.app.call_from_thread(self._complete_poll, path, result)

    def _complete_poll(self, path: Path, result: StreamPollResult) -> None:
        self._poll_inflight = False

        if path != self.log_path:
            self._poll_queued = False
            if self.log_path is not None and self.log_path.exists():
                self.poll()
            return

        if result.reset:
            self._reset_stream_state()
            self.clear()
            self._write_header()

        self._file_pos = result.next_position
        if result.chunk:
            self._ingest_bytes(result.chunk)

        if result.has_more or self._poll_queued:
            self._poll_queued = False
            self.poll()

    def _reset_stream_state(self) -> None:
        self._file_pos = 0
        self._pending_fragment = ""
        self._buffer.clear()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.line_count = 0

    def _write_header(self) -> None:
        if self.agent_alias:
            self.write(Text.assemble(("agent ", "dim"), (self.agent_alias, "bold cyan")))

    def _ingest_bytes(self, chunk: bytes) -> None:
        text = self._decoder.decode(chunk, final=False)
        if text:
            self._ingest_chunk(text)

    def _ingest_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        text = self._pending_fragment + chunk
        parts = text.splitlines(keepends=True)

        if parts and not parts[-1].endswith(("\n", "\r")):
            self._pending_fragment = parts.pop()
        else:
            self._pending_fragment = ""

        lines = [part.rstrip("\r\n") for part in parts]
        self._append_lines(lines)

    def _append_lines(self, lines: list[str]) -> None:
        if not lines:
            return

        render_lines: list[Text] = []
        tool_lines: list[str] = []
        for line in lines:
            text = Text.from_ansi(line) if "\x1b[" in line else Text(line)
            if _looks_like_tool_use(line):
                text = Text.assemble(("⚙ ", "yellow"), text)
                tool_lines.append(line)
            render_lines.append(text)

        self._buffer.extend(lines)
        if self.max_lines is not None and len(self._buffer) > self.max_lines:
            del self._buffer[: len(self._buffer) - self.max_lines]

        self.line_count += len(lines)
        self.write(Text("\n").join(render_lines))
        for line in tool_lines:
            self.post_message(self.ToolUseDetected(self.agent_alias, line))
        self.post_message(self.StreamLineAppended(self.agent_alias, lines[-1], self.line_count))


def _looks_like_tool_use(line: str) -> bool:
    lower = line.lower()
    return any(hint in lower for hint in _TOOL_HINTS)


def _read_log_chunk(path: Path, start_position: int, max_read_bytes: int) -> StreamPollResult:
    try:
        size = path.stat().st_size
    except OSError:
        return StreamPollResult(start_position, b"")

    reset = size < start_position
    read_from = 0 if reset else start_position
    if size <= read_from:
        return StreamPollResult(read_from, b"", reset=reset, has_more=False)

    read_size = min(size - read_from, max_read_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(read_from)
            chunk = handle.read(read_size)
            next_position = handle.tell()
    except OSError:
        return StreamPollResult(read_from, b"", reset=reset, has_more=False)

    return StreamPollResult(
        next_position=next_position,
        chunk=chunk,
        reset=reset,
        has_more=next_position < size,
    )
