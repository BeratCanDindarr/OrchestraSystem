"""Claude Code statusline renderer — stdlib only, no orchestra imports.

Reads ``.orchestra/live_status.json`` from the project directory tree and
prints one or more status lines to stdout.  Designed to be invoked every 1-2s
by Claude Code as its statusline command.

Usage::

    python -m orchestra.statusline [--no-color] [--now <iso>] [--dir <path>]

Exit code is always 0; any error silently falls back to the idle line.
Set ``ORCHESTRA_STATUSLINE_DEBUG=1`` to get tracebacks on stderr instead.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_CYAN = "\x1b[36m"
_YELLOW = "\x1b[33m"
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"

_NO_COLOR = False  # overridden by flag / env var


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI colour code, or return plain text when colours disabled."""
    if _NO_COLOR:
        return text
    return f"{code}{text}{_RESET}"


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MODE_MAX_LEN = 12


def _sanitize(s: str) -> str:
    """Strip ASCII control characters (including ESC) from a file-sourced string."""
    return _CONTROL_RE.sub("", s)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STALE_THRESHOLD_S = 20.0
_FLASH_THRESHOLD_S = 30.0
_HIDE_ACTIVE_AFTER_SECONDS = 600.0
_MAX_AGENT_ROWS = 5
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_ACTIVE_STATUSES = frozenset({"running", "waiting_approval"})
_ROW_WIDTH = 110


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp string; return None on error.

    Naive timestamps (no timezone info) are coerced to UTC to prevent
    TypeError when comparing against aware datetimes.
    """
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _elapsed(start: Optional[datetime], end: Optional[datetime], now: datetime) -> str:
    """Format elapsed time as m:ss; missing start → '--:--'."""
    if start is None:
        return "--:--"
    finish = end if end is not None else now
    total_s = max(0, int((finish - start).total_seconds()))
    total_s = min(total_s, 99 * 60 + 59)  # cap at 99:59
    minutes = total_s // 60
    seconds = total_s % 60
    return f"{minutes}:{seconds:02d}"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_status_file(start_dir: Path) -> Optional[Path]:
    """Walk from *start_dir* upward searching for .orchestra/live_status.json."""
    current = start_dir.resolve()
    for _ in range(64):  # safety: at most 64 levels
        candidate = current / ".orchestra" / "live_status.json"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ---------------------------------------------------------------------------
# Status file loading
# ---------------------------------------------------------------------------

def _load_status(path: Path) -> dict:
    """Load and parse live_status.json; return empty dict on any error."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def _is_displayable(run: dict, now: datetime) -> bool:
    """Return True if this run should be displayed.

    Active runs (running/waiting_approval) are hidden when:
    - the heartbeat field is absent (writer died without ever writing one), or
    - the heartbeat is older than HIDE_ACTIVE_AFTER_SECONDS (writer died).
    Active runs with heartbeat between STALE_THRESHOLD_S and HIDE_ACTIVE_AFTER_SECONDS
    remain visible (with a (stale) suffix added by _is_stale).
    """
    status = run.get("status", "")
    if status in _ACTIVE_STATUSES:
        hb = _parse_ts(run.get("heartbeat"))
        if hb is None:
            return False
        age = (now - hb).total_seconds()
        return age < _HIDE_ACTIVE_AFTER_SECONDS
    if status in _TERMINAL_STATUSES:
        hb = _parse_ts(run.get("heartbeat"))
        if hb is not None:
            age = (now - hb).total_seconds()
            return age <= _FLASH_THRESHOLD_S
    return False


def _is_stale(run: dict, now: datetime) -> bool:
    """Return True if an active run's heartbeat is older than the stale threshold."""
    status = run.get("status", "")
    if status not in _ACTIVE_STATUSES:
        return False
    hb = _parse_ts(run.get("heartbeat"))
    if hb is None:
        return False
    return (now - hb).total_seconds() > _STALE_THRESHOLD_S


def _is_dim_run(run: dict, now: datetime) -> bool:
    """Dim completed/cancelled/stale runs."""
    status = run.get("status", "")
    if status in _TERMINAL_STATUSES:
        return True
    return _is_stale(run, now)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _model_tail(model: Optional[str]) -> str:
    """Extract last path-ish segment from a model string (e.g. 'ollama/qwen2.5:7b' → 'qwen2.5:7b')."""
    if not model:
        return ""
    return model.split("/")[-1]


def _format_cost(cost: Optional[float]) -> str:
    """Format a cost as '$0.0004' or '—'."""
    if cost is None:
        return "—"  # em dash
    return f"${cost:.4f}"


def _truncate_visible(text: str, max_len: int) -> str:
    """Truncate text to at most max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "…"


def _agent_icon_color(status: str) -> tuple[str, str]:
    """Return (icon, color_code) for an agent status."""
    mapping = {
        "started":   ("▶", _CYAN),
        "retrying":  ("↻", _YELLOW),
        "completed": ("✔", _GREEN),
        "failed":    ("✘", _RED),
        "cancelled": ("∅", _DIM),
    }
    return mapping.get(status, ("?", ""))


def _run_badge(status: str) -> tuple[str, str]:
    """Return (badge_text, color_code) for a run status."""
    mapping = {
        "running":          ("▶", _CYAN),
        "waiting_approval": ("⏸ approval", _YELLOW),
        "completed":        ("✔", _GREEN),
        "failed":           ("✘", _RED),
        "cancelled":        ("∅", _DIM),
    }
    return mapping.get(status, (status, ""))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_agent_row(alias: str, agent: dict, now: datetime, dim: bool) -> str:
    """Render a single agent row (no trailing newline)."""
    agent_status = agent.get("status", "?")
    icon, icon_color = _agent_icon_color(agent_status)
    icon_str = _c(icon_color, icon)

    alias_padded = _sanitize(alias)[:10].ljust(10)
    model_str = _model_tail(_sanitize(agent.get("model") or "")) or ""
    model_padded = model_str[:14].ljust(14)

    a_started = _parse_ts(agent.get("started_at"))
    a_ended = _parse_ts(agent.get("ended_at"))
    a_elapsed = _elapsed(a_started, a_ended, now)

    a_cost = _format_cost(agent.get("est_cost_usd"))
    last_line = _sanitize(agent.get("last_line") or "")

    # Fixed portion visible length: "  X alias______  model____________  m:ss  $0.0000  "
    fixed_len = 2 + 1 + 1 + 10 + 2 + 14 + 2 + 5 + 2 + len(a_cost) + 1
    remaining = max(0, _ROW_WIDTH - fixed_len)
    last_line_trunc = _truncate_visible(last_line, remaining)

    row = f"  {icon_str} {alias_padded}  {model_padded}  {a_elapsed}  {a_cost}  {last_line_trunc}"
    if dim:
        row = _c(_DIM, row) if not _NO_COLOR else row
    return row


def _render_run(
    run_id: str,
    run: dict,
    now: datetime,
    dim: bool,
) -> list[str]:
    """Render one run as a list of text lines (no trailing newlines)."""
    status = run.get("status", "unknown")
    short_id = run_id[:8]
    mode = _sanitize(run.get("mode", "?"))[:_MODE_MAX_LEN]
    badge_text, badge_color = _run_badge(status)
    badge = _c(badge_color, badge_text)

    started = _parse_ts(run.get("started_at"))
    elapsed = _elapsed(started, None, now)

    agents_data: dict = run.get("agents") or {}
    total_cost: float = sum(
        a.get("est_cost_usd") or 0.0
        for a in agents_data.values()
        if a.get("est_cost_usd") is not None
    )
    cost_str = f"${total_cost:.4f}"

    stale_suffix = " (stale)" if _is_stale(run, now) else ""
    header = f"🎼 {short_id} {mode} {badge} {elapsed} {cost_str}{stale_suffix}"
    if dim:
        header = _c(_DIM, header) if not _NO_COLOR else header

    agent_rows = [
        _render_agent_row(alias, agent, now, dim)
        for alias, agent in agents_data.items()
    ]
    return [header, *agent_rows]


def _render(runs_data: dict, now: datetime) -> list[str]:
    """Render all displayable runs; return list of output lines."""
    displayable = [
        (run_id, run)
        for run_id, run in runs_data.items()
        if _is_displayable(run, now)
    ]

    if not displayable:
        idle = "🎼 orchestra idle"
        return [_c(_DIM, idle) if not _NO_COLOR else idle]

    # Sort newest first by started_at
    def _sort_key(item: tuple) -> datetime:
        ts = _parse_ts(item[1].get("started_at"))
        if ts is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return ts

    displayable.sort(key=_sort_key, reverse=True)

    # Collect all lines with agent count tracking
    all_lines: list[str] = []
    total_agent_rows = 0
    overflow_count = 0

    for run_id, run in displayable:
        dim = _is_dim_run(run, now)
        run_lines = _render_run(run_id, run, now, dim)

        if not run_lines:
            continue

        header = run_lines[0]
        agent_lines = run_lines[1:]

        all_lines.append(header)

        for agent_line in agent_lines:
            if total_agent_rows < _MAX_AGENT_ROWS:
                all_lines.append(agent_line)
                total_agent_rows += 1
            else:
                overflow_count += 1

    if overflow_count > 0:
        overflow_line = f"   … +{overflow_count} more agent{'s' if overflow_count != 1 else ''}"
        all_lines.append(overflow_line)

    return all_lines


# ---------------------------------------------------------------------------
# stdin / directory resolution
# ---------------------------------------------------------------------------

def _resolve_dir_from_stdin() -> Optional[Path]:
    """Parse Claude Code session JSON from stdin; extract project directory."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return None
        data = json.loads(raw)
        # Primary path: nested workspace object
        ws = data.get("workspace", {})
        if isinstance(ws, dict):
            for sub in ("current_dir", "project_dir"):
                val = ws.get(sub)
                if val and isinstance(val, str):
                    return Path(val)
        # Fallback: flat "cwd" key
        val = data.get("cwd")
        if val and isinstance(val, str):
            return Path(val)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> tuple[bool, Optional[datetime], Optional[Path]]:
    """Parse CLI flags; return (no_color, now_override, dir_override)."""
    no_color = False
    now_override: Optional[datetime] = None
    dir_override: Optional[Path] = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--no-color":
            no_color = True
        elif arg == "--now" and i + 1 < len(argv):
            i += 1
            try:
                now_override = datetime.fromisoformat(argv[i])
                # Ensure timezone-aware
                if now_override.tzinfo is None:
                    now_override = now_override.replace(tzinfo=timezone.utc)
            except (ValueError, IndexError):
                pass
        elif arg == "--dir" and i + 1 < len(argv):
            i += 1
            dir_override = Path(argv[i])
        i += 1
    return no_color, now_override, dir_override


# ---------------------------------------------------------------------------
# Inner implementation (may raise)
# ---------------------------------------------------------------------------

def _main_inner(argv: list[str]) -> None:
    """Core logic; may raise — callers are responsible for exception handling."""
    global _NO_COLOR

    no_color, now_override, dir_override = _parse_args(argv)

    # Respect NO_COLOR env var
    if os.environ.get("NO_COLOR") or no_color:
        _NO_COLOR = True
    else:
        _NO_COLOR = False

    now = now_override if now_override is not None else datetime.now(timezone.utc)

    # Resolve project directory
    if dir_override is not None:
        start_dir = dir_override
    else:
        start_dir = _resolve_dir_from_stdin() or Path(os.getcwd())

    # Find the status file
    status_file = _find_status_file(start_dir)

    idle_line = "🎼 orchestra idle"
    if not _NO_COLOR:
        idle_line = _c(_DIM, idle_line)

    if status_file is None:
        print(idle_line)
        return

    data = _load_status(status_file)
    runs_data: dict = data.get("runs") or {}

    output_lines = _render(runs_data, now)
    for line in output_lines:
        print(line)


# ---------------------------------------------------------------------------
# Public entry point — never raises
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point.

    Never raises: any unhandled exception prints the idle line (and, when
    ``ORCHESTRA_STATUSLINE_DEBUG=1``, the traceback to stderr).
    """
    if argv is None:
        argv = sys.argv[1:]

    _debug = os.environ.get("ORCHESTRA_STATUSLINE_DEBUG") == "1"
    try:
        _main_inner(argv)
    except Exception:
        if _debug:
            traceback.print_exc(file=sys.stderr)
        idle = "🎼 orchestra idle"
        if not _NO_COLOR:
            idle = _c(_DIM, idle)
        print(idle)


# ---------------------------------------------------------------------------
# __main__ guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    sys.exit(0)
