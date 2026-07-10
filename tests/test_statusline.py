"""Tests for orchestra/statusline.py — Claude Code statusline renderer.

Run from a scratch directory:
    cd /tmp && /abs/.venv/bin/pytest /abs/tests/test_statusline.py -q --no-cov -p no:cacheprovider
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
VENV_PYTHON = sys.executable
_NOW_STR = "2026-01-15T12:00:00+00:00"
_NOW = datetime.fromisoformat(_NOW_STR)


def _ts(delta_seconds: float = 0) -> str:
    """Return a UTC isoformat timestamp offset from _NOW."""
    return (_NOW + timedelta(seconds=delta_seconds)).isoformat()


def _write_status(orchestra_dir: Path, runs: dict) -> None:
    """Write a live_status.json into orchestra_dir."""
    orchestra_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": _ts(),
        "runs": runs,
    }
    (orchestra_dir / "live_status.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _run_statusline(
    *args: str,
    env: dict | None = None,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    """Invoke statusline as a subprocess. Returns (returncode, stdout, stderr)."""
    base_env = {**os.environ, "NO_COLOR": "1"}
    if env:
        base_env.update(env)
    proc = subprocess.run(
        [VENV_PYTHON, "-m", "orchestra.statusline", *args],
        capture_output=True,
        text=True,
        env=base_env,
        input=input_text or "",
    )
    return proc.returncode, proc.stdout, proc.stderr


def _render_in_process(
    project_dir: Path,
    now_str: str = _NOW_STR,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Call statusline render in-process with --no-color --now --dir flags."""
    import orchestra.statusline as sl

    args = [
        "--no-color",
        "--now", now_str,
        "--dir", str(project_dir),
        *(extra_args or []),
    ]
    captured: list[str] = []
    orig_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        try:
            sl.main(args)
        except SystemExit:
            pass
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = orig_stdout
    return [ln for ln in output.splitlines() if ln]


# ---------------------------------------------------------------------------
# Test 1: Running dual run with 2 agents
# ---------------------------------------------------------------------------

def test_running_dual_run_golden(tmp_path):
    """Golden exact-line output for a running dual run with 2 agents."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "abcdef1234567890": {
            "mode": "dual",
            "status": "running",
            "started_at": _ts(-134),  # 2m14s ago
            "heartbeat": _ts(-2),
            "agents": {
                "claude-primary": {
                    "status": "started",
                    "model": "anthropic/claude-sonnet-4-5",
                    "started_at": _ts(-130),
                    "ended_at": None,
                    "last_line": "Processing request...",
                    "est_cost_usd": None,
                },
                "claude-review": {
                    "status": "completed",
                    "model": "anthropic/claude-haiku-4-5",
                    "started_at": _ts(-90),
                    "ended_at": _ts(-10),
                    "last_line": None,
                    "est_cost_usd": 0.0004,
                },
            },
        }
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)

    # Golden exact-line comparison (--no-color, --now fixed at _NOW_STR)
    # Header: run id prefix (8 chars), mode, running badge ▶, elapsed 2:14, total cost $0.0004
    # Agent rows: alias[:10].ljust(10), model tail[:14].ljust(14), elapsed, cost, last_line
    expected_lines = [
        "🎼 abcdef12 dual ▶ 2:14 $0.0004",
        "  ▶ claude-pri  claude-sonnet-  2:10  —  Processing request...",
        "  ✔ claude-rev  claude-haiku-4  1:20  $0.0004  ",
    ]
    assert lines == expected_lines, (
        f"Golden mismatch.\nExpected: {expected_lines!r}\nGot:      {lines!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: waiting_approval run
# ---------------------------------------------------------------------------

def test_waiting_approval_badge(tmp_path):
    """waiting_approval run shows approval badge."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "approval00run00001": {
            "mode": "critical",
            "status": "waiting_approval",
            "started_at": _ts(-60),
            "heartbeat": _ts(-1),
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)
    lines = _render_in_process(tmp_path)

    assert len(lines) >= 1
    header = lines[0]
    assert "approval" in header, f"approval badge missing: {header}"


# ---------------------------------------------------------------------------
# Test 3: Stale active run (heartbeat 25s old) -> (stale) suffix
# ---------------------------------------------------------------------------

def test_stale_run_shows_stale_suffix(tmp_path):
    """Active run with heartbeat 25s older than --now shows (stale) suffix."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "stalerun12345678": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-120),
            "heartbeat": _ts(-25),  # 25s stale
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)
    lines = _render_in_process(tmp_path)

    assert any("stale" in ln for ln in lines), f"stale suffix missing: {lines}"


# ---------------------------------------------------------------------------
# Test 4: Recently-finished run shown; old terminal run hidden
# ---------------------------------------------------------------------------

def test_recently_finished_shown_old_hidden(tmp_path):
    """Completed run 10s ago shown; completed run 5min ago hidden."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "recent0000000001": {
            "mode": "ask",
            "status": "completed",
            "started_at": _ts(-70),
            "heartbeat": _ts(-10),  # 10s ago → show
            "agents": {},
        },
        "oldrun0000000001": {
            "mode": "ask",
            "status": "completed",
            "started_at": _ts(-400),
            "heartbeat": _ts(-300),  # 5min ago → hide
            "agents": {},
        },
    }
    _write_status(orch_dir, runs)
    lines = _render_in_process(tmp_path)

    output = "\n".join(lines)
    assert "recent00" in output, f"recent run missing: {output}"
    assert "oldrun00" not in output, f"old run should be hidden: {output}"


# ---------------------------------------------------------------------------
# Test 5: No file / corrupt JSON / empty runs → idle line, exit 0
# ---------------------------------------------------------------------------

def test_no_file_idle(tmp_path):
    """Missing .orchestra/live_status.json → idle line."""
    lines = _render_in_process(tmp_path)
    assert len(lines) == 1
    assert "orchestra idle" in lines[0]


def test_corrupt_json_idle(tmp_path):
    """Corrupt JSON → idle line, exit 0."""
    orch_dir = tmp_path / ".orchestra"
    orch_dir.mkdir()
    (orch_dir / "live_status.json").write_text("not valid json{{{", encoding="utf-8")
    lines = _render_in_process(tmp_path)
    assert len(lines) == 1
    assert "orchestra idle" in lines[0]


def test_empty_runs_idle(tmp_path):
    """Empty runs dict → idle line."""
    orch_dir = tmp_path / ".orchestra"
    _write_status(orch_dir, {})
    lines = _render_in_process(tmp_path)
    assert len(lines) == 1
    assert "orchestra idle" in lines[0]


# ---------------------------------------------------------------------------
# Test 6: Cap total agent rows at 5 + overflow line
# ---------------------------------------------------------------------------

def test_agent_rows_capped_at_five(tmp_path):
    """3 runs × 3 agents = 9 agents; capped at 5 + '+N more agents'."""
    orch_dir = tmp_path / ".orchestra"
    runs = {}
    for i in range(3):
        run_id = f"run{i:014d}"
        agents = {}
        for j in range(3):
            alias = f"agent{i}-{j}"
            agents[alias] = {
                "status": "started",
                "model": "some-model",
                "started_at": _ts(-60),
                "ended_at": None,
                "last_line": None,
                "est_cost_usd": None,
            }
        runs[run_id] = {
            "mode": "dual",
            "status": "running",
            "started_at": _ts(-60 - i),
            "heartbeat": _ts(-1),
            "agents": agents,
        }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    # Count agent rows (indented lines) — should be capped at 5
    agent_rows = [ln for ln in lines if ln.startswith("  ")]
    # Should have at most 6 non-header lines (5 agents + overflow)
    assert len(agent_rows) <= 6, f"Too many agent rows: {len(agent_rows)}: {lines}"
    # Should have an overflow line
    overflow = [ln for ln in agent_rows if "more agent" in ln]
    assert overflow, f"No overflow line found: {lines}"


# ---------------------------------------------------------------------------
# Test 7: Subprocess latency < 1.0s
# ---------------------------------------------------------------------------

def test_subprocess_latency(tmp_path):
    """Subprocess invocation completes within 1.0s."""
    orch_dir = tmp_path / ".orchestra"
    _write_status(orch_dir, {})  # empty → idle, no processing

    start = time.perf_counter()
    rc, stdout, stderr = _run_statusline("--dir", str(tmp_path), "--now", _NOW_STR)
    elapsed = time.perf_counter() - start

    assert rc == 0, f"Non-zero exit: {rc}, stderr={stderr}"
    assert elapsed < 1.0, f"Subprocess too slow: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Test 8: stdin fallback chain — invalid stdin falls back to cwd; valid stdin
#         resolves workspace dir.
# ---------------------------------------------------------------------------

def _run_statusline_in_process(
    stdin_text: str,
    cwd_path: Path,
    now_str: str = _NOW_STR,
) -> list[str]:
    """Invoke sl.main WITHOUT --dir so the stdin/cwd resolution path runs.

    Monkeypatches sys.stdin and os.getcwd inside orchestra.statusline so the
    full fallback chain executes.
    """
    import orchestra.statusline as sl

    captured: list[str] = []
    orig_stdout = sys.stdout
    orig_stdin = sys.stdin
    orig_getcwd = os.getcwd
    sys.stdout = io.StringIO()
    sys.stdin = io.StringIO(stdin_text)
    os.getcwd = lambda: str(cwd_path)  # type: ignore[assignment]
    try:
        try:
            sl.main(["--no-color", "--now", now_str])
        except SystemExit:
            pass
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = orig_stdout
        sys.stdin = orig_stdin
        os.getcwd = orig_getcwd  # type: ignore[assignment]
    return [ln for ln in output.splitlines() if ln]


def test_invalid_stdin_falls_back_to_cwd(tmp_path):
    """Garbage stdin causes _resolve_dir_from_stdin to return None; falls back to cwd.

    The fallback cwd points at a directory containing .orchestra/live_status.json
    so we can prove the full chain executed (output reflects the fixture, not idle).
    """
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "fallback12345678": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-60),
            "heartbeat": _ts(-1),
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)

    lines = _run_statusline_in_process(
        stdin_text="not json at all {{{",
        cwd_path=tmp_path,
    )

    # The fallback chain must have loaded the fixture: short run id prefix must appear
    # run_id[:8] of "fallback12345678" is "fallback"
    output = "\n".join(lines)
    assert "fallback" in output, (
        f"Expected fixture run id in output (proving cwd fallback executed): {lines!r}"
    )


def test_valid_stdin_resolves_workspace_dir(tmp_path):
    """Valid Claude Code session JSON with workspace.current_dir → fixture dir loaded.

    Proves the primary stdin path executes correctly end-to-end.
    """
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "stdinrun12345678": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-30),
            "heartbeat": _ts(-1),
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)

    session_json = json.dumps({
        "workspace": {
            "current_dir": str(tmp_path),
        }
    })

    lines = _run_statusline_in_process(
        stdin_text=session_json,
        cwd_path=Path("/"),  # cwd is irrelevant — workspace.current_dir wins
    )

    output = "\n".join(lines)
    assert "stdinrun" in output, (
        f"Expected fixture run id in output (proving stdin path executed): {lines!r}"
    )


# ---------------------------------------------------------------------------
# Test 9: Internal exception with DEBUG unset → idle line, exit 0, no traceback
# ---------------------------------------------------------------------------

def test_internal_exception_no_debug(tmp_path, monkeypatch):
    """With ORCHESTRA_STATUSLINE_DEBUG unset, internal exception → idle, exit 0."""
    import orchestra.statusline as sl

    original_find = sl._find_status_file

    def _exploding_find(start_dir: Path):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(sl, "_find_status_file", _exploding_find)
    monkeypatch.delenv("ORCHESTRA_STATUSLINE_DEBUG", raising=False)

    captured_stdout: list[str] = []
    captured_stderr: list[str] = []

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        try:
            sl.main(["--no-color", "--now", _NOW_STR, "--dir", str(tmp_path)])
        except SystemExit:
            pass
        stdout_val = sys.stdout.getvalue()
        stderr_val = sys.stderr.getvalue()
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    assert "orchestra idle" in stdout_val, f"Expected idle: {stdout_val!r}"
    assert "Traceback" not in stderr_val, f"Traceback leaked to stderr: {stderr_val!r}"


# ---------------------------------------------------------------------------
# Test 10: Subprocess with --no-color exits 0 and produces expected output
# ---------------------------------------------------------------------------

def test_subprocess_no_color_exit_zero(tmp_path):
    """Subprocess with --no-color and --now exits 0 and produces idle for empty status."""
    orch_dir = tmp_path / ".orchestra"
    _write_status(orch_dir, {})

    rc, stdout, stderr = _run_statusline(
        "--dir", str(tmp_path),
        "--no-color",
        "--now", _NOW_STR,
    )
    assert rc == 0, f"Non-zero exit: rc={rc}, stderr={stderr}"
    assert "orchestra idle" in stdout, f"Missing idle: {stdout!r}"
    assert "Traceback" not in stderr


# ---------------------------------------------------------------------------
# Test 11: Naive timestamps are coerced to UTC (finding #2)
# ---------------------------------------------------------------------------

def test_naive_and_aware_ts_both_render(tmp_path):
    """Runs with naive timestamps are coerced to UTC; both renders appear (not idle)."""
    orch_dir = tmp_path / ".orchestra"
    # One aware-tz run
    aware_ts = _ts(-10)  # "2026-01-15T11:59:50+00:00"
    # One naive-tz run (no +00:00 suffix)
    naive_ts = (_NOW + timedelta(seconds=-20)).replace(tzinfo=None).isoformat()
    assert "+" not in naive_ts, f"expected naive ts, got: {naive_ts}"

    runs = {
        "awaretsnrun0001": {
            "mode": "ask",
            "status": "running",
            "started_at": aware_ts,
            "heartbeat": aware_ts,
            "agents": {},
        },
        "naivetsnrun0001": {
            "mode": "ask",
            "status": "running",
            "started_at": naive_ts,
            "heartbeat": naive_ts,
            "agents": {},
        },
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    output = "\n".join(lines)
    # Neither should be idle; both run IDs must appear
    assert "orchestra idle" not in output, f"should not be idle: {output}"
    assert "awarets" in output, f"aware-ts run missing: {output}"
    assert "naivetsn" in output, f"naive-ts run missing: {output}"


# ---------------------------------------------------------------------------
# Test 12: Active runs with stale/missing heartbeat are hidden (finding #3)
# ---------------------------------------------------------------------------

def test_active_run_hidden_when_heartbeat_very_old(tmp_path):
    """Active run with heartbeat 700s old (> 600s cutoff) → hidden (idle shown)."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "deadsession12345": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-800),
            "heartbeat": _ts(-700),  # 700s > 600s cutoff → hidden
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    output = "\n".join(lines)
    assert "orchestra idle" in output, f"expected idle for dead active run: {output}"


def test_active_run_shown_with_stale_when_heartbeat_30s_old(tmp_path):
    """Active run with heartbeat 30s old (20s–600s) → shown with (stale) suffix."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "staleactive12345": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-200),
            "heartbeat": _ts(-30),  # 30s → stale but within 600s → show with (stale)
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    output = "\n".join(lines)
    assert "stale" in output, f"expected (stale) in output: {output}"
    assert "staleact" in output, f"run should still show: {output}"


def test_active_run_hidden_when_heartbeat_missing(tmp_path):
    """Active run with no heartbeat field → hidden (idle shown)."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "nohbsession12345": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-60),
            # heartbeat key deliberately absent
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    output = "\n".join(lines)
    assert "orchestra idle" in output, f"expected idle for no-heartbeat active run: {output}"


# ---------------------------------------------------------------------------
# Test 13: Sanitization strips control characters (finding #4)
# ---------------------------------------------------------------------------

def test_sanitize_strips_ansi_and_control_in_last_line(tmp_path):
    """last_line containing ANSI escape and BEL → stripped from output."""
    orch_dir = tmp_path / ".orchestra"
    runs = {
        "sanitizerun12345": {
            "mode": "ask",
            "status": "running",
            "started_at": _ts(-10),
            "heartbeat": _ts(-1),
            "agents": {
                "cdx": {
                    "status": "started",
                    "model": "some-model",
                    "started_at": _ts(-10),
                    "ended_at": None,
                    "last_line": "normal\x1b[2Jclear\x07bel",
                    "est_cost_usd": None,
                }
            },
        }
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    output = "\n".join(lines)
    assert "\x1b" not in output, f"ANSI escape leaked into output: {output!r}"
    assert "\x07" not in output, f"BEL leaked into output: {output!r}"
    # Meaningful text still present
    assert "normal" in output and "clear" in output and "bel" in output


def test_sanitize_strips_control_in_mode(tmp_path):
    """mode field containing control character → stripped; mode is length-capped."""
    orch_dir = tmp_path / ".orchestra"
    # mode with embedded escape and excessively long string
    runs = {
        "sanitizemoderun1": {
            "mode": "ask\x1b[31mINJECTED",  # has ESC + extra chars
            "status": "running",
            "started_at": _ts(-10),
            "heartbeat": _ts(-1),
            "agents": {},
        }
    }
    _write_status(orch_dir, runs)

    lines = _render_in_process(tmp_path)
    output = "\n".join(lines)
    assert "\x1b" not in output, f"ANSI escape in mode leaked: {output!r}"
