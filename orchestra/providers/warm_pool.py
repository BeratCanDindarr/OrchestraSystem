"""Claude CLI processes started ahead of time, so an agent call does not wait for the start-up.

Starting `claude -p` with three MCP servers took ~15 s before the model was asked anything; a
process started with `--input-format stream-json` does all that and then waits for its prompt
on stdin (answer in ~3 s). The pool keeps one such process per command and working folder, hands
it to the next call with that command, and starts the next one. A process answers one task, so
no task sees another's conversation; one waiting longer than MAX_IDLE_SECONDS is ended (its
skills or MCP config may have changed).

Only the long-lived MCP server turns the pool on (`enable()`): a one-shot `orchestra ask` would
leave a process started for nobody.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import threading
import time
from typing import Optional

MAX_IDLE_SECONDS = 600

_lock = threading.Lock()
_enabled = False
_waiting: dict[tuple, tuple[subprocess.Popen, float]] = {}


def _key(command: list[str], cwd) -> tuple:
    return tuple(command[:-1]), str(cwd or "")   # the command without its prompt (the last argument)


def _end(process: subprocess.Popen) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)   # the CLI and the MCP servers it started
        except OSError:
            process.kill()
    process.wait()


def enable() -> None:
    global _enabled
    _enabled = True


def disable() -> None:
    """Turn the pool off and end every waiting process."""
    global _enabled
    _enabled = False
    with _lock:
        waiting = [process for process, _ in _waiting.values()]
        _waiting.clear()
    for process in waiting:
        _end(process)


atexit.register(disable)


def waiting() -> int:
    return len(_waiting)


def wait_ready(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not _waiting and time.monotonic() < deadline:
        time.sleep(0.05)


def take(command: list[str], cwd) -> Optional[subprocess.Popen]:
    """A started process for this command, or None (pool off, none waiting, or too old)."""
    if not _enabled:
        return None
    with _lock:
        entry = _waiting.pop(_key(command, cwd), None)
    if entry is None:
        return None
    process, born = entry
    if process.poll() is not None or time.monotonic() - born > MAX_IDLE_SECONDS:
        _end(process)
        return None
    return process


def refill(command: list[str], cwd) -> None:
    """Start the process the next call with this command will take."""
    if not _enabled:
        return
    key = _key(command, cwd)
    with _lock:
        if key in _waiting:
            return
        process = subprocess.Popen([*command[:-1], "--input-format", "stream-json"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd,
                                   start_new_session=True)
        _waiting[key] = (process, time.monotonic())
    timer = threading.Timer(MAX_IDLE_SECONDS, _expire, (key, process))
    timer.daemon = True   # an idle MCP server must not keep a waiting CLI and its MCP servers for hours
    timer.start()


def _expire(key: tuple, process: subprocess.Popen) -> None:
    with _lock:
        if _waiting.get(key, (None,))[0] is not process:
            return   # taken, or replaced
        del _waiting[key]
    _end(process)


def send(process: subprocess.Popen, prompt: str) -> None:
    """Give a waiting process its one task; it answers and exits."""
    assert process.stdin is not None
    process.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": prompt}}) + "\n")
    process.stdin.close()
    process.stdin = None   # communicate() must not touch the closed pipe
