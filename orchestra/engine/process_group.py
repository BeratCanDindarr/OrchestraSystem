"""Run provider commands with Inactivity Timeout and surgical idempotency."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import hashlib
from typing import Callable, Optional

from orchestra.providers.base import BaseProvider
from orchestra.storage.events import find_tool_result

def _kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass

class ProcessGroupManager:
    def __init__(self) -> None:
        self._pids: set[int] = set()
        self._lock = threading.Lock()

    def register(self, pid: int) -> None:
        with self._lock: self._pids.add(pid)

    def unregister(self, pid: int | None) -> None:
        if pid is None: return
        with self._lock: self._pids.discard(pid)

    def kill_all(self) -> None:
        pgids = list(self._pids)
        for pgid in pgids: _kill_group(pgid, signal.SIGTERM)
        time.sleep(0.5)
        for pgid in pgids: _kill_group(pgid, signal.SIGKILL)
        with self._lock: self._pids.clear()

def generate_idempotency_key(cmd: list[str], prompt: str) -> str:
    payload = "".join(cmd) + prompt
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def run_provider_process(
    provider: BaseProvider,
    prompt: str,
    effort_or_model: str,
    timeout: int = 1200, 
    inactivity_timeout: int = 300, 
    process_manager: ProcessGroupManager | None = None,
    pid_callback: Callable[[int | None], None] | None = None,
    idempotency_key: str | None = None,
    stream_callback: Callable[[str], None] | None = None, # 🚀 NEW: For Live Streaming
    cwd: str | os.PathLike | None = None,
) -> tuple[str, int, str]:
    # Providers with native_run() bypass subprocess entirely (e.g. Ollama HTTP)
    if hasattr(provider, "native_run"):
        stdout, returncode = provider.native_run(
            prompt, effort_or_model, timeout=timeout, stream_callback=stream_callback
        )
        if pid_callback:
            pid_callback(None)
        return stdout, returncode, ""

    cmd = provider.build_command(prompt, effort_or_model)
    key = idempotency_key or generate_idempotency_key(cmd, prompt)

    past_result = find_tool_result(key)
    if past_result:
        return past_result.get("stdout", ""), 0, past_result.get("stderr", "")

    process: subprocess.Popen[str] | None = None
    stdout_buf = []
    stderr_buf = []
    last_output_time = time.time()
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid, cwd=cwd
        )
        if process_manager: process_manager.register(process.pid)
        if pid_callback: pid_callback(process.pid)

        def _read_stream(stream, buffer, is_stdout):
            nonlocal last_output_time
            for line in iter(stream.readline, ''):
                buffer.append(line)
                last_output_time = time.time()
                if is_stdout and stream_callback:
                    stream_callback(line) # 🚀 STREAMING TO UNITY
            stream.close()

        t_out = threading.Thread(target=_read_stream, args=(process.stdout, stdout_buf, True))
        t_err = threading.Thread(target=_read_stream, args=(process.stderr, stderr_buf, False))
        t_out.start()
        t_err.start()

        while process.poll() is None:
            now = time.time()
            if now - start_time > timeout: break
            if now - last_output_time > inactivity_timeout: break
            time.sleep(0.5)

        if process.poll() is None:
            _kill_group(process.pid, signal.SIGKILL)
            return "".join(stdout_buf), 124, "Timeout"

        t_out.join()
        t_err.join()
        return "".join(stdout_buf).strip(), process.returncode, "".join(stderr_buf).strip()

    except Exception as exc:
        return f"[ERROR] {exc}", 1, str(exc)
    finally:
        if process_manager and process: process_manager.unregister(process.pid)
        if pid_callback: pid_callback(None)
