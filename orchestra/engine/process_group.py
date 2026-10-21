"""Run provider commands with Inactivity Timeout and surgical idempotency."""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

_IS_WINDOWS = sys.platform == "win32"

from orchestra import config

def _kill_group(pgid: int, sig: int) -> None:
    try:
        if _IS_WINDOWS:
            os.kill(pgid, sig)
        else:
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
        grace_seconds = float(config.execution_config().get("kill_grace_seconds", 0.5))
        pgids = list(self._pids)
        term_sig = signal.SIGTERM if not _IS_WINDOWS else signal.SIGBREAK
        kill_sig = signal.SIGTERM if _IS_WINDOWS else signal.SIGKILL  # SIGKILL unavailable on Windows
        for pgid in pgids: _kill_group(pgid, term_sig)
        time.sleep(grace_seconds)
        for pgid in pgids: _kill_group(pgid, kill_sig)
        with self._lock: self._pids.clear()

    async def kill_all_async(self) -> None:
        """Async-friendly version of kill_all that does not block the event loop."""
        import asyncio
        await asyncio.to_thread(self.kill_all)


