"""Cross-process live state file for UIs.

Subscribes to the in-process event bus and maintains
``.orchestra/live_status.json`` so external UI processes can read run/agent
state without coupling into the engine internals.

Public API
----------
- ``install() -> None``   — idempotent; registers singleton writer on bus.
- ``note_stream(run_id, alias, chunk) -> None``  — engine stream callback.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orchestra import config
from orchestra.events import bus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRUNE_AFTER_SECONDS = 600  # 10 minutes

_STREAM_THROTTLE_SECONDS = 0.5

# ANSI CSI sequences (e.g. \x1b[31m) and OSC sequences (\x1b]...\x07 or ST).
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"     # CSI sequences
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences
)
# Non-printable control characters (except \n and \t which we keep for splitting)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Terminal statuses eligible for pruning.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_writer: Optional["LiveStatusWriter"] = None
_install_lock = threading.Lock()


# ---------------------------------------------------------------------------
# ANSI / text cleaning helpers
# ---------------------------------------------------------------------------

def _clean_chunk(chunk: str) -> str:
    """Strip ANSI escapes, control chars; return last non-empty line, ≤120 chars."""
    cleaned = _ANSI_RE.sub("", chunk)
    cleaned = _CONTROL_RE.sub("", cleaned)
    lines = [ln for ln in cleaned.split("\n") if ln.strip()]
    if not lines:
        return ""
    last = lines[-1].rstrip()
    return last[:120]


# ---------------------------------------------------------------------------
# LiveStatusWriter
# ---------------------------------------------------------------------------

class LiveStatusWriter:
    """Maintains .orchestra/live_status.json from bus events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {run_id: run_dict} — internal mutable state; never handed out directly.
        self._runs: dict[str, dict] = {}
        # Throttle: (run_id, alias) -> last monotonic write time.
        self._last_stream_write: dict[tuple[str, str], float] = {}
        # Coalesce buffer: (run_id, alias) -> newest cleaned last_line pending flush.
        self._pending_stream: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Bus handler (called from ThreadPoolExecutor — must be thread-safe)
    # ------------------------------------------------------------------

    def _handle(self, run_id: str, event: dict) -> None:
        """Dispatch incoming bus event to the appropriate handler method."""
        try:
            self._dispatch(run_id, event)
        except Exception:
            logger.exception("LiveStatusWriter._handle error for run=%s event=%s", run_id, event.get("event"))

    def _dispatch(self, run_id: str, event: dict) -> None:
        name = event.get("event", "")
        ts = event.get("ts") or datetime.now(timezone.utc).isoformat()

        with self._lock:
            changed = True  # most events change state
            if name == "run_started":
                self._on_run_started(run_id, event, ts)
            elif name == "run_finished":
                self._on_run_finished(run_id, event, ts)
            elif name == "run_cancelled":
                self._on_run_cancelled(run_id, ts)
            elif name == "agent_started":
                self._on_agent_event(run_id, event, ts, status="started")
            elif name in ("agent_retry", "agent_retrying"):
                self._on_agent_event(run_id, event, ts, status="retrying")
            elif name == "agent_failed":
                self._on_agent_event(run_id, event, ts, status="failed", terminal=True)
            elif name == "agent_completed":
                self._on_agent_event(run_id, event, ts, status="completed", terminal=True)
            elif name == "agent_cancelled":
                self._on_agent_event(run_id, event, ts, status="cancelled", terminal=True)
            elif name == "approval_requested":
                self._on_approval_requested(run_id, ts)
            elif name == "budget_exceeded":
                self._on_budget_exceeded(run_id, ts)
            else:
                # Unknown event: only update heartbeat if run is tracked.
                if run_id in self._runs:
                    self._runs[run_id] = {**self._runs[run_id], "heartbeat": ts}
                else:
                    changed = False

            # Apply any pending coalesced stream values for this run's agents.
            if changed:
                changed = self._flush_pending(run_id) or changed

            if changed:
                self._write()

    # ------------------------------------------------------------------
    # Event handlers (all called UNDER self._lock)
    # ------------------------------------------------------------------

    def _on_run_started(self, run_id: str, event: dict, ts: str) -> None:
        mode = event.get("mode") or "unknown"
        self._runs[run_id] = {
            "mode": mode,
            "status": "running",
            "started_at": ts,
            "heartbeat": ts,
            "agents": {},
        }

    def _on_run_finished(self, run_id: str, event: dict, ts: str) -> None:
        if run_id not in self._runs:
            return
        status = event.get("status") or "completed"
        run = dict(self._runs[run_id])
        run["status"] = status
        run["heartbeat"] = ts
        self._runs[run_id] = run

    def _on_run_cancelled(self, run_id: str, ts: str) -> None:
        if run_id not in self._runs:
            return
        run = dict(self._runs[run_id])
        run["status"] = "cancelled"
        run["heartbeat"] = ts
        self._runs[run_id] = run

    def _on_agent_event(
        self,
        run_id: str,
        event: dict,
        ts: str,
        *,
        status: str,
        terminal: bool = False,
    ) -> None:
        alias = event.get("alias")
        if not alias:
            raise ValueError(f"agent event missing 'alias': {event}")

        if run_id not in self._runs:
            return  # ignore agent events for untracked runs

        run = {**self._runs[run_id], "heartbeat": ts}

        agents = dict(run.get("agents", {}))
        existing = dict(agents.get(alias, {}))

        # Build new agent dict immutably.
        agent = {
            "status": status,
            "model": event.get("model") or existing.get("model"),
            "started_at": existing.get("started_at") if existing else (ts if status == "started" else None),
            "ended_at": ts if terminal else existing.get("ended_at"),
            "last_line": existing.get("last_line"),
            "est_cost_usd": (
                event.get("estimated_cost_usd")
                if event.get("estimated_cost_usd") is not None
                else existing.get("est_cost_usd")
            ),
        }
        # When first seen with "started" status, record started_at.
        if status == "started":
            agent["started_at"] = ts

        agents = {**agents, alias: agent}
        self._runs[run_id] = {**run, "agents": agents}

    def _on_approval_requested(self, run_id: str, ts: str) -> None:
        if run_id not in self._runs:
            return
        run = dict(self._runs[run_id])
        run["status"] = "waiting_approval"
        run["heartbeat"] = ts
        self._runs[run_id] = run

    def _on_budget_exceeded(self, run_id: str, ts: str) -> None:
        if run_id not in self._runs:
            return
        run = dict(self._runs[run_id])
        run["heartbeat"] = ts
        self._runs[run_id] = run

    def _flush_pending(self, run_id: str) -> bool:
        """Apply any coalesced pending stream values for agents of *run_id*.

        Returns True if at least one pending value was applied (caller holds self._lock).
        """
        if run_id not in self._runs:
            return False

        flushed = False
        run = self._runs[run_id]
        agents = dict(run.get("agents", {}))

        keys_for_run = [k for k in self._pending_stream if k[0] == run_id]
        for key in keys_for_run:
            alias = key[1]
            if alias not in agents:
                continue
            pending_line = self._pending_stream.pop(key)
            agent = {**agents[alias], "last_line": pending_line}
            agents = {**agents, alias: agent}
            self._last_stream_write[key] = time.monotonic()
            flushed = True

        if flushed:
            self._runs[run_id] = {**run, "agents": agents}
        return flushed

    # ------------------------------------------------------------------
    # Pruning + atomic write (called UNDER self._lock)
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove runs with terminal status whose heartbeat is older than PRUNE_AFTER_SECONDS."""
        now = datetime.now(timezone.utc)
        to_delete = []
        for run_id, run in self._runs.items():
            if run.get("status") not in _TERMINAL_STATUSES:
                continue
            hb_str = run.get("heartbeat")
            if not hb_str:
                continue
            try:
                hb = datetime.fromisoformat(hb_str)
                if (now - hb).total_seconds() > PRUNE_AFTER_SECONDS:
                    to_delete.append(run_id)
            except ValueError:
                pass
        for run_id in to_delete:
            del self._runs[run_id]

    def _write(self) -> None:
        """Serialize state to live_status.json atomically (caller holds self._lock)."""
        self._prune()

        status_path = self._status_path()
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": dict(self._runs),
        }
        text = json.dumps(payload, ensure_ascii=False)

        # Atomic write: temp file in same dir, then os.replace.
        status_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(status_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp_path, str(status_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _status_path() -> Path:
        """Resolve path lazily so config CWD is respected at call time."""
        return config.artifact_root().parent / "live_status.json"

    # ------------------------------------------------------------------
    # Stream callback support
    # ------------------------------------------------------------------

    def note_stream(self, run_id: str, alias: str, chunk: str) -> None:
        """Update agent's last_line from a streaming chunk (throttled, coalescing)."""
        last_line = _clean_chunk(chunk)
        if not last_line:
            return

        now = time.monotonic()
        key = (run_id, alias)

        with self._lock:
            last_write = self._last_stream_write.get(key, 0.0)
            if now - last_write < _STREAM_THROTTLE_SECONDS:
                # Throttled — coalesce: keep the newest value for later flush.
                self._pending_stream[key] = last_line
                return

            if run_id not in self._runs:
                return
            if alias not in self._runs[run_id].get("agents", {}):
                return

            # Discard any older pending value; we're writing the live chunk now.
            self._pending_stream.pop(key, None)

            # Build new agent dict with updated last_line.
            run = self._runs[run_id]
            agents = dict(run.get("agents", {}))
            agent = {**dict(agents.get(alias, {})), "last_line": last_line}
            agents = {**agents, alias: agent}
            self._runs[run_id] = {**run, "agents": agents}

            self._last_stream_write[key] = now
            self._write()


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

def install() -> None:
    """Idempotent: register singleton writer on the event bus (at most once)."""
    global _writer
    with _install_lock:
        if _writer is not None:
            return
        writer = LiveStatusWriter()
        bus.subscribe_all(writer._handle)
        _writer = writer


def note_stream(run_id: str, alias: str, chunk: str) -> None:
    """Module-level convenience for engine stream callbacks.

    No-op if install() has not been called.
    """
    if _writer is not None:
        _writer.note_stream(run_id, alias, chunk)
