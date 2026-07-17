from __future__ import annotations
"""Live model presence, derived from the event bus.

Tracks how many agents each alias currently has in flight (``agent_started`` /
``agent_completed`` / ``agent_failed`` all carry an ``alias``) and when it last
finished work, so the fleet dashboard can show each character as busy, available,
idle, or offline. Pure state + thread-safe counters; ``now`` is a parameter so the
idle logic is deterministic in tests.
"""
import threading
import time
from typing import Optional

_BUSY = "busy"
_AVAILABLE = "available"
_IDLE = "idle"
_OFFLINE = "offline"


class PresenceService:
    """In-memory presence tracker fed by the event bus."""

    def __init__(self, *, idle_after_secs: float = 300.0):
        self._idle_after = idle_after_secs
        self._inflight: dict[str, int] = {}
        self._last_finished: dict[str, float] = {}
        self._lock = threading.Lock()
        self._subscribed = False

    # ── state mutations (called from bus handlers) ───────────────────────────
    def record_started(self, alias: str) -> None:
        if not alias:
            return
        with self._lock:
            self._inflight[alias] = self._inflight.get(alias, 0) + 1

    def record_finished(self, alias: str, *, now: Optional[float] = None) -> None:
        if not alias:
            return
        now = now if now is not None else time.time()
        with self._lock:
            self._inflight[alias] = max(0, self._inflight.get(alias, 0) - 1)
            self._last_finished[alias] = now

    # ── queries ──────────────────────────────────────────────────────────────
    def state_for(self, alias: str, *, available: bool, now: Optional[float] = None) -> str:
        now = now if now is not None else time.time()
        if not available:
            return _OFFLINE
        with self._lock:
            if self._inflight.get(alias, 0) > 0:
                return _BUSY
            last = self._last_finished.get(alias)
        if last is None or (now - last) > self._idle_after:
            return _IDLE
        return _AVAILABLE

    def snapshot(self, roster: list[tuple[str, bool]], *, now: Optional[float] = None) -> dict[str, str]:
        """Map alias -> state for a roster of (alias, available) pairs."""
        now = now if now is not None else time.time()
        return {alias: self.state_for(alias, available=available, now=now) for alias, available in roster}

    # ── bus wiring ───────────────────────────────────────────────────────────
    def _on_started(self, run_id: str, event: dict) -> None:
        self.record_started(str(event.get("alias", "")))

    def _on_finished(self, run_id: str, event: dict) -> None:
        self.record_finished(str(event.get("alias", "")))

    def subscribe(self) -> None:
        """Attach to the event bus (idempotent)."""
        if self._subscribed:
            return
        from orchestra.events import bus
        bus.subscribe("agent_started", self._on_started)
        bus.subscribe("agent_completed", self._on_finished)
        bus.subscribe("agent_failed", self._on_finished)
        self._subscribed = True


_service: Optional[PresenceService] = None
_service_lock = threading.Lock()


def get_presence() -> PresenceService:
    """Process-wide presence singleton, subscribed to the bus on first use."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                svc = PresenceService()
                svc.subscribe()
                _service = svc
    return _service
